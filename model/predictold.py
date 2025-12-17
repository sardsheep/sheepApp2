import os, io, re, json, base64
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.errors import HttpError

from tensorflow.keras.models import load_model

scaler_path = Path(os.getenv("SCALER_PATH", "artifacts/scaler.joblib"))

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def drive_client(sa_path: str = "sa.json"):
    b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    if b64:
        info = json.loads(base64.b64decode(b64))
        creds = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
    else:
        creds = service_account.Credentials.from_service_account_file(sa_path, scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds)


def find_csv_in_folder(service, folder_id: str, filename: str = ""):
    if filename:
        q = f"'{folder_id}' in parents and name = '{filename}' and mimeType = 'text/csv' and trashed = false"
    else:
        q = f"'{folder_id}' in parents and mimeType = 'text/csv' and trashed = false"
    resp = service.files().list(
        q=q, orderBy="modifiedTime desc",
        fields="files(id,name,modifiedTime,size,mimeType,parents)"
    ).execute()
    files = resp.get("files", [])
    if not files:
        raise FileNotFoundError("No matching CSV files found in the specified folder.")
    f = files[0]
    return f["id"], f["name"]


def download_file(service, file_id: str, local_path: str):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(local_path, mode="wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()


def upload_create(service, folder_id: str, local_path: str, upload_name: str):
    meta = {"name": upload_name, "parents": [folder_id]}
    media = MediaIoBaseUpload(io.FileIO(local_path, "rb"), mimetype="text/csv", resumable=True)
    # supportsAllDrives helps when folder is a Shared Drive
    return service.files().create(
        body=meta, media_body=media, fields="id,name,webViewLink", supportsAllDrives=True
    ).execute()


def upload_update(service, file_id: str, local_path: str, new_name: str | None = None):
    meta = {"name": new_name} if new_name else None
    media = MediaIoBaseUpload(io.FileIO(local_path, "rb"), mimetype="text/csv", resumable=True)
    return service.files().update(
        fileId=file_id, body=meta, media_body=media,
        fields="id,name,webViewLink"
    ).execute()


def detect_xyz_columns(columns):
    if not any(re.fullmatch(r"[xyz]_\d+", c) for c in columns):
        return 0, [], [], []
    T, xs, ys, zs = 0, [], [], []
    i = 1
    while True:
        x, y, z = f"x_{i}", f"y_{i}", f"z_{i}"
        if x in columns and y in columns and z in columns:
            xs.append(x); ys.append(y); zs.append(z)
            T += 1; i += 1
        else:
            break
    return T, xs, ys, zs


def build_input_tensor(df: pd.DataFrame):
    for c in ["sheep number", "real Time"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    cols = list(df.columns)
    T, xs, ys, zs = detect_xyz_columns(cols)
    if T < 3:
        raise ValueError("Could not auto-detect x_i,y_i,z_i columns like x_1..z_T.")

    N = len(df)
    Xx = df[xs].to_numpy(dtype=np.float32).reshape(N, T)
    Xy = df[ys].to_numpy(dtype=np.float32).reshape(N, T)
    Xz = df[zs].to_numpy(dtype=np.float32).reshape(N, T)

    # Build 5 channels: x, y, z, magnitude, delta_magnitude
    mag = np.sqrt(Xx**2 + Xy**2 + Xz**2)
    dmag = np.concatenate([np.zeros((N, 1), np.float32), np.diff(mag, axis=1).astype(np.float32)], axis=1)

    X = np.stack([Xx, Xy, Xz, mag, dmag], axis=2).astype(np.float32)  # [N, T, 5]
    return X, T


# --- New: recreate training-time features on an unlabeled df ---
def engineer_features_like_training(df: pd.DataFrame) -> pd.DataFrame:
    # Match your training drops/renames
    df = df.copy()
    df.drop(columns=["sheep number", "video_time", "real Time"], inplace=True, errors="ignore")
    df.rename(columns={"behaivour": "behaviour"}, inplace=True)  # harmless if absent
    if "behaviour" in df.columns:
        df = df.drop(columns=["behaviour"])  # inference has no labels

    # Magnitude per step
    for i in range(1, 31):
        df[f"mag_{i}"] = np.sqrt(df[f"x_{i}"]**2 + df[f"y_{i}"]**2 + df[f"z_{i}"]**2)

    # Delta magnitude per step (last step = 0)
    for i in range(1, 30):
        df[f"delta_mag_{i}"] = (df[f"mag_{i+1}"] - df[f"mag_{i}"]).abs()

    # Aggregates (training had these)
    mag_cols   = [f"mag_{i}" for i in range(1, 31)]
    delta_cols = [f"delta_mag_{i}" for i in range(1, 30)]
    df["mag_mean"]   = df[mag_cols].mean(axis=1)
    df["mag_std"]    = df[mag_cols].std(axis=1)
    df["delta_mean"] = df[delta_cols].mean(axis=1)
    df["delta_std"]  = df[delta_cols].std(axis=1)

    return df

# --- New: build sequence from *scaled* flat vector using saved feature_cols order ---
def to_sequence_from_scaled(X_scaled: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    X_seq = []
    for row in X_scaled:
        sample = []
        for i in range(1, 31):
            x_idx = feature_cols.index(f"x_{i}")
            y_idx = feature_cols.index(f"y_{i}")
            z_idx = feature_cols.index(f"z_{i}")
            m_idx = feature_cols.index(f"mag_{i}")
            d = row[feature_cols.index(f"delta_mag_{i}")] if i < 30 else 0.0
            sample.append([row[x_idx], row[y_idx], row[z_idx], row[m_idx], d])
        X_seq.append(sample)
    return np.asarray(X_seq, dtype=np.float32)




def main():
    import os
    import numpy as np
    import pandas as pd
    import joblib
    from tensorflow.keras.models import load_model

    # ---- Inputs (from env) ----
    folder_id        = os.getenv("INPUT_DRIVE_FOLDER_ID")
    csv_filename     = os.getenv("INPUT_CSV_FILENAME", "").strip()
    model_path       = os.getenv("INPUT_MODEL_PATH")
    class_labels_env = [c.strip() for c in os.getenv("INPUT_CLASS_LABELS", "").split(",") if c.strip()]
    output_mode      = os.getenv("OUTPUT_MODE", "update").lower()  # 'update' | 'create'

    scaler_path      = os.getenv("INPUT_SCALER_PATH", "artifacts/scaler.joblib")
    featcols_path    = os.getenv("INPUT_FEATURECOLS_PATH", "artifacts/feature_cols.joblib")
    labelenc_path    = os.getenv("INPUT_LABELENC_PATH", "artifacts/label_encoder.joblib")  # optional

    # ---- Validate required inputs ----
    if not folder_id:
        raise RuntimeError("INPUT_DRIVE_FOLDER_ID is required.")
    if not model_path:
        raise RuntimeError("INPUT_MODEL_PATH is required (e.g., model/ram_blstm_model.h5).")

    # ---- Drive: find & download CSV ----
    service = drive_client()
    file_id, file_name = find_csv_in_folder(service, folder_id, csv_filename)
    print(f"Using CSV: {file_name} (id={file_id})")

    in_csv = "input.csv"
    download_file(service, file_id, in_csv)

    # ---- Load raw data ----
    df_raw = pd.read_csv(in_csv)
    df_out = df_raw.copy()

    # ---- Load artifacts (must match training) ----
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler at '{scaler_path}'. Save it during training with joblib.dump(scaler, ...).")
    if not os.path.exists(featcols_path):
        raise FileNotFoundError(f"Missing feature_cols at '{featcols_path}'. Save it during training (df.drop(['behaviour'],1).columns.tolist()).")

    scaler       = joblib.load(scaler_path)
    feature_cols = joblib.load(featcols_path)
    le = joblib.load(labelenc_path) if os.path.exists(labelenc_path) else None

    # ---- Recreate training-time features on the new CSV ----
    df = df_raw.copy()
    # Same drops/renames as training
    df.drop(columns=["sheep number", "video_time", "real Time"], inplace=True, errors="ignore")
    df.rename(columns={"behaivour": "behaviour"}, inplace=True)
    if "behaviour" in df.columns:
        df.drop(columns=["behaviour"], inplace=True)

    # Determine expected T from training feature list (e.g., x_1..x_30)
    def max_t_from_feats(cols):
        import re
        ts = []
        for c in cols:
            m = re.match(r"^[xyz]_(\d+)$", c)
            if m:
                ts.append(int(m.group(1)))
        return max(ts) if ts else 0

    T_expected = max_t_from_feats(feature_cols)
    if T_expected <= 0:
        raise RuntimeError("Could not infer sequence length T from feature_cols. Check your saved artifacts.")

    # Check required base columns exist in the CSV
    missing_base = [c for i in range(1, T_expected+1) for c in (f"x_{i}", f"y_{i}", f"z_{i}") if c not in df.columns]
    if missing_base:
        raise KeyError(f"CSV is missing required accelerometer columns from training: {missing_base[:6]}{'...' if len(missing_base)>6 else ''}")

    # Magnitude per step
    for i in range(1, T_expected+1):
        df[f"mag_{i}"] = np.sqrt(df[f"x_{i}"]**2 + df[f"y_{i}"]**2 + df[f"z_{i}"]**2)

    # Delta magnitude (last step = 0 in training code)
    for i in range(1, T_expected):
        df[f"delta_mag_{i}"] = (df[f"mag_{i+1}"] - df[f"mag_{i}"]).abs()

    # Aggregates (must match training)
    mag_cols   = [f"mag_{i}" for i in range(1, T_expected+1)]
    delta_cols = [f"delta_mag_{i}" for i in range(1, T_expected)]
    df["mag_mean"]   = df[mag_cols].mean(axis=1)
    df["mag_std"]    = df[mag_cols].std(axis=1)
    df["delta_mean"] = df[delta_cols].mean(axis=1) if delta_cols else 0.0
    df["delta_std"]  = df[delta_cols].std(axis=1)  if delta_cols else 0.0

    # ---- Align column order exactly as training and scale ----
    # This will raise if the CSV doesn't have a column expected by training (which is good!)
    X_flat = df[feature_cols].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X_flat)

    # ---- Build (N, T, 5) sequence from the *scaled* vector using feature_cols indices ----
    idx_cache = {name: feature_cols.index(name) for name in feature_cols}  # speed-up
    X_seq = []
    for row in X_scaled:
        sample = []
        for i in range(1, T_expected+1):
            x = row[idx_cache[f"x_{i}"]]
            yv = row[idx_cache[f"y_{i}"]]
            z  = row[idx_cache[f"z_{i}"]]
            m  = row[idx_cache[f"mag_{i}"]]
            d  = row[idx_cache[f"delta_mag_{i}"]] if i < T_expected and f"delta_mag_{i}" in idx_cache else 0.0
            sample.append([x, yv, z, m, d])
        X_seq.append(sample)
    X = np.asarray(X_seq, dtype=np.float32)
    print(f"Prepared input tensor: {X.shape} (N, T={T_expected}, 5)")

    # ---- Load model & predict ----
    print(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        print("Current working dir:", os.getcwd())
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Check workflow input 'model_path'.")
    model = load_model(model_path)
    raw = model.predict(X, verbose=0)

    # ---- Map predictions to labels (+ confidence) ----
    if raw.ndim == 2 and raw.shape[1] > 1:
        n_classes = raw.shape[1]
        idx = np.argmax(raw, axis=1)
        conf = raw.max(axis=1)
        if le is not None and len(le.classes_) == n_classes:
            labels = [str(le.classes_[i]) for i in idx]
        elif class_labels_env and len(class_labels_env) == n_classes:
            labels = [class_labels_env[i] for i in idx]
        else:
            if class_labels_env and len(class_labels_env) != n_classes:
                print(f"Provided class_labels ({len(class_labels_env)}) != model outputs ({n_classes}); falling back to class_#.")
            labels = [f"class_{i}" for i in idx]
    else:
        # Binary fallback
        pred = raw.ravel()
        idx = (pred >= 0.5).astype(int)
        conf = np.where(idx == 1, pred, 1.0 - pred)
        if class_labels_env and len(class_labels_env) >= 2:
            labels = [class_labels_env[i] for i in idx]
        elif le is not None and len(le.classes_) >= 2:
            labels = [str(le.classes_[i]) for i in idx]
        else:
            labels = [f"class_{i}" for i in idx]

    # ---- Save predictions locally ----
    base, ext = os.path.splitext(file_name)
    out_name = f"{base}_predicted.csv"
    df_out["predict"] = labels
    df_out["confidence"] = conf
    df_out.to_csv(out_name, index=False)
    print(f"Saved predictions to {out_name}")

    # ---- Upload back to Drive ----
    try:
        if output_mode == "create":
            print("OUTPUT_MODE=create → attempting to create a new file...")
            created = upload_create(service, folder_id, out_name, out_name)
            print(f"Uploaded (create): {created.get('name')} (id={created.get('id')})")
            print(f"Web link: {created.get('webViewLink')}")
        else:
            print("OUTPUT_MODE=update → overwriting the original file content...")
            updated = upload_update(service, file_id, out_name, None)  # keep same name
            print(f"Uploaded (update): {updated.get('name')} (id={updated.get('id')})")
            print(f"Web link: {updated.get('webViewLink')}")
    except HttpError as e:
        if output_mode == "create" and e.resp.status == 403 and b"storageQuotaExceeded" in e.content:
            print("Create failed due to service account quota. Falling back to update-in-place...")
            updated = upload_update(service, file_id, out_name, None)
            print(f"Uploaded (update fallback): {updated.get('name')} (id={updated.get('id')})")
            print(f"Web link: {updated.get('webViewLink')}")
        else:
            raise



if __name__ == "__main__":
    main()
