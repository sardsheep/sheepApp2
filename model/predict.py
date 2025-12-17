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
    return service.files().create(
        body=meta, media_body=media, fields="id,name,webViewLink", supportsAllDrives=True
    ).execute()


def upload_update(service, file_id: str, local_path: str, new_name: str | None = None):
    meta = {"name": new_name} if new_name else None
    media = MediaIoBaseUpload(io.FileIO(local_path, "rb"), mimetype="text/csv", resumable=True)
    return service.files().update(
        fileId=file_id, body=meta, media_body=media, fields="id,name,webViewLink"
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


def engineer_features_like_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["sheep number", "video_time", "real Time"], inplace=True, errors="ignore")
    df.rename(columns={"behaivour": "behaviour"}, inplace=True)
    if "behaviour" in df.columns:
        df = df.drop(columns=["behaviour"])
    for i in range(1, 31):
        df[f"mag_{i}"] = np.sqrt(df[f"x_{i}"]**2 + df[f"y_{i}"]**2 + df[f"z_{i}"]**2)
    for i in range(1, 30):
        df[f"delta_mag_{i}"] = (df[f"mag_{i+1}"] - df[f"mag_{i}"]).abs()
    mag_cols   = [f"mag_{i}" for i in range(1, 31)]
    delta_cols = [f"delta_mag_{i}" for i in range(1, 30)]
    df["mag_mean"]   = df[mag_cols].mean(axis=1)
    df["mag_std"]    = df[mag_cols].std(axis=1)
    df["delta_mean"] = df[delta_cols].mean(axis=1)
    df["delta_std"]  = df[delta_cols].std(axis=1)
    return df


def main():
    # ---- Inputs (from env) ----
    folder_id        = os.getenv("INPUT_DRIVE_FOLDER_ID")
    csv_filename     = os.getenv("INPUT_CSV_FILENAME", "").strip()
    model_path       = os.getenv("INPUT_MODEL_PATH")
    class_labels_env = [c.strip() for c in os.getenv("INPUT_CLASS_LABELS", "").split(",") if c.strip()]
    output_mode      = os.getenv("OUTPUT_MODE", "update").lower()  # 'update' | 'create'

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

    # ---- Load preprocessing bundle (ONE file), or auto-build it ----
    model_dir = Path(model_path).parent
    preproc_path = os.getenv("PREPROC_PATH", str(model_dir / "preproc.joblib"))

    if not os.path.exists(preproc_path):
        # Try to build from known artifact locations
        candidates = [
            ("model/artifacts/scaler.joblib", "model/artifacts/feature_cols.joblib", "model/artifacts/label_encoder.joblib"),
            ("artifacts/scaler.joblib",       "artifacts/feature_cols.joblib",       "artifacts/label_encoder.joblib"),
        ]
        built = False
        for s_path, f_path, l_path in candidates:
            if os.path.exists(s_path) and os.path.exists(f_path):
                scaler = joblib.load(s_path)
                feature_cols = joblib.load(f_path)
                le = joblib.load(l_path) if os.path.exists(l_path) else None
                Path(preproc_path).parent.mkdir(parents=True, exist_ok=True)
                joblib.dump({"scaler": scaler, "feature_cols": feature_cols, "label_encoder": le}, preproc_path)
                print(f"Auto-created preprocessing bundle at {preproc_path} from '{s_path}' and '{f_path}'.")
                built = True
                break
        if not built:
            import glob
            print("Current working dir:", os.getcwd())
            print("Top-level entries:", os.listdir("."))
            print("*.joblib found:", glob.glob("**/*.joblib", recursive=True))
            raise FileNotFoundError(
                f"Missing preprocessing bundle at '{preproc_path}' and could not auto-build it."
            )

    # Load bundle
    bundle = joblib.load(preproc_path)
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    le = bundle.get("label_encoder")

    # ---- Recreate training-time features on the new CSV ----
    df = df_raw.copy()
    df.drop(columns=["sheep number", "video_time", "real Time"], inplace=True, errors="ignore")
    df.rename(columns={"behaivour": "behaviour"}, inplace=True)
    if "behaviour" in df.columns:
        df.drop(columns=["behaviour"], inplace=True)

    # Infer T from feature_cols (e.g., x_1..x_30)
    def max_t_from_feats(cols):
        ts = []
        for c in cols:
            m = re.match(r"^[xyz]_(\d+)$", c)
            if m:
                ts.append(int(m.group(1)))
        return max(ts) if ts else 0

    T_expected = max_t_from_feats(feature_cols)
    if T_expected <= 0:
        raise RuntimeError("Could not infer sequence length T from feature_cols in preproc bundle.")

    # Check required base columns exist in the CSV
    missing_base = [c for i in range(1, T_expected+1) for c in (f"x_{i}", f"y_{i}", f"z_{i}") if c not in df.columns]
    if missing_base:
        preview = ", ".join(missing_base[:6])
        raise KeyError(f"CSV is missing required accelerometer columns from training: {preview}"
                       f"{' ...' if len(missing_base)>6 else ''}")

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

    # ---- Align order exactly as training and scale ----
    X_flat = df[feature_cols].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X_flat)

    # ---- Build (N, T, 5) sequence from the *scaled* vector ----
    idx_cache = {name: i for i, name in enumerate(feature_cols)}
    X_seq = []
    for row in X_scaled:
        sample = []
        for i in range(1, T_expected+1):
            x = row[idx_cache[f"x_{i}"]]
            yv = row[idx_cache[f"y_{i}"]]
            z  = row[idx_cache[f"z_{i}"]]
            m  = row[idx_cache[f"mag_{i}"]]
            d  = row[idx_cache.get(f"delta_mag_{i}", -1)] if i < T_expected and f"delta_mag_{i}" in idx_cache else 0.0
            sample.append([x, yv, z, m, d])
        X_seq.append(sample)
    X = np.asarray(X_seq, dtype=np.float32)
    print(f"Prepared input tensor: {X.shape} (N, T={T_expected}, 5)")

    # ---- Load model & predict ----
    print(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        print("Current working dir:", os.getcwd())
        raise FileNotFoundError(f"Model file not found at '{model_path}'.")
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
    base, _ext = os.path.splitext(file_name)
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
            updated = upload_update(service, file_id, out_name, None)
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
