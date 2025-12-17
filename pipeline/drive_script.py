#!/usr/bin/env python3
# pipeline/drive_script.py

import os
import io
import json
import argparse
from pathlib import Path

import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError


# =========================
# Google Drive utilities
# =========================

def get_drive_service():
    """Build Drive v3 client using a service account in GDRIVE_CREDENTIALS env var."""
    creds_dict = json.loads(os.environ["GDRIVE_CREDENTIALS"])
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


def get_sa_email():
    """Return the service account email for clear permission error messages."""
    return json.loads(os.environ["GDRIVE_CREDENTIALS"])["client_email"]


def resolve_file(service, file_id):
    """
    Resolve a file ID to its real target, handling Drive shortcuts.
    Returns (real_file_id, metadata).
    """
    meta = service.files().get(
        fileId=file_id,
        fields=("id,name,mimeType,parents,owners(emailAddress),driveId,"
                "shortcutDetails(targetId,targetMimeType)"),
        supportsAllDrives=True,
    ).execute()

    if meta.get("mimeType") == "application/vnd.google-apps.shortcut":
        target_id = meta["shortcutDetails"]["targetId"]
        meta = service.files().get(
            fileId=target_id,
            fields="id,name,mimeType,parents,owners(emailAddress),driveId",
            supportsAllDrives=True,
        ).execute()
        file_id = meta["id"]

    return file_id, meta


def download_file(service, file_id, local_path: Path):
    """
    Download a Drive file to local_path.
    - If it's a Google Sheet, export as CSV.
    - Else download the file content (works for CSV/XLSX/etc.).
    """
    try:
        real_id, meta = resolve_file(service, file_id)
        mime = meta["mimeType"]
        name = meta["name"]
        print(f"📎 Resolved file: name='{name}' id={real_id} mime={mime}", flush=True)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        fh = io.FileIO(str(local_path), "wb")

        if mime == "application/vnd.google-apps.spreadsheet":
            # Export Google Sheet as CSV
            request = service.files().export_media(
                fileId=real_id, mimeType="text/csv"
            )
        else:
            # Direct download
            request = service.files().get_media(fileId=real_id)

        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        print(f"✅ Downloaded to {local_path}", flush=True)

    except HttpError as e:
        sa = get_sa_email()
        print(f"❌ Drive error while downloading id={file_id}: {e}", flush=True)
        print("ℹ️ Common causes:", flush=True)
        print(f"   • The service account ({sa}) does NOT have access to the file/folder.", flush=True)
        print("   • The ID is a shortcut you can’t follow due to permissions.", flush=True)
        print("   • The file was moved/trashed.", flush=True)
        raise


def upload_file(service, local_path: Path, file_id: str):
    """
    Overwrite the original Drive file (dangerous if you want to keep the raw CSV).
    """
    media = MediaFileUpload(str(local_path), mimetype="text/csv", resumable=True)
    updated = service.files().update(
        fileId=file_id,
        media_body=media,
        supportsAllDrives=True,
        fields="id,name"
    ).execute()
    print(f"☁️ Overwrote Drive file: {updated['name']} (id={updated['id']})", flush=True)


def upload_new_file(service, local_path: Path, source_file_id: str):
    """
    Create a NEW file named '<source_stem>__flattened.csv' in the same folder(s) as the source.
    Keeps the original intact (recommended).
    """
    src = service.files().get(
        fileId=source_file_id,
        fields="name,parents",
        supportsAllDrives=True
    ).execute()
    base = Path(src["name"]).stem
    new_name = f"{base}__flattened.csv"
    file_metadata = {"name": new_name}

    # If we know the parents, place the new file alongside the source.
    parents = src.get("parents")
    if parents:
        file_metadata["parents"] = parents

    media = MediaFileUpload(str(local_path), mimetype="text/csv", resumable=True)
    created = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id,name",
        supportsAllDrives=True,
    ).execute()
    print(f"☁️ Uploaded new file to Drive: {created['name']} (id={created['id']})", flush=True)


# =========================
# Flatten CSV logic
# =========================


def find_col(df, target, fallbacks, required=True):
    cols = {c.lower(): c for c in df.columns}
    for name in [target, *fallbacks]:
        c = cols.get(name.lower())
        if c:
            return c
    if required:
        raise ValueError(f"Required column '{target}' not found. Available: {list(df.columns)}")
    return None





def flatten_file(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 30000,
    window: int = 30,
    sheep_id_const: str | None = None,   # <-- add this
):
    print(f"📂 Flattening {input_path} → {output_path} (chunk={chunk_size}, window={window})", flush=True)

    if chunk_size % window != 0:
        raise ValueError(f"chunk_size ({chunk_size}) must be a multiple of window ({window}).")

    flattened_rows = []

    # Process file in chunks to keep memory low
    for chunk_index, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        print(f"🔹 Processing chunk {chunk_index+1} with {len(chunk)} rows, columns={list(chunk.columns)}", flush=True)

        try:
            x_col = find_col(chunk, "X", ["x", "acc_x", "accel_x"])
            y_col = find_col(chunk, "Y", ["y", "acc_y", "accel_y"])
            z_col = find_col(chunk, "Z", ["z", "acc_z", "accel_z"])
            t_col = find_col(chunk, "time", ["Time", "timestamp", "datetime"])
            type_col = find_col(chunk, "type", ["Type"])  # <-- added

            # NEW: optional sheep id column (auto-detect if present)
            sheep_col = find_col(
                chunk,
                "sheep_id",
                ["Sheep ID", "sheep number", "sheep", "id"],
                required=False
            )
     
        except ValueError as e:
            print(f"⚠️ Skipping chunk: {e}", flush=True)
            continue



        # keep only needed columns (don't force sheep to be non-null)
        cols_keep = [x_col, y_col, z_col, t_col, type_col] + ([sheep_col] if sheep_col else [])
        chunk = chunk[cols_keep].dropna(subset=[x_col, y_col, z_col, t_col, type_col])
        n = len(chunk)
        print(f"   After filtering, {n} rows remain", flush=True)
        if n < window:
            print(f"   ⚠️ Not enough rows for one full window (need {window}, got {n})", flush=True)
            continue

        
        # Convert to numpy for speed
        x = chunk[x_col].to_numpy()
        y = chunk[y_col].to_numpy()
        z = chunk[z_col].to_numpy()
        t = chunk[t_col].to_numpy()
        typ = chunk[type_col].to_numpy()

        if sheep_col:
            shp = chunk[sheep_col].astype("string").to_numpy()
        else:
            shp = None

        


        usable = (n // window) * window
        x, y, z, t, typ = x[:usable], y[:usable], z[:usable], t[:usable], typ[:usable]
        if shp is not None:
            shp = shp[:usable]


        
        # Reshape into (num_windows, window)
        xw = x.reshape(-1, window)
        yw = y.reshape(-1, window)
        zw = z.reshape(-1, window)
        tw = t.reshape(-1, window)
        typew = typ.reshape(-1, window)
        if shp is not None:
            sheepw = shp.reshape(-1, window)
        
        print(f"   ✅ Creating {xw.shape[0]} flattened rows", flush=True)

        # Build flattened rows
        for i in range(xw.shape[0]):
            row = {}
            for j in range(window):
                row[f"x_{j+1}"] = xw[i, j]
                row[f"y_{j+1}"] = yw[i, j]
                row[f"z_{j+1}"] = zw[i, j]
            # Use last time in window
            row["Time"] = tw[i, -1]
            row["type"] = typew[i, -1]


            # NEW: sheep_id (prefer source column's last value in the window, else constant)
            if shp is not None:
                row["sheep_id"] = str(sheepw[i, -1])
            elif sheep_id_const is not None:
                row["sheep_id"] = str(sheep_id_const)
            else:
                # still include the column (empty) so downstream schema is stable
                row["sheep_id"] = ""

            flattened_rows.append(row)


    if not flattened_rows:
        print("⚠️ No flattened rows created! Writing empty CSV with headers only.", flush=True)
        cols = [*(f"x_{i}" for i in range(1, window+1)),
                *(f"y_{i}" for i in range(1, window+1)),
                *(f"z_{i}" for i in range(1, window+1)),
                "Time", "type", "sheep_id"]   # <-- include sheep_id header
        pd.DataFrame(columns=cols).to_csv(output_path, index=False)
        return

    pd.DataFrame(flattened_rows).to_csv(output_path, index=False)
    print(f"✅ Flattened file saved: {output_path} with {len(flattened_rows)} rows", flush=True)

# =========================
# Main workflow
# =========================

def main():
    print("🚀 script started", flush=True)

    ap = argparse.ArgumentParser(
        description="Download CSV/Sheet from Drive, flatten, and upload back."
    )
    ap.add_argument("--file-id", required=True, help="Google Drive file ID")
    ap.add_argument("--chunk-size", type=int, default=30000,
                    help="Rows per chunk (multiple of window)")
    ap.add_argument("--window", type=int, default=30, help="Window size")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite original file instead of creating <name>__flattened.csv")
    ap.add_argument("--sheep-id-const", default=None,
                    help="Optional constant sheep_id to write when the source has no sheep column")
    args = ap.parse_args()

    file_id = args.file_id.strip()
    if not file_id:
        raise SystemExit("Missing --file-id")

    print(f"👤 Service account: {get_sa_email()}", flush=True)
    service = get_drive_service()

    work_dir = Path("work")
    work_dir.mkdir(parents=True, exist_ok=True)
    local_in = work_dir / "data.csv"
    local_out = work_dir / "data_flattened.csv"

    print(f"⬇️ Downloading file from Drive (file_id={file_id})...", flush=True)
    download_file(service, file_id, local_in)

    print("🔧 Flattening CSV...", flush=True)
    flatten_file(
        local_in, local_out,
        chunk_size=args.chunk_size,
        window=args.window,
        sheep_id_const=args.sheep_id_const
    )


    print("⬆️ Uploading flattened file back to Drive...", flush=True)
    if args.overwrite:
        upload_file(service, local_out, file_id)
    else:
        upload_new_file(service, local_out, file_id)

    print("✅ Done! Flattened CSV uploaded successfully.", flush=True)


if __name__ == "__main__":
    main()
