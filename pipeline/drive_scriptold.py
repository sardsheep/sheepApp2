#!/usr/bin/env python3
# drive_script.py

import os, io, json
from pathlib import Path
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import argparse

# -------------------------
# Google Drive utilities
# -------------------------

def get_drive_service():
    creds_dict = json.loads(os.environ["GDRIVE_CREDENTIALS"])
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def download_file(service, file_id, local_path):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    print(f"Downloaded {local_path}")

def upload_file(service, local_path, file_id):
    media = MediaFileUpload(local_path, mimetype="text/csv", resumable=True)
    updated = service.files().update(fileId=file_id, media_body=media).execute()
    print(f"Uploaded {updated['name']} to Drive")

# -------------------------
# Flatten CSV logic
# -------------------------

def find_col(df, target, fallbacks):
    cols = {c.lower(): c for c in df.columns}
    for name in [target, *fallbacks]:
        c = cols.get(name.lower())
        if c:
            return c
    raise ValueError(f"Required column '{target}' not found. Available: {list(df.columns)}")

def flatten_file(input_path: Path, output_path: Path, chunk_size: int = 30000, window: int = 30):
    print(f"📂 Flattening {input_path} → {output_path} (chunk={chunk_size}, window={window})")

    if chunk_size % window != 0:
        raise ValueError(f"chunk_size ({chunk_size}) must be a multiple of window ({window}).")

    flattened_rows = []

    # Process file in chunks to keep memory low
    for chunk_index, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        print(f"🔹 Processing chunk {chunk_index+1} with {len(chunk)} rows, columns={list(chunk.columns)}")

        try:
            x_col = find_col(chunk, "X", ["x", "acc_x", "accel_x"])
            y_col = find_col(chunk, "Y", ["y", "acc_y", "accel_y"])
            z_col = find_col(chunk, "Z", ["z", "acc_z", "accel_z"])
            t_col = find_col(chunk, "time", ["Time", "timestamp", "datetime"])
        except ValueError as e:
            print(f"⚠️ Skipping chunk: {e}")
            continue

        # Keep only needed columns
        chunk = chunk[[x_col, y_col, z_col, t_col]].dropna()
        n = len(chunk)
        print(f"   After filtering, {n} rows remain")

        if n < window:
            print(f"   ⚠️ Not enough rows for one full window (need {window}, got {n})")
            continue

        # Convert to numpy for speed
        x = chunk[x_col].to_numpy()
        y = chunk[y_col].to_numpy()
        z = chunk[z_col].to_numpy()
        t = chunk[t_col].to_numpy()

        usable = (n // window) * window
        x = x[:usable]; y = y[:usable]; z = z[:usable]; t = t[:usable]

        # Reshape into (num_windows, window)
        xw = x.reshape(-1, window)
        yw = y.reshape(-1, window)
        zw = z.reshape(-1, window)
        tw = t.reshape(-1, window)

        print(f"   ✅ Creating {xw.shape[0]} flattened rows")

        # Build flattened rows
        for i in range(xw.shape[0]):
            row = {}
            for j in range(window):
                row[f"x_{j+1}"] = xw[i, j]
                row[f"y_{j+1}"] = yw[i, j]
                row[f"z_{j+1}"] = zw[i, j]
            # Use last time in window
            row["Time"] = tw[i, -1]
            flattened_rows.append(row)

    if not flattened_rows:
        print("⚠️ No flattened rows created! Writing empty CSV with headers only.")
        cols = [*(f"x_{i}" for i in range(1, window+1)),
                *(f"y_{i}" for i in range(1, window+1)),
                *(f"z_{i}" for i in range(1, window+1)),
                "Time"]
        pd.DataFrame(columns=cols).to_csv(output_path, index=False)
        return

    pd.DataFrame(flattened_rows).to_csv(output_path, index=False)
    print(f"✅ Flattened file saved: {output_path} with {len(flattened_rows)} rows")
# -------------------------
# Main workflow
# -------------------------

def main():
    print("🚀 script started", flush=True)

    ap = argparse.ArgumentParser(
        description="Download CSV from Drive, flatten, and upload back."
    )
    ap.add_argument("--file-id", required=True, help="Google Drive file ID of CSV")
    ap.add_argument("--chunk-size", type=int, default=30000, help="Rows per chunk (multiple of window)")
    ap.add_argument("--window", type=int, default=30, help="Window size")
    args = ap.parse_args()

    file_id = args.file_id.strip()
    if not file_id:
        raise SystemExit("Missing --file-id")

    service = get_drive_service()

    work_dir = Path("work")
    work_dir.mkdir(parents=True, exist_ok=True)
    local_in = work_dir / "data.csv"
    local_out = work_dir / "data_flattened.csv"

    print(f"⬇️ Downloading file from Drive (file_id={file_id})...", flush=True)
    download_file(service, file_id, local_in)

    print("🔧 Flattening CSV...", flush=True)
    flatten_file(local_in, local_out, chunk_size=args.chunk_size, window=args.window)

    print("⬆️ Uploading flattened file back to Drive (overwriting original)...", flush=True)
    upload_file(service, local_out, file_id)

    print("✅ Done! Flattened CSV uploaded successfully.", flush=True)


if __name__ == "__main__":
    main()
