import os, io, sys
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
FILENAME  = os.environ.get("CSV_FILENAME","").strip()
CONTAINS  = os.environ.get("NAME_CONTAINS","").strip()
DOWNLOAD_ALL = os.environ.get("DOWNLOAD_ALL","false").lower() == "true"

creds = service_account.Credentials.from_service_account_file(
    "sa.json", scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
svc = build("drive","v3",credentials=creds)

if FILENAME:
    q = f"'{FOLDER_ID}' in parents and name = '{FILENAME}' and mimeType = 'text/csv' and trashed = false"
elif CONTAINS:
    q = f"'{FOLDER_ID}' in parents and name contains '{CONTAINS}' and mimeType = 'text/csv' and trashed = false"
else:
    q = f"'{FOLDER_ID}' in parents and mimeType = 'text/csv' and trashed = false"

resp = svc.files().list(q=q, orderBy="modifiedTime desc", fields="files(id,name,modifiedTime)").execute()
files = resp.get("files", [])
if not files:
    print("No matching CSV files found.", file=sys.stderr)
    sys.exit(1)

targets = files if DOWNLOAD_ALL else [files[0]]

with open("downloaded_list.txt","w") as out:
    for f in targets:
        req = svc.files().get_media(fileId=f["id"])
        with open(f["name"],"wb") as fh:
            dl = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
        print("DOWNLOADED::", f["name"])
        out.write(f["name"]+"\n")
