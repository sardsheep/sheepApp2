import os, sys, pandas as pd, pytz
from pathlib import Path
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write.point import WritePrecision

URL    = os.getenv("INFLUX_URL")
TOKEN  = os.getenv("INFLUX_TOKEN")   # WRITE token
ORG    = os.getenv("INFLUX_ORG")
BUCKET = os.getenv("INFLUX_BUCKET")
MODEL_VERSION = os.getenv("MODEL_VERSION","drive-to-influx")


if not (URL and TOKEN and ORG and BUCKET):
    print("Missing INFLUX_URL / INFLUX_TOKEN / INFLUX_ORG / INFLUX_BUCKET", file=sys.stderr)
    sys.exit(1)

def find_ts_col(df):
    for c in ["timestamp","time", "Time","real Time","real_time","video_time"]:
        if c in df.columns: return c
    return None

def parse_ts(series):
    s = pd.to_datetime(series, errors="coerce", utc=False, infer_datetime_format=True)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(pytz.timezone("Europe/Rome"), nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s

def to_points(df: pd.DataFrame, src_name: str):
    tcol = find_ts_col(df)
    if not tcol:
        raise ValueError("No timestamp column found (expected one of: timestamp, time, real Time, real_time, video_time).")
    t = parse_ts(df[tcol])
    if t.isna().all():
        raise ValueError(f"Could not parse timestamps from '{tcol}'.")

    sheep_col = next((c for c in ["sheep number","sheep_id","sheep"] if c in df.columns), None)

        # NEW: support both 'type' and 'Type' (case-tolerant)
    type_col = next((c for c in ["type", "Type", "sheep_type"] if c in df.columns), None)


    if "predict" not in df.columns:    raise ValueError("Missing 'predict' column.")
    if "confidence" not in df.columns: raise ValueError("Missing 'confidence' column.")

    for i, row in df.iterrows():
        ts = t.iloc[i]
        if pd.isna(ts): continue
        p = Point("sheep_behavior_pred") \
              .tag("label", str(row["predict"])) \
              .tag("source", src_name) \
              .tag("model_version", MODEL_VERSION) \
              .field("confidence", float(row["confidence"])) \
              .time(ts.to_pydatetime(), WritePrecision.NS)
        if sheep_col and pd.notna(row[sheep_col]):
            p = p.tag("sheep_id", str(row[sheep_col]))

        # ✅ NEW: write 'type' as a tag (only if present and not NaN)
        if type_col:
            val = str(row[type_col]).strip() if pd.notna(row[type_col]) else ""
            if val:  # only add non-empty strings
                p = p.tag("type", val.lower())

        
        yield p

def ingest(paths):
    with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
        write_api = client.write_api(write_options=WriteOptions(batch_size=5000, flush_interval=5000))
        for pth in paths:
            df = pd.read_csv(pth)


            



            
            recs = list(to_points(df, Path(pth).name))
            if recs:
                write_api.write(bucket=BUCKET, record=recs)
                print(f"Wrote {len(recs)} points from {pth}")
        write_api.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/ingest_influx.py file1.csv [file2.csv ...]"); sys.exit(2)
    ingest(sys.argv[1:])
