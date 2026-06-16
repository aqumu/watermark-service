import argparse
import time
from pathlib import Path
import httpx

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--url", default="http://localhost:8000/api/v1")
parser.add_argument("--mode", default="old", choices=["old", "new"])
parser.add_argument("-o", "--output", default="results.zip")
parser.add_argument("--timeout", type=int, default=3600)
args = parser.parse_args()

files = sorted(args.folder.glob("*.[jJ][pP][gG]")) + sorted(args.folder.glob("*.[jJ][pP][eE][gG]")) + sorted(args.folder.glob("*.[pP][nN][gG]"))
if not files:
    print(f"No images found in {args.folder}")
    exit(1)

print(f"Uploading {len(files)} images...")

with httpx.Client(timeout=httpx.Timeout(args.timeout)) as client:
    # Submit async job
    resp = client.post(f"{args.url}/process/batch/async?mode={args.mode}", files=[("images", f.open("rb")) for f in files])
    resp.raise_for_status()
    job = resp.json()
    job_id = job["job_id"]
    print(f"Job submitted: {job_id}")

    # Poll until done
    while True:
        resp = client.get(f"{args.url}/jobs/{job_id}")
        resp.raise_for_status()
        state = resp.json()
        status = state["status"]
        done = state.get("completed", 0) + state.get("failed", 0)
        total = state.get("total", 0)
        print(f"  {status} — {done}/{total}", end="\r")
        if status == "completed":
            print()
            break
        if status == "failed":
            print(f"\nJob failed: {state}")
            exit(1)
        time.sleep(2)

    # Download results
    resp = client.get(f"{args.url}/jobs/{job_id}/results")
    resp.raise_for_status()
    Path(args.output).write_bytes(resp.content)
    print(f"Done — saved to {args.output}")
