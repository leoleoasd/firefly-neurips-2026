#!/usr/bin/env python3
"""Stop the running train_async.py Ray job."""

from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient(address="auto")
jobs = client.list_jobs()

matches = [
    j
    for j in jobs
    if j.status.value == "RUNNING" and j.submission_id is not None and "train_async.py" in (j.entrypoint or "")
]

if not matches:
    print("No running train_async.py job found.")
    exit(0)

for j in matches:
    print(f"Stopping {j.submission_id}  {j.entrypoint[:100]}")
    try:
        client.stop_job(j.submission_id)
        print("  Stopped.")
    except Exception as e:
        print(f"  Failed: {e}")
