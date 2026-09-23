#!/usr/bin/env python3
"""Stop all running Ray jobs without stopping the Ray cluster."""

from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient(address="auto")
jobs = client.list_jobs()

running = [j for j in jobs if j.status.value == "RUNNING" and j.submission_id is not None]

if not running:
    print("No running jobs found.")
    exit(0)

print(f"Found {len(running)} running job(s):")
for j in running:
    print(f"  {j.submission_id}  {j.entrypoint[:80]}...")

for j in running:
    print(f"Stopping {j.submission_id}...")
    try:
        client.stop_job(j.submission_id)
    except Exception as e:
        print(f"  Failed: {e}")

print("Done.")
