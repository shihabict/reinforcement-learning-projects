import os
import re

def get_next_run_dir(base_dir="runs_lka", prefix="run_"):
    os.makedirs(base_dir, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    existing = []

    for name in os.listdir(base_dir):
        m = pattern.match(name)
        if m:
            existing.append(int(m.group(1)))

    next_id = (max(existing) + 1) if existing else 1
    run_name = f"{prefix}{next_id:03d}"   # run_001, run_002, ...
    run_dir = os.path.join(base_dir, run_name)

    # subfolders
    log_dir = os.path.join(run_dir, "logs")
    model_dir = os.path.join(run_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    return run_dir, log_dir, model_dir, next_id