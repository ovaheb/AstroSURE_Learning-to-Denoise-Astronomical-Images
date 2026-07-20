#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from typing import List, Iterable


def read_paths(file_path: Path) -> List[str]:
    if not file_path.exists():
        return []
    lines = [line.strip() for line in file_path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def write_paths(file_path: Path, paths: List[str]) -> None:
    content = "\n".join(paths) + ("\n" if paths else "")
    file_path.write_text(content)


def normalize_path(path: str) -> str:
    p = Path(path)
    if p.name == "files" and p.parent.name.startswith("offline-run-"):
        return str(p.parent)
    return str(p)


def discover_offline_runs(base_dirs: Iterable[str]) -> List[str]:
    runs = []
    for base in base_dirs:
        base_path = Path(base)
        if not base_path.exists():
            continue
        for run_dir in base_path.glob("offline-run-*/"):
            if any(run_dir.glob("run-*.wandb")):
                runs.append(str(run_dir))
            else:
                # Fallback: if any .wandb file exists inside, still include
                if list(run_dir.glob("*.wandb")):
                    runs.append(str(run_dir))
    return runs


def sync_run(path: str, wandb_args: List[str], entity: str | None, project: str | None) -> bool:
    cmd = ["wandb", "sync"]
    if entity:
        cmd += ["--entity", entity]
    if project:
        cmd += ["--project", project]
    cmd += wandb_args + [path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync offline W&B runs listed in a text file.")
    parser.add_argument("--list", default="/home/ovaheb/scratch/jobs_to_sync.txt", help="Path to txt list of run dirs")
    parser.add_argument("--wandb-args", default="", help="Extra args to pass to 'wandb sync' (quoted)")
    parser.add_argument("--entity", default=None, help="W&B entity to sync to")
    parser.add_argument("--project", default=None, help="W&B project to sync to")
    parser.add_argument("--auto-discover", action="store_true", help="Auto-discover offline runs under common wandb dirs")
    parser.add_argument("--discover-dirs", default="/home/ovaheb/wandb,/scratch/ovaheb/slurm_jobs/wandb", help="Comma-separated base dirs to scan")
    args = parser.parse_args()

    list_path = Path(args.list)
    paths = read_paths(list_path)
    if args.auto_discover:
        discover_dirs = [p for p in args.discover_dirs.split(",") if p]
        paths.extend(discover_offline_runs(discover_dirs))
    paths = [normalize_path(p) for p in paths]
    # de-dup while preserving order
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]
    if not paths:
        print("No runs to sync.")
        return 0

    extra_args = [arg for arg in args.wandb_args.split(" ") if arg]

    remaining = []
    for path in paths:
        ok = sync_run(path, extra_args, args.entity, args.project)
        if ok:
            print(f"Synced: {path}")
        else:
            print(f"Failed: {path}")
            remaining.append(path)

    write_paths(list_path, remaining)
    if remaining:
        print(f"Remaining unsynced: {len(remaining)}")
        return 1
    print("All runs synced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
