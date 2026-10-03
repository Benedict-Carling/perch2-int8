"""Mirror the files tracked by Git to the Hugging Face model repo."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SKIP = (".github/", ".gitignore")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="BenedictCarling/perch2-int8")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT).strip():
        raise SystemExit("Commit tracked changes before syncing")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    with tempfile.TemporaryDirectory(prefix="perch-hf-") as directory:
        stage = Path(directory)
        for filename in files:
            if not filename or filename.startswith(SKIP):
                continue
            target = stage / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / filename, target)
        (stage / "SOURCE_COMMIT").write_text(commit + "\n")
        result = HfApi().upload_folder(
            repo_id=args.repo_id, repo_type="model", folder_path=stage,
            delete_patterns=["*"],
            commit_message=f"Sync from GitHub {commit[:12]}",
        )
    print(f"https://huggingface.co/{args.repo_id} at {result.oid}")


if __name__ == "__main__":
    main()
