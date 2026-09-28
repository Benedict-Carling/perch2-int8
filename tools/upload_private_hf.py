"""Upload the committed release to a PRIVATE Hugging Face model repository.

Run after `hf auth login`. Never changes a repository to public. Existing
public destinations are rejected. Uploads only files tracked by Git.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
import shutil

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Default: signed-in-user/perch2-int8")
    args = parser.parse_args()
    api = HfApi()
    owner = api.whoami()["name"]
    repo_id = args.repo_id or f"{owner}/perch2-int8"
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT).strip():
        raise SystemExit("Commit tracked changes before uploading")
    try:
        info = api.model_info(repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True)
        info = api.model_info(repo_id)
    if not info.private:
        raise SystemExit("Refusing upload: destination is not private")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    with tempfile.TemporaryDirectory(prefix="perch-private-hf-") as directory:
        stage = Path(directory)
        for filename in filter(None, files):
            source = ROOT / filename
            if source.is_symlink():
                raise SystemExit(f"Refusing symlink: {filename}")
            target = stage / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (stage / "SOURCE_COMMIT").write_text(commit + "\n")
        result = api.upload_folder(repo_id=repo_id, folder_path=stage, repo_type="model",
                                   commit_message=f"Private review candidate from GitHub {commit[:12]}")
    if not api.model_info(repo_id).private:
        raise SystemExit("Unexpected privacy state after upload; inspect destination")
    print(f"Private model page: https://huggingface.co/{repo_id}")
    print(f"Upload commit: {result.oid}")


if __name__ == "__main__":
    main()
