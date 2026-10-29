"""Snapshot storage (working doc D6): a private Hugging Face Hub repo, or a local folder for smoke runs.

Every put() re-reads the stored file's sha256 and raises if it differs from the local file's.
The trainer never overwrites a snapshot file (names carry the update number), so a later save
cannot change a file while it is uploading.
"""
import shutil
from pathlib import Path

from common.results import sha256_file


class LocalStore:
    """Stand-in for the Hub with the same interface, used by smoke runs and tests."""

    def __init__(self, root):
        self.root = Path(root)

    def put(self, local, dest: str) -> str:
        target = self.root / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
        sha = sha256_file(local)
        if sha256_file(target) != sha:
            raise RuntimeError(f"sha mismatch after copy: {dest}")
        return sha

    def get(self, dest: str, local) -> Path:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / dest, local)
        return Path(local)

    def delete(self, dest: str) -> None:
        (self.root / dest).unlink(missing_ok=True)

    def read_text(self, dest: str) -> str | None:
        p = self.root / dest
        return p.read_text() if p.exists() else None

    def write_text(self, dest: str, text: str) -> None:
        p = self.root / dest
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


class HubStore:
    def __init__(self, repo_id: str):
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.repo_id = repo_id
        self.api.create_repo(repo_id, private=True, exist_ok=True)

    def put(self, local, dest: str) -> str:
        sha = sha256_file(local)
        self.api.upload_file(path_or_fileobj=str(local), path_in_repo=dest, repo_id=self.repo_id)
        info = self.api.get_paths_info(self.repo_id, [dest], expand=True)[0]
        lfs = info.lfs
        hub_sha = (lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256) if lfs else None
        if hub_sha != sha:
            raise RuntimeError(f"Hub sha {hub_sha} != local sha {sha} for {dest}")
        return sha

    def get(self, dest: str, local) -> Path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.repo_id, dest, local_dir=Path(local).parent)
        if Path(path) != Path(local):
            shutil.move(path, local)
        return Path(local)

    def delete(self, dest: str) -> None:
        self.api.delete_file(dest, repo_id=self.repo_id)
        # history keeps deleted LFS blobs; squashing frees the storage of old resume snapshots
        self.api.super_squash_history(repo_id=self.repo_id)

    def read_text(self, dest: str) -> str | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            return Path(hf_hub_download(self.repo_id, dest, force_download=True)).read_text()
        except EntryNotFoundError:
            return None

    def write_text(self, dest: str, text: str) -> None:
        self.api.upload_file(path_or_fileobj=text.encode(), path_in_repo=dest, repo_id=self.repo_id)


def make_store(spec: str):
    """`hub:<user>/<repo>` for the Hub, anything else is a local folder."""
    return HubStore(spec[4:]) if spec.startswith("hub:") else LocalStore(spec)
