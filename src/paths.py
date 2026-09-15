"""Paths and environment bootstrap.

Works in three places without edits:
  * Colab       -> mounts Drive, uses MyDrive/odiaeval_g6
  * nano server -> uses ./work (or $ODIAEVAL_ROOT)
  * anywhere    -> set ODIAEVAL_ROOT to override
"""
import os, sys
from pathlib import Path


def _in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


def get_root() -> Path:
    env = os.environ.get("ODIAEVAL_ROOT")
    if env:
        return Path(env)
    if _in_colab():
        try:
            from google.colab import drive
            if not os.path.ismount("/content/drive"):
                drive.mount("/content/drive")
            return Path("/content/drive/MyDrive/odiaeval_g6")
        except Exception:
            return Path("/content/odiaeval_g6")
    return Path(__file__).resolve().parents[1] / "work"


ROOT     = get_root()
RAW      = ROOT / "raw"
INTERIM  = ROOT / "interim"
PROC     = ROOT / "processed" / "odia_hate_v1"
ART      = ROOT / "artifacts"
CKPT_DIR = ROOT / "checkpoints"
CKPT     = CKPT_DIR / "best"


def ensure_dirs():
    for d in (RAW, INTERIM, PROC.parent, ART, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return ROOT
