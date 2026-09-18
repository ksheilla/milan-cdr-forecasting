from __future__ import annotations
 
from pathlib import Path
 
# scripts/config.py -> scripts/ -> project root
ROOT = Path(__file__).resolve().parent.parent
 
DATA_RAW_DIR = ROOT / "dataverse Files"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
 
PARQUET_NAME = "milan_internet_optimized.parquet"
 
# Canonical location for NEW writes. step1 should write here.
PARQUET_WRITE_PATH = DATA_PROCESSED_DIR / PARQUET_NAME
 
# Locations searched when READING, newest convention first, then the legacy
# repo-root location so an existing file from an earlier run still resolves.
_PARQUET_CANDIDATES = (
    PARQUET_WRITE_PATH,
    ROOT / PARQUET_NAME,
)
 
 
def _resolve_parquet() -> Path:
    """Return the first Parquet file that exists, else the canonical write path.
 
    Returning the canonical path when nothing exists keeps the eventual
    FileNotFoundError pointing at the place the file SHOULD be, which is more
    useful than a bare relative filename.
    """
    for candidate in _PARQUET_CANDIDATES:
        if candidate.exists():
            return candidate
    return PARQUET_WRITE_PATH
 
 
PARQUET_PATH = _resolve_parquet()
 
 
def require_parquet() -> Path:
    """Fail fast with an actionable message instead of a raw FileNotFoundError.
 
    Call this at the top of step2/step4:
 
        from config import require_parquet
        df = pd.read_parquet(require_parquet())
    """
    if PARQUET_PATH.exists():
        return PARQUET_PATH
 
    searched = "\n".join(f"    {c}" for c in _PARQUET_CANDIDATES)
    raise SystemExit(
        f"\nOptimised Parquet file not found. Searched:\n{searched}\n\n"
        f"Run the data-preparation step first, from anywhere:\n"
        f"    python {ROOT / 'scripts' / 'step1_memory_optimization.py'}\n\n"
        f"That step needs the 62 raw sms-call-internet-mi-*.txt files in:\n"
        f"    {DATA_RAW_DIR}\n"
    )
 
 
# Create output directories on import so no script has to.
for _directory in (DATA_PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
 
 
if __name__ == "__main__":
    print(f"Project root      : {ROOT}")
    print(f"Raw data dir      : {DATA_RAW_DIR}   (exists: {DATA_RAW_DIR.exists()})")
    print(f"Parquet write path: {PARQUET_WRITE_PATH}")
    print(f"Parquet read path : {PARQUET_PATH}   (exists: {PARQUET_PATH.exists()})")
    print(f"Figures dir       : {FIGURES_DIR}")
    print(f"Results dir       : {RESULTS_DIR}")
 