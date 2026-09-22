"""Load the specified OpenRec algorithm source tree."""
import sys
from pathlib import Path


def load_openrec(root):
    root = Path(root).resolve()
    if not (root / "algorithm/rank/lr.py").is_file():
        raise ValueError("openrec_algorithm must point to the rec-algorithm source repository")
    sys.path.insert(0, str(root))
    import algorithm.rank.lr as lr
    if not Path(lr.__file__).resolve().is_relative_to(root):
        raise ValueError("a different rec-algorithm is already imported")
    return root
