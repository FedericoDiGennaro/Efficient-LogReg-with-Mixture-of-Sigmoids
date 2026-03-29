from __future__ import annotations

import bz2
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, Dict

import numpy as np
from sklearn.datasets import load_svmlight_file


LIBSVM_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"

# File names on the LIBSVM site
LIBSVM_FILES: Dict[str, Dict[str, str]] = {
    "a9a": {
        "train": "a9a",
        "test": "a9a.t",
    },
    "w8a": {
        "train": "w8a",
        "test": "w8a.t",
    },
    "ijcnn1": {
        "train": "ijcnn1.bz2",
        # LIBSVM provides multiple splits; "test" is the official testing file.
        "test": "ijcnn1.t.bz2",
        "tr": "ijcnn1.tr.bz2",
        "val": "ijcnn1.val.bz2",
    },
}

# Feature counts from LIBSVM binary page (used to keep train/test dimensions consistent).
N_FEATURES: Dict[str, int] = {
    "a9a": 123,
    "w8a": 300,
    "ijcnn1": 22,
}


@dataclass
class LibSVMMeta:
    name: str
    split: str
    n: int
    d: int
    R: float
    local_path: str
    url: str


def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        # Backward-compatibility: reuse legacy cache layout data/<name>/<file>
        # when present (e.g., existing a9a files).
        legacy = dst.parents[2] / dst.parent.name / dst.name
        if legacy.exists():
            print(f"[cache] {legacy} -> {dst}")
            shutil.copy2(legacy, dst)
        else:
            print(f"[download] {url} -> {dst}")
            urllib.request.urlretrieve(url, dst)
    return dst


def _decompress_bz2(path_bz2: Path) -> Path:
    assert path_bz2.suffix == ".bz2"
    out_path = path_bz2.with_suffix("")  # drop .bz2
    if out_path.exists():
        return out_path
    print(f"[decompress] {path_bz2} -> {out_path}")
    with bz2.open(path_bz2, "rb") as f_in, open(out_path, "wb") as f_out:
        f_out.write(f_in.read())
    return out_path


def _to_pm1(y) -> np.ndarray:
    y = np.asarray(y)
    # LIBSVM files usually are already ±1, but normalize anyway.
    vals = np.unique(y)
    if len(vals) != 2:
        raise ValueError(f"Expected binary labels, got {vals}")
    if set(vals.tolist()) == {-1, 1}:
        return y.astype(np.int64)
    # map larger value to +1
    hi = vals.max()
    return np.where(y == hi, 1, -1).astype(np.int64)


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)


def _scale_max_norm_1(X: np.ndarray) -> Tuple[np.ndarray, float]:
    norms = np.linalg.norm(X, axis=1)
    R = float(norms.max()) if norms.size else 0.0
    if R > 0:
        X = X / R
    return X, R


def load_libsvm_binary(
    name: str,
    *,
    split: str = "train",               # "train"/"test" (and for ijcnn1 also "tr"/"val")
    data_dir: Union[str, Path] = "data",
    max_rounds: Optional[int] = None,   # prefix length
    dense: bool = True,
    add_intercept: bool = True,
    scale_max_norm: bool = True,
) -> Tuple[np.ndarray, np.ndarray, LibSVMMeta]:
    """
    Loads a9a / w8a / ijcnn1 from LIBSVM (download+cache).
    Returns dense float64 X, y in {-1,+1}, plus meta including R after scaling.

    Notes:
      - If scale_max_norm=True: rescales so max ||x||_2 <= 1 (after intercept if enabled).
      - For consistent dimensions across splits, uses n_features from LIBSVM metadata.
    """
    name = name.lower().strip()
    if name not in LIBSVM_FILES:
        raise ValueError(f"Unknown dataset '{name}'. Allowed: {list(LIBSVM_FILES.keys())}")

    split = split.lower().strip()
    if split not in LIBSVM_FILES[name]:
        raise ValueError(f"Unknown split '{split}' for {name}. Allowed: {list(LIBSVM_FILES[name].keys())}")

    d_site = N_FEATURES[name]
    fname = LIBSVM_FILES[name][split]
    url = LIBSVM_BASE + fname

    data_dir = Path(data_dir)
    raw_path = data_dir / "libsvm" / name / fname
    _download(url, raw_path)

    # Decompress if needed
    if raw_path.suffix == ".bz2":
        svm_path = _decompress_bz2(raw_path)
    else:
        svm_path = raw_path

    # Load (sparse) -> maybe dense
    X_sp, y_raw = load_svmlight_file(str(svm_path), n_features=d_site)
    y = _to_pm1(y_raw)

    if dense:
        X = X_sp.toarray().astype(np.float64, copy=False)
    else:
        # If you ever want sparse, you can return X_sp directly
        X = X_sp

    if max_rounds is not None:
        X = X[:max_rounds]
        y = y[:max_rounds]

    if dense and add_intercept:
        X = _add_intercept(X)

    if dense and scale_max_norm:
        X, R = _scale_max_norm_1(X)
    elif dense:
        R = float(np.max(np.linalg.norm(X, axis=1))) if X.shape[0] else 0.0
    else:
        R = float("nan")

    meta = LibSVMMeta(
        name=name,
        split=split,
        n=int(X.shape[0]),
        d=int(X.shape[1]),
        R=R,
        local_path=str(svm_path),
        url=url,
    )
    return X, y, meta


if __name__ == "__main__":
    for ds in ["a9a", "w8a", "ijcnn1"]:
        X, y, meta = load_libsvm_binary(ds, split="train", max_rounds=2000)
        print(meta)
        print(" y counts:", {v: int((y == v).sum()) for v in (-1, 1)})
