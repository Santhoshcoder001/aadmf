"""UCI Gas Sensor Array Drift dataset loader.

This module implements ``UCIGasSensorLoader`` for the UCI dataset:
"Gas Sensor Array Drift Dataset at Different Concentrations" (id=270).

Primary mode reads the original ``batch*.dat`` files from ``data/raw``.
Optional fallback uses ``ucimlrepo`` when local files are unavailable.

The public interface matches ``StreamingSimulator``:
``next_batch() -> (X: pd.DataFrame, y: pd.Series)`` and ``(None, None)``
when exhausted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class UCIGasSensorLoader:
    """Stream loader for UCI Gas Sensor Array Drift batches.

    Supports two data sources:
    1. Local raw files: ``data/raw/batch1.dat`` ... ``data/raw/batch10.dat``
    2. ``ucimlrepo`` fallback (dataset id=270) if raw files are not found

    The loader can read all 10 batches or a specific subset.

    Args:
        data_dir: Directory containing ``batch*.dat`` files from UCI zip.
        batch_numbers: Specific batches to load (1..10). ``None`` means all 10.
        normalize: If True, apply ``StandardScaler`` per batch.
        use_ucimlrepo: If True, attempt ``ucimlrepo`` fallback when local files
            are unavailable.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        batch_numbers: Optional[Sequence[int]] = None,
        normalize: bool = True,
        use_ucimlrepo: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.batch_numbers = self._validate_batch_numbers(batch_numbers)
        self.normalize = normalize
        self.use_ucimlrepo = use_ucimlrepo
        self._current_index = 0

        self._batch_paths: Dict[int, Path] = {
            b: self.data_dir / f"batch{b}.dat" for b in self.batch_numbers
        }
        self._all_batches: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None

    @staticmethod
    def _validate_batch_numbers(batch_numbers: Optional[Sequence[int]]) -> List[int]:
        if batch_numbers is None:
            return list(range(1, 11))
        nums = sorted(set(int(b) for b in batch_numbers))
        if not nums:
            raise ValueError("batch_numbers cannot be empty")
        invalid = [b for b in nums if b < 1 or b > 10]
        if invalid:
            raise ValueError(f"Invalid batch numbers {invalid}. Valid range is 1..10")
        return nums

    @staticmethod
    def _parse_line(line: str, n_features: int) -> Tuple[int, np.ndarray]:
        """Parse one sparse-format row from UCI ``.dat`` files.

        Supported label token formats:
        - ``label``
        - ``label;concentration``
        """
        parts = line.strip().split()
        if not parts:
            raise ValueError("Encountered empty line in batch file")

        # Label token can be either `label` or `label;concentration`.
        raw_label = parts[0].split(";")[0]
        label = int(float(raw_label))

        row = np.zeros(n_features, dtype=float)
        for token in parts[1:]:
            idx_str, val_str = token.split(":", 1)
            idx = int(idx_str) - 1  # input uses 1-based feature indexes
            if 0 <= idx < n_features:
                row[idx] = float(val_str)
        return label, row

    @staticmethod
    def _normalize_batch(X_arr: np.ndarray, normalize: bool) -> np.ndarray:
        if normalize and len(X_arr) > 0:
            scaler = StandardScaler()
            return scaler.fit_transform(X_arr)
        return X_arr

    def _load_batch(self, batch_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
        """Load one ``batch*.dat`` file into (X, y)."""
        with batch_path.open("r", encoding="utf-8") as f:
            lines = [ln for ln in (line.strip() for line in f) if ln]

        if not lines:
            return pd.DataFrame(), pd.Series(dtype=int, name="gas_class")

        # Infer dimensionality from max sparse index across all lines.
        n_features = 0
        for line in lines:
            parts = line.split()
            for tok in parts[1:]:
                n_features = max(n_features, int(tok.split(":", 1)[0]))

        y_values: List[int] = []
        rows: List[np.ndarray] = []
        for line in lines:
            label, row = self._parse_line(line, n_features)
            y_values.append(label)
            rows.append(row)

        X_arr = np.vstack(rows)
        X_arr = self._normalize_batch(X_arr, self.normalize)

        cols = [f"sensor_{i}" for i in range(n_features)]
        X = pd.DataFrame(X_arr, columns=cols)
        y = pd.Series(y_values, name="gas_class")
        return X, y

    def _load_from_local_files(self) -> Optional[List[Tuple[pd.DataFrame, pd.Series]]]:
        """Load configured batches from local ``data/raw`` if all files exist."""
        missing = [str(self._batch_paths[b]) for b in self.batch_numbers if not self._batch_paths[b].exists()]
        if missing:
            return None

        out: List[Tuple[pd.DataFrame, pd.Series]] = []
        for b in self.batch_numbers:
            out.append(self._load_batch(self._batch_paths[b]))
        return out

    def _load_from_ucimlrepo(self) -> Optional[List[Tuple[pd.DataFrame, pd.Series]]]:
        """Best-effort fallback using ucimlrepo dataset id=270."""
        if not self.use_ucimlrepo:
            return None

        try:
            from ucimlrepo import fetch_ucirepo
        except Exception:
            return None

        try:
            dataset = fetch_ucirepo(id=270)
            X_raw = dataset.data.features.copy()
            y_raw = dataset.data.targets
            if isinstance(y_raw, pd.DataFrame):
                y_series = y_raw.iloc[:, 0]
            else:
                y_series = pd.Series(y_raw)
            y_series = pd.to_numeric(y_series, errors="coerce").fillna(0).astype(int)
            y_series.name = "gas_class"

            batch_col = None
            for col in X_raw.columns:
                if str(col).lower() == "batch":
                    batch_col = col
                    break

            batches: Dict[int, Tuple[pd.DataFrame, pd.Series]] = {}
            if batch_col is not None:
                batch_ids = pd.to_numeric(X_raw[batch_col], errors="coerce")
                X_feat = X_raw.drop(columns=[batch_col])
                X_feat = X_feat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                for b in range(1, 11):
                    mask = batch_ids == b
                    if mask.any():
                        X_arr = self._normalize_batch(X_feat.loc[mask].to_numpy(), self.normalize)
                        X = pd.DataFrame(X_arr, columns=[f"sensor_{i}" for i in range(X_feat.shape[1])])
                        y = y_series.loc[mask].reset_index(drop=True)
                        batches[b] = (X.reset_index(drop=True), y)
            else:
                # Fallback split if batch column is not provided by ucimlrepo source.
                X_feat = X_raw.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                idx_splits = np.array_split(np.arange(len(X_feat)), 10)
                for b, idxs in enumerate(idx_splits, start=1):
                    X_arr = self._normalize_batch(X_feat.iloc[idxs].to_numpy(), self.normalize)
                    X = pd.DataFrame(X_arr, columns=[f"sensor_{i}" for i in range(X_feat.shape[1])])
                    y = y_series.iloc[idxs].reset_index(drop=True)
                    batches[b] = (X.reset_index(drop=True), y)

            out: List[Tuple[pd.DataFrame, pd.Series]] = []
            for b in self.batch_numbers:
                if b not in batches:
                    return None
                out.append(batches[b])
            return out
        except Exception:
            return None

    def load_all_batches(self) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """Load and return all configured batches as ``[(X, y), ...]``.

        With default configuration this returns 10 tuples, one per batch.
        """
        if self._all_batches is None:
            loaded = self._load_from_local_files()
            if loaded is None:
                loaded = self._load_from_ucimlrepo()
            if loaded is None:
                raise FileNotFoundError(
                    "Unable to load UCI Gas Sensor batches. "
                    f"Expected batch files in {self.data_dir} or ucimlrepo dataset id=270 availability."
                )
            self._all_batches = loaded

        return self._all_batches

    def next_batch(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Return next UCI batch in the same interface as StreamingSimulator."""
        batches = self.load_all_batches()
        if self._current_index >= len(batches):
            return None, None

        X, y = batches[self._current_index]
        self._current_index += 1
        return X, y
