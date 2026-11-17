"""DisasterAI multimodal dataset with deterministic evaluation pairing.

Important: unless source metadata supplies a shared event identifier, cross-modal
pairs are severity-stratified proxies, NOT verified observations from the same
physical flood event. The dataset exposes this limitation in `pairing_mode`.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class DisasterTokenizer:
    """Small vocabulary fitted from training-split reports only."""
    def __init__(self, max_vocab_size: int = 800, max_length: int = 32):
        self.pad_token, self.unk_token = "<PAD>", "<UNK>"
        self.cls_token, self.sep_token = "<CLS>", "<SEP>"
        self.special_tokens = [self.pad_token, self.unk_token, self.cls_token, self.sep_token]
        self.pad_id, self.unk_id, self.cls_id, self.sep_id = 0, 1, 2, 3
        self.max_vocab_size = int(max_vocab_size)
        self.max_length = int(max_length)
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.is_fitted = False

    @staticmethod
    def tokenize_words(text):
        if not isinstance(text, str):
            return []
        return re.findall(r"\b\w+\b|[!?,.]", text.lower())

    def fit(self, texts):
        counts = Counter()
        for text in texts:
            counts.update(self.tokenize_words(text))
        self.word2idx = {token: i for i, token in enumerate(self.special_tokens)}
        for word, _ in counts.most_common(max(0, self.max_vocab_size - len(self.special_tokens))):
            self.word2idx[word] = len(self.word2idx)
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        self.is_fitted = True
        return self

    def encode(self, text):
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before encode().")
        words = self.tokenize_words(text)
        ids = [self.cls_id] + [self.word2idx.get(w, self.unk_id) for w in words] + [self.sep_id]
        ids = ids[: self.max_length]
        if ids[-1] != self.sep_id:
            ids[-1] = self.sep_id
        valid_len = len(ids)
        pad_len = self.max_length - valid_len
        ids += [self.pad_id] * pad_len
        pad_mask = [False] * valid_len + [True] * pad_len
        return np.asarray(ids, dtype=np.int64), np.asarray(pad_mask, dtype=bool)

    def decode(self, token_ids):
        return " ".join(self.idx2word.get(int(i), self.unk_token) for i in token_ids if int(i) != self.pad_id)

    @property
    def vocab_size(self):
        return len(self.word2idx) if self.word2idx else len(self.special_tokens)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "max_vocab_size": self.max_vocab_size,
            "max_length": self.max_length,
            "word2idx": self.word2idx,
        }, indent=2), encoding="utf-8")

    def load(self, path):
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        self.max_vocab_size = int(meta["max_vocab_size"])
        self.max_length = int(meta["max_length"])
        self.word2idx = {str(k): int(v) for k, v in meta["word2idx"].items()}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.is_fitted = True
        return self


def _stable_index(key: str, length: int, seed: int = 42) -> int:
    if length <= 0:
        raise ValueError("Cannot choose from an empty candidate list.")
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % length


def _robust_stats(array: np.ndarray):
    """Per-feature median/IQR with std and 1.0 fallbacks for degenerate columns."""
    flat = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
    center = np.nanmedian(flat, axis=0)
    q25, q75 = np.nanpercentile(flat, [25, 75], axis=0)
    scale = q75 - q25
    std = np.nanstd(flat, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
    center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    return center.astype(np.float32), scale.astype(np.float32)


class RealMultimodalDataset(Dataset):
    """SAR chips paired deterministically with split-local hydro/text proxies."""
    def __init__(self, data_root="data/cleaned", split="train", tokenizer=None, augment=False, seed=42):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.data_root = Path(data_root)
        self.split = split
        self.augment = bool(augment and split == "train")
        self.seed = int(seed)
        self.epoch = 0
        self.pairing_mode = "severity_stratified_proxy_unverified_event_match"

        image_manifest = self.data_root / "imagery" / "clean_manifest.csv"
        text_manifest = self.data_root / "text" / "cleaned_disaster_reports.csv"
        hydro_file = self.data_root / "hydrology" / "hydro_sequences.npy"
        hydro_manifest = self.data_root / "hydrology" / "sequence_manifest.csv"
        for path in (image_manifest, text_manifest, hydro_file, hydro_manifest):
            if not path.exists():
                raise FileNotFoundError(f"Required cleaned dataset file missing: {path}. Run clean_data.py first.")

        all_chips = pd.read_csv(image_manifest)
        self.df_chips = all_chips[all_chips["split"].astype(str).str.lower() == split].reset_index(drop=True)
        if self.df_chips.empty:
            raise ValueError(f"No SAR chips in split='{split}'. Refusing to fall back to other splits.")

        self.df_text = pd.read_csv(text_manifest)
        if "split" not in self.df_text.columns:
            raise ValueError("Cleaned text manifest has no 'split' column. Re-run the updated clean_data.py.")
        self.df_text["split"] = self.df_text["split"].astype(str).str.lower()
        self.df_text = self.df_text[self.df_text["split"] == split].reset_index(drop=True)
        if self.df_text.empty:
            raise ValueError(f"No cleaned crisis reports in text split='{split}'. Re-run clean_data.py or inspect the source corpus.")

        self.hydro_sequences = np.load(hydro_file).astype(np.float32)
        self.df_hydro = pd.read_csv(hydro_manifest)
        if len(self.df_hydro) != len(self.hydro_sequences):
            raise ValueError("Hydrology manifest row count does not match hydro_sequences.npy.")
        if "split" not in self.df_hydro.columns:
            raise ValueError("Hydrology manifest has no temporal 'split' column. Re-run the updated clean_data.py.")
        self.df_hydro["split"] = self.df_hydro["split"].astype(str).str.lower()
        self.hydro_split_indices = np.flatnonzero(self.df_hydro["split"].to_numpy() == split)
        if len(self.hydro_split_indices) == 0:
            raise ValueError(f"No hydrology windows in split='{split}'. Inspect station lengths and temporal split settings.")

        # Tokenizer must be fitted on training text only and reused for val/test.
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            if split != "train":
                raise ValueError("Validation/test datasets must receive the training-fitted tokenizer.")
            self.tokenizer = DisasterTokenizer().fit(self.df_text["clean_text"].fillna("").tolist())

        self.text_by_sev = {
            s: self.df_text.loc[self.df_text["severity_tier"].astype(int) == s, "clean_text"].fillna("").tolist()
            for s in (0, 1, 2)
        }
        self.hydro_by_sev = {
            s: self.hydro_split_indices[
                self.df_hydro.iloc[self.hydro_split_indices]["severity_tier"].astype(int).to_numpy() == s
            ] for s in (0, 1, 2)
        }

        # Deterministic per-chip pairing: repeated evaluation passes see identical inputs.
        self.hydro_indices_for_chip = []
        self.text_for_chip = []
        self.pairing_fallback_counts = {"hydro_severity_fallback": 0, "text_severity_fallback": 0}
        for _, row in self.df_chips.iterrows():
            sev = int(row["severity_tier"])
            h_candidates = self.hydro_by_sev[sev]
            if len(h_candidates) == 0:
                h_candidates = self.hydro_split_indices
                self.pairing_fallback_counts["hydro_severity_fallback"] += 1
            h_pos = _stable_index(str(row["chip_id"]), len(h_candidates), self.seed)
            self.hydro_indices_for_chip.append(int(h_candidates[h_pos]))

            t_candidates = self.text_by_sev[sev]
            if not t_candidates:
                t_candidates = self.df_text["clean_text"].fillna("").tolist()
                self.pairing_fallback_counts["text_severity_fallback"] += 1
            t_pos = _stable_index(str(row["chip_id"]), len(t_candidates), self.seed + 1)
            self.text_for_chip.append(str(t_candidates[t_pos]))

        # Set by loader factory using training split only.
        self.hydro_input_center = np.zeros(5, dtype=np.float32)
        self.hydro_input_scale = np.ones(5, dtype=np.float32)
        self.target_center = 0.0
        self.target_scale = 1.0

    def __len__(self):
        return len(self.df_chips)

    def set_scalers(self, input_center, input_scale, target_center, target_scale):
        self.hydro_input_center = np.asarray(input_center, dtype=np.float32)
        self.hydro_input_scale = np.asarray(input_scale, dtype=np.float32)
        self.target_center = float(target_center)
        self.target_scale = float(max(target_scale, 1e-6))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, idx):
        row = self.df_chips.iloc[idx]
        severity = int(row["severity_tier"])
        image = np.load(row["clean_s1_path"]).astype(np.float32)
        mask = np.load(row["clean_mask_path"]).astype(np.int64)

        if self.augment:
            # Stateless augmentation seed: deterministic for a given sample/epoch.
            rng = np.random.default_rng(self.seed + self.epoch * 100003 + idx)
            if rng.random() < 0.5:
                image, mask = np.flip(image, -1).copy(), np.flip(mask, -1).copy()
            if rng.random() < 0.5:
                image, mask = np.flip(image, -2).copy(), np.flip(mask, -2).copy()
            k = int(rng.integers(0, 4))
            if k:
                image = np.rot90(image, k, axes=(-2, -1)).copy()
                mask = np.rot90(mask, k, axes=(-2, -1)).copy()

        h_idx = self.hydro_indices_for_chip[idx]
        hydro_raw = self.hydro_sequences[h_idx].copy()
        hydro_norm = (hydro_raw - self.hydro_input_center) / self.hydro_input_scale
        hydro_norm = np.nan_to_num(hydro_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        h_row = self.df_hydro.iloc[h_idx]
        future_peak = float(h_row["future_peak_ft"])
        future_peak_norm = (future_peak - self.target_center) / self.target_scale

        report_text = self.text_for_chip[idx]
        token_ids, pad_mask = self.tokenizer.encode(report_text)
        flood_ratio = float(row["flood_ratio"])
        hazard_index = float(np.clip(0.5 * flood_ratio + 0.5 * (severity / 2.0), 0.0, 1.0))

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).long(),
            "env_seq": torch.from_numpy(hydro_norm).float(),
            "env_seq_raw": torch.from_numpy(hydro_raw).float(),
            "trend_target": torch.tensor(future_peak, dtype=torch.float32),
            "trend_target_norm": torch.tensor(future_peak_norm, dtype=torch.float32),
            "severity": torch.tensor(severity, dtype=torch.long),
            "hazard_index": torch.tensor(hazard_index, dtype=torch.float32),
            "token_ids": torch.from_numpy(token_ids).long(),
            "pad_mask": torch.from_numpy(pad_mask).bool(),
            "report_text": report_text,
            "chip_id": str(row["chip_id"]),
            "country": str(row.get("country", "Unknown")),
            "flood_ratio": torch.tensor(flood_ratio, dtype=torch.float32),
            "hydro_sequence_id": str(h_row.get("seq_id", h_idx)),
            "hydro_station_id": str(h_row.get("station_id", "Unknown")),
            "pairing_mode": self.pairing_mode,
        }


def get_multimodal_dataloaders(data_root="data/cleaned", batch_size=8, seed=42):
    """Build split-safe loaders; all normalization statistics come from train only."""
    root = Path(data_root)
    text_path = root / "text" / "cleaned_disaster_reports.csv"
    hydro_path = root / "hydrology" / "hydro_sequences.npy"
    hydro_meta_path = root / "hydrology" / "sequence_manifest.csv"
    if not text_path.exists() or not hydro_path.exists() or not hydro_meta_path.exists():
        raise FileNotFoundError("Cleaned text/hydrology files are missing. Run clean_data.py first.")

    text_df = pd.read_csv(text_path)
    if "split" not in text_df.columns:
        raise ValueError("Text manifest is missing split column; run updated clean_data.py.")
    train_text = text_df.loc[text_df["split"].astype(str).str.lower() == "train", "clean_text"].fillna("").tolist()
    if not train_text:
        raise ValueError("No training text records available to fit tokenizer.")
    tokenizer = DisasterTokenizer(max_vocab_size=800, max_length=32).fit(train_text)

    train_ds = RealMultimodalDataset(data_root, "train", tokenizer, augment=True, seed=seed)
    val_ds = RealMultimodalDataset(data_root, "val", tokenizer, augment=False, seed=seed)
    test_ds = RealMultimodalDataset(data_root, "test", tokenizer, augment=False, seed=seed)

    hydro_array = np.load(hydro_path).astype(np.float32)
    hydro_meta = pd.read_csv(hydro_meta_path)
    train_hydro_idx = np.flatnonzero(hydro_meta["split"].astype(str).str.lower().to_numpy() == "train")
    if len(train_hydro_idx) == 0:
        raise ValueError("No training hydrology windows available for input scaling.")
    input_center, input_scale = _robust_stats(hydro_array[train_hydro_idx])

    train_target = hydro_meta.iloc[train_hydro_idx]["future_peak_ft"].to_numpy(dtype=np.float64)
    target_center = float(np.median(train_target))
    q25, q75 = np.percentile(train_target, [25, 75])
    target_scale = float(q75 - q25)
    if not np.isfinite(target_scale) or target_scale < 1e-6:
        target_scale = float(np.std(train_target))
    if not np.isfinite(target_scale) or target_scale < 1e-6:
        target_scale = 1.0

    for ds in (train_ds, val_ds, test_ds):
        ds.set_scalers(input_center, input_scale, target_center, target_scale)

    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Keep scaler metadata accessible to training code and serialize separately for app inference.
    train_ds.scaler_metadata = {
        "input_center": input_center.tolist(),
        "input_scale": input_scale.tolist(),
        "target_center_ft": target_center,
        "target_scale_ft": target_scale,
        "target_method": "median_iqr_train_split_only",
        "input_method": "median_iqr_train_split_only",
    }
    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    train_l, val_l, test_l, tok = get_multimodal_dataloaders(batch_size=4)
    print(f"Tokenizer vocabulary: {tok.vocab_size}")
    for name, loader in (("train", train_l), ("val", val_l), ("test", test_l)):
        labels = [int(x) for x in loader.dataset.df_chips["severity_tier"].tolist()]
        print(f"{name}: {len(loader.dataset)} samples; class counts={dict(Counter(labels))}; hydro fallbacks={loader.dataset.pairing_fallback_counts}")
    sample = next(iter(train_l))
    print("Image:", tuple(sample["image"].shape), "Hydrology:", tuple(sample["env_seq"].shape), "Tokens:", tuple(sample["token_ids"].shape))
    print("Pairing note:", sample["pairing_mode"][0])
