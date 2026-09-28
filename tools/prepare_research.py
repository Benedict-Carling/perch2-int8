"""Prepare historical research code from a locally downloaded Perch SavedModel.

Requires TensorFlow separately. Does not download datasets or redistribute audio.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--savedmodel", required=True, type=Path)
    parser.add_argument("--classes", required=True, type=Path,
                        help="Perch class CSV with the ebird2021 column in original class order")
    args = parser.parse_args()
    import tensorflow as tf
    checkpoint = args.savedmodel / "variables" / "variables"
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for relative, expected in manifest["source_checkpoint_files"].items():
        path = args.savedmodel / relative
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Source checkpoint mismatch: {relative}. Use the verified original checkpoint, or revalidate the mapping before proceeding.")
    reader = tf.train.load_checkpoint(str(checkpoint))
    leaves = {}
    for name in reader.get_variable_to_shape_map():
        match = re.match(r"_tf_var_leaves/(\d+)/", name)
        if match:
            leaves[int(match.group(1))] = reader.get_tensor(name)
    mapping = json.loads((ROOT / "research/src/perch2_leaf_mapping.json").read_text())
    if set(map(int, mapping)) != set(leaves):
        raise SystemExit("Checkpoint leaf set does not match the verified mapping")
    import csv
    with args.classes.open() as f:
        rows = list(csv.DictReader(f))
    expected_classes = manifest["source_checkpoint_files"]["assets/perch_v2_ebird_classes.csv"]
    if hashlib.sha256(args.classes.read_bytes()).hexdigest() != expected_classes:
        raise SystemExit("Class CSV does not match the verified original class order")
    if len(rows) != 14795 or not rows or "ebird2021" not in rows[0]:
        raise SystemExit("Expected 14,795 original-order classes with an ebird2021 column")
    destination = ROOT / "research/src"
    np.savez(destination / "perch2_weights.npz", **{str(k): leaves[k] for k in sorted(leaves)})
    shutil.copy2(ROOT / "perch2_mel_matrix.npy", destination / "perch2_mel_matrix.npy")
    shutil.copy2(args.classes, destination / "perch_perch_v2_ebird_classes.csv")
    print("Prepared weights, mel matrix and class mapping. Dataset downloads remain separate.")


if __name__ == "__main__":
    main()
