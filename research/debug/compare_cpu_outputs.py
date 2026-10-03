"""Compare INT8 embeddings across CPUs on four synthetic signals."""
import json
from pathlib import Path
import platform
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perch_int8 import log_mel

fixture = np.load(ROOT / "synthetic_reference.npz")
basic_reference = np.load(ROOT / "research/debug/basic_reference.npz")
matrix = np.load(ROOT / "perch2_mel_matrix.npy")
print('PORTABILITY', platform.platform(), 'ORT', ort.__version__)
for name, level in [('all', ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                    ('basic', ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                    ('disabled', ort.GraphOptimizationLevel.ORT_DISABLE_ALL)]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = level
    session = ort.InferenceSession(str(ROOT / "perch2_backbone_int8.onnx"), opts,
                                   providers=['CPUExecutionProvider'])
    for i in range(len(fixture['audio'])):
        for source, mel, expected in [
                ('fixed_features', fixture['log_mel'][i][None, None], fixture['embedding_direct_dft'][i]),
                ('numpy_frontend', log_mel(fixture['audio'][i], matrix), fixture['embedding'][i]),
                ('fixed_features_vs_basic', fixture['log_mel'][i][None, None], basic_reference['fixed'][i]),
                ('numpy_frontend_vs_basic', log_mel(fixture['audio'][i], matrix), basic_reference['audio'][i])]:
            actual = session.run(['embedding'], {'mel': mel})[0][0]
            cosine = float(np.dot(actual, expected) / np.linalg.norm(actual) / np.linalg.norm(expected))
            print('PORTABILITY', json.dumps({'level': name, 'input': source, 'sample': i,
                  'cosine': cosine, 'max_error': float(np.max(np.abs(actual-expected))),
                  'relative_l2': float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))}))
