# SPDX-License-Identifier: MIT
"""The same problem on 1 GPU and on several: the results are bit-identical (equal checksums).

    python multi_gpu.py [NGPUS]       (default 2; CUDA_VISIBLE_DEVICES chooses which GPUs)
"""
import sys
import numpy as np
import alpha_fdtd as af

ngpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
shape = (192, 384, 160)                              # on n GPUs in GPU memory the domain is cut along y: ny > 24 (2n - 1)
rng = np.random.default_rng(0)
E = np.zeros((3,) + shape, np.float32)
E[:, 1:, 1:, 1:] = rng.standard_normal((3, shape[0] - 1, shape[1] - 1, shape[2] - 1))   # index 0 is the wall
ones = np.broadcast_to(np.float32(1), (3,) + shape)
quarter = np.broadcast_to(np.float32(0.25), (3,) + shape)

sums = []
for g in (1, ngpus):
    with af.Plan(shape, ngpus=g) as plan:
        plan.set(E=E, Ca=ones, Cb=quarter)
        plan.execute(140)
        print(f"{g} GPU(s): {plan.describe()}")
        sums.append(plan.checksums())
print("bit-identical:", sums[0] == sums[1])
sys.exit(0 if sums[0] == sums[1] else 1)
