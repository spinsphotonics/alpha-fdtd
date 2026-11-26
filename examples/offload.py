# SPDX-License-Identifier: MIT
"""A domain in host memory, streamed through the GPU: size it, plan it, and read it back in pieces.

When a domain does not fit in GPU memory, alpha-fdtd keeps it in pinned host memory if you allow it with host_bytes;
the results are bit-identical, only slower.  This example forces host memory on a small domain so it runs anywhere;
for a real one, drop force_host and pass the host_bytes that bytes_needed reports.
"""
import numpy as np
import alpha_fdtd as af

shape = (256, 960, 256)
gpu_bytes, host_bytes = af.bytes_needed(shape)
print(f"{shape}: {gpu_bytes / 2**30:.2f} GiB in GPU memory, or host_bytes >= {host_bytes / 2**30:.2f} GiB")

with af.Plan(shape, host_bytes=host_bytes, force_host=True) as plan:
    info = plan.info
    print("on host:", info.on_host, f" pinned: {info.host_bytes_used / 2**30:.2f} GiB, on the GPU: "
          f"{info.gpu_bytes_used / 2**30:.2f} GiB")
    nx, ny, nz = shape
    ones = np.broadcast_to(np.float32(1), (3,) + shape)                    # broadcast: no memory, even for huge domains
    plan.set(Ca=ones, Cb=np.broadcast_to(np.float32(0.25), (3,) + shape))
    src = np.zeros((16, 16, 16), np.float32)
    src[:] = 1.0
    plan.set(offset=(nx // 2, ny // 2, nz // 2), Ez=src)                   # a sub-box: only its own bytes move
    plan.execute(7 * 20)
    # Read Ez in slabs of whole y-blocks (the fastest boxes in host memory), never the whole domain at once.
    peak, by = 0.0, info.io_block_y * 8
    for y0 in range(0, ny, by):
        ez = plan.get("Ez", offset=(0, y0, 0), shape=(nx, min(by, ny - y0), nz))
        peak = max(peak, float(np.abs(ez).max()))
    print("max |Ez| after 140 steps:", peak)
    print(plan.describe())
