#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate the committed kernel texts in src/kernels/ (developer tool; users never run it).

    python3 tools/gen/gen.py            write src/kernels/*.cuh
    python3 tools/gen/gen.py --check    exit 1 unless every committed text equals the generated one, byte for byte

The texts are generated from the configurations in KERNELS; only the code paths those configurations reach are kept here.

Pipeline: gen3.parse_cfg -> smx.patch_gen (IR hoist) -> cudagen6.cuda -> persist.persist (persistent kernel, L2 hints)
          [-> sm80.lower / fold / pubmid / split2 for the cp.async (sm_80) texts].  Pure Python 3, no dependencies.
"""
import hashlib, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen3, cudagen6, persist, smx, sm80   # noqa: E402

BASE = ("mg=1,hp=1,ctlw=312,WGP=1,sph=1,D=4,DX=2,NSI=8,split=1,wc=4,k=7,G=64,"
        "ctl=nofb+ctrim+nopoll+mrg+uwait+onewait+cx+dup+pm+pw3+xt+ct+cp+gt+dxp,pubw=1,pub=16,l2h=3,ir=hoist1")
CPASYNC = BASE.replace("D=4", "D=2").replace("NSI=8", "NSI=4").replace("pubw=1", "pubw=0")
KERNELS = {      # name -> (generator config, lowering to cp.async)
    "zs7_tma": (BASE, False),
    "zs7_cpasync_l2h3": (CPASYNC, True),
    "zs7_cpasync_l2h0": (CPASYNC.replace("l2h=3", "l2h=0"), True),
}
GEN_ONLY = ("hp=1", "mg=1", "pubw=1", "pubw=0")
OWN = ("pub", "l2h", "ir")


def kern(cfg, cpasync):
    toks = cfg.split(",")
    val = lambda key, d: ([t.split("=", 1)[1] for t in toks if t.startswith(key + "=")] or [d])[0]
    g = gen3.parse_cfg(",".join(t for t in toks if t not in GEN_ONLY and t.split("=")[0] not in OWN))
    smx.patch_gen(g, val("ir", ""))
    src = persist.persist(cudagen6.cuda(g), int(val("pub", "4")), "pubw=1" in toks, 0, False, False, "sc", int(val("l2h", "0")))
    if cpasync:
        k = sm80.fold(sm80.lower(src, 3, {"S": 8, "X": 9, "O": 10}), {"F": 3, "C": 4, "X": 1, "O": 2}, 0, "", "end", 0)
        src = sm80.split2(sm80.pubmid(k), {3: 5, 4: 6, 2: 0})
    return "// SPDX-License-Identifier: MIT\n" + src


def main():
    out = os.path.join(HERE, "..", "..", "src", "kernels")
    check, bad = "--check" in sys.argv, 0
    for name, (cfg, cp) in KERNELS.items():
        text, path = kern(cfg, cp), os.path.join(out, name + ".cuh")
        if check:
            same = os.path.exists(path) and open(path).read() == text
            bad += not same
            print(f"{name}: {'ok' if same else 'DIFFERS'}")
        else:
            open(path, "w").write(text)
            print(f"{name}: sha1 {hashlib.sha1(text.encode()).hexdigest()} -> {os.path.relpath(path)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
