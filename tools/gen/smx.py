# SPDX-License-Identifier: MIT
"""smx: IR transforms applied before CUDA generation.  `hoist1` moves the load-type operations of each half step (mailbox
loads, shuffles, y-halo get) ahead of the arithmetic; the set of arithmetic operations is unchanged (bit-identity is checked on
the GPU against the reference kernel).
"""
import re

def t_id(src, arg):
    return src
TR = {'id': t_id}
C6 = ['Cax', 'Cay', 'Caz', 'Cbx', 'Cby', 'Cbz']
WRITE1 = ('sub', 'mul', 'fma', 'prmt', 'shfl', 'mov', 'lds', 'xown')

def _is_coef(n):
    return isinstance(n, str) and n[:3] in C6 and ('_' in n) and (n.split('_')[1][:1] in 'AB')

def _rw(op):
    """-> (reads, writes) name lists of one non-ifq op"""
    k = op[0]
    return ([], [])

def _rename(op, f):
    k = op[0]
    return op

def _is_load(op):
    k = op[0]
    if k in ('lds', 'shfl', 'yget'):
        return True
    if k == 'ifq':
        ks = {o[0] for o in op[2] + op[3]}
        return ks <= {'lds', 'xld', 'xown', 'prmt'} and bool(ks & {'lds', 'xld', 'xown'})
    return False

def ir_hoist(g, d=1):
    """Within each half (H before the middle barrier, E after it), move the load-type ops (mailbox LDS, shuffles, y-halo get, the
    q-specific load blocks) of level group i+d to just before the first arithmetic op of group i.  Groups end at 'yput' (H) and at
    'sts MBE' (E).  Legal: those ops read only values that no op of the half writes (fields of the other half, loads), see README."""
    orig_body = g.body

    def body():
        out = []
        for ops in orig_body():
            res = []
            seg = []

            def flush(seg):
                groups, cur = ([], [])
                for op in seg:
                    cur.append(op)
                    if op[0] == 'yput' or (op[0] == 'sts' and op[2] == 'MBE'):
                        groups.append(cur)
                        cur = []
                tail = cur
                if len(groups) < 2:
                    return seg
                loads = [[op for op in gr if _is_load(op)] for gr in groups]
                rest = [[op for op in gr if not _is_load(op)] for gr in groups]
                new = []
                for i, gr in enumerate(groups):
                    pre = []
                    if i == 0:
                        for j in range(0, min(d + 1, len(groups))):
                            pre += loads[j]
                    elif i + d < len(groups):
                        pre += loads[i + d]
                    r_ = rest[i]
                    pos = next((p for p, op in enumerate(r_) if op[0] in ('sub', 'mul', 'fma')), len(r_))
                    new += r_[:pos][:0] + pre + r_[:pos] + r_[pos:]
                return new + tail
            for op in ops:
                if op[0] in ('bar', 'stout'):
                    res += flush(seg)
                    seg = []
                    res.append(op)
                else:
                    seg.append(op)
            res += flush(seg)
            assert sorted(map(repr, res)) == sorted(map(repr, ops))
            out.append(res)
        return out
    g.body = body

def patch_gen(g, spec):
    for t in [x for x in spec.split('+') if x]:
        if t.startswith('hoist'):
            ir_hoist(g, int(t[5:] or 1))
    return g
