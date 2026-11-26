# SPDX-License-Identifier: MIT
"""sm80: lower a persistent TMA kernel text (producer warpgroup form) to sm_80 (A100), as a source transform.

sm_80 lacks cp.async.bulk (TMA), mbarrier expect_tx / complete_tx, try_wait, fence.proxy.async and setmaxnreg.  Every bulk copy
of a producer thread becomes a warp-cooperative cp.async copy (16 B per lane), arrivals become cp.async.mbarrier.arrive.noinc,
stores become ld.shared + st.global per lane, waits become mbarrier.test_wait.parity spins.  lower() keeps producer warps;
fold() moves their roles onto the compute warps (no producer warps, 256 threads); pubmid() and split2() move the progress
publish and share the copy jobs between two warps.  Each edit is asserted to apply the expected number of times.
"""
import re

def subn(src, old, new, n=1):
    c = src.count(old)
    assert c == n, (c, n, old[:160])
    return src.replace(old, new)
HELPERS_80 = '__device__ __forceinline__ void mb_expect(unsigned, unsigned) {}      // sm80: arrivals are cp.async.mbarrier.arrive.noinc (mb_arr)\n__device__ __forceinline__ void mb_arr(unsigned mb) { asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(mb) : "memory"); }\n// sm80 warp-cooperative copies: EVERY lane of the calling warp calls with the same arguments; 16-B chunks lane, lane + 32, ...\n__device__ __forceinline__ void tma_ld(unsigned dst, const void* src, unsigned bytes, unsigned) {\n  _Pragma("unroll 1") for (unsigned o = (threadIdx.x & 31u) * 16u; o < bytes; o += 512u)\n    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + o), "l"((const char*)src + o) : "memory"); }\n__device__ __forceinline__ void box_ld(unsigned dst, const void* src, unsigned) {      // tensor box B\': 9 pairs x (SGH lanes x 24 B), pair stride SS x 24 B\n  _Pragma("unroll 1") for (unsigned i = threadIdx.x & 31u; i < 9u * (SGH * 24u / 16u); i += 32u) { const unsigned p = i / (SGH * 24u / 16u), c = i % (SGH * 24u / 16u);\n    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + p * (SGH * 24u) + 16u * c), "l"((const char*)src + p * (SS * 24u) + 16u * c) : "memory"); } }\n__device__ __forceinline__ void tma_st(void* dst, unsigned src, unsigned bytes) {\n  _Pragma("unroll 1") for (unsigned o = (threadIdx.x & 31u) * 16u; o < bytes; o += 512u) { unsigned a, b, c, d;\n    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(src + o) : "memory");\n    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"((char*)dst + o), "r"(a), "r"(b), "r"(c), "r"(d) : "memory"); } }\n'
HINTS_80 = '__device__ __forceinline__ void tma_st_h(void* dst, unsigned src, unsigned bytes, u64 pol) {\n  _Pragma("unroll 1") for (unsigned o = (threadIdx.x & 31u) * 16u; o < bytes; o += 512u) { unsigned a, b, c, d;\n    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(src + o) : "memory");\n    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;" ::"l"((char*)dst + o), "r"(a), "r"(b), "r"(c), "r"(d), "l"(pol) : "memory"); } }\n__device__ __forceinline__ void tma_ld_h(unsigned dst, const void* src, unsigned bytes, unsigned, u64 pol) {\n  _Pragma("unroll 1") for (unsigned o = (threadIdx.x & 31u) * 16u; o < bytes; o += 512u)\n    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst + o), "l"((const char*)src + o), "l"(pol) : "memory"); }\n'
WAIT_80 = '__device__ __forceinline__ void mb_wait(unsigned mb, unsigned par) {\n  asm volatile("{\\n .reg .pred p;\\n W%=:\\n mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\\n @!p bra W%=;\\n}" ::"r"(mb), "r"(par) : "memory"); }\n'

def lower(src, npw=1, roles=None, regs=0):
    """src: persist.persist output of a persistent TMA config (pw3, onewait, pm, ct + cp, NSI = 4).  npw: producer warps (1..3).
    roles: dict role -> warp, roles 'S' (stream loads, thread 256), 'X' (XP/YH loads + polls, 288), 'O' (stores + publish, 320).
    regs: maxrregcount hint for __launch_bounds__ min blocks (0 = from the thread count)."""
    roles = roles or {'S': 8, 'X': 8, 'O': 8}
    assert 1 <= npw <= 3 and all((8 <= w < 8 + npw for w in roles.values()))
    pubw = 'publisher warp' in src
    assert not pubw or npw == 3, 'the pubw publisher is warp 11: needs npw = 3'
    nt = 256 + 32 * npw + (32 if pubw else 0)
    nbar = 256 + 32 * npw
    assert '#define SPM 1' in src and '#define SCP 1' in src and ('#define SPW 3' in src)
    a = src.index('__device__ __forceinline__ void mb_expect(')
    b = src.index('__device__ __forceinline__ void tma_st(')
    b = src.index('\n', src.index('\n', b) + 1) + 1
    src = src[:a] + HELPERS_80 + src[b:]
    if 'tma_st_h(void*' in src:
        a = src.index('__device__ __forceinline__ void tma_st_h(')
        b = src.index('__device__ __forceinline__ void mb_wait(')
        src = src[:a] + HINTS_80 + src[b:]
    a = src.index('__device__ __forceinline__ void mb_wait(')
    b = src.index('\n', src.index('\n', a) + 1) + 1
    src = src[:a] + WAIT_80 + src[b:]
    assert 'cp.async.bulk.shared' not in src.split('__global__')[0]
    bounds = f'__maxnreg__({regs})' if regs else f'__launch_bounds__({nt}, 1)'
    assert not regs or (regs * nt <= 65536 and regs % 8 == 0), (regs, nt)
    src = subn(src, '__global__ void __launch_bounds__(384, 1) stream(KP P, const __grid_constant__ CUtensorMap tmB, const __grid_constant__ CUtensorMap tmC)', f'__global__ void {bounds} stream(KP P)')
    src = subn(src, '#define SNTHR 384', f'#define SNTHR {nt}\n#define SM80 1\n#define SNPW {npw}')
    src = re.sub('for \\(int i = me; i < (\\d+); i \\+= 384\\) smem\\[i\\] = 0;', f'for (int i = me; i < \\1; i += {nt}) smem[i] = 0;', src)
    n = src.count('bar.sync 1, 352;')
    assert n >= 4, n
    src = src.replace('bar.sync 1, 352;', f'bar.sync 1, {nbar};')
    src = re.sub('[ \\t]*asm volatile\\(\\"setmaxnreg\\.(inc|dec)\\.sync\\.aligned\\.u32 \\d+;\\" ::: \\"memory\\"\\);[^\\n]*\\n', '', src)
    assert 'setmaxnreg' not in src
    src = re.sub('[ \\t]*asm volatile\\(\\"fence\\.proxy\\.async\\.shared::cta;\\" ::: \\"memory\\"\\);\\n', '', src)
    assert 'fence.proxy' not in src, 'fence.proxy left'
    src = subn(src, '    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n', '')
    src = re.sub('(mbarrier\\.init\\.shared::cta\\.b64 \\[%0\\], )3;', '\\g<1>96;', src)
    assert src.count('], 96;') == 1
    n = src.count('tma_ld4(d_, &tmB, 0, (int)cs[1], (int)zi_, (int)cs[2], m_);')
    assert n >= 2, n
    src = src.replace('tma_ld4(d_, &tmB, 0, (int)cs[1], (int)zi_, (int)cs[2], m_);', 'box_ld(d_, (const char*)P.fld + (((long)cs[2] * P.ROWREC + zi_ * P.PLREC + (long)cs[1] * SS) * 24), m_);')
    assert 'tma_ld4' not in src.split('__global__')[1] and 'tmC' not in src.split('__global__')[1]
    body = src.split('__global__', 1)
    j0 = body[1].count('m_); } }') + body[1].count('pol_first()); } }')
    t = body[1].replace('m_); } }', 'm_); mb_arr(m_); } }').replace('pol_first()); } }', 'pol_first()); mb_arr(m_); } }')
    assert t.count('mb_arr(m_)') == j0 and j0 == t.count('mb_expect(m_,'), (j0, t.count('mb_expect(m_,'))
    src = body[0] + '__global__' + t
    src = src.replace('asm volatile("cp.async.bulk.commit_group;" ::: "memory"); asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");', '')
    src = src.replace('asm volatile("cp.async.bulk.commit_group;" ::: "memory"); asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");', '')
    src = subn(src, 'asm volatile("cp.async.bulk.wait_group 1;" ::: "memory"); __threadfence(); stv((unsigned*)cs[11], (unsigned)(b - 1 + 1)); }', "__threadfence(); __syncwarp(); if ((threadIdx.x & 31) == 0) stv((unsigned*)cs[11], (unsigned)(b - 1 + 1)); }      // sm80: every lane's stores first")
    src = subn(src, '        __threadfence(); stv((unsigned*)cs[11], 0x3fffffffu);', "        __threadfence(); __syncwarp(); if ((threadIdx.x & 31) == 0) stv((unsigned*)cs[11], 0x3fffffffu);      // sm80: every lane's stores first")
    assert 'cp.async.bulk' not in src, [l for l in src.splitlines() if 'cp.async.bulk' in l][:3]
    src = subn(src, 'const unsigned cf = ((lane == 0 && q >= 8 && q < 11) ? 1u : 0u) + opq();', f'const unsigned cf = ((q >= 8 && q < {8 + npw}) ? 1u : 0u) + opq();      // sm80: producer WARPS (warp-cooperative copies)')
    src = subn(src, '    if (q < 11) {\n', f'    if (q < {8 + npw}) {{\n')
    th = {'S': 256, 'X': 288, 'O': 320}
    for r_, t_ in th.items():
        w_ = roles[r_]
        src = src.replace(f'else if (me == {t_}) {{', f'if (q == {w_}) {{')
        src = src.replace(f'if (me == {t_}) {{', f'if (q == {w_}) {{')
        src = src.replace(f'if (me == {t_}) cs[14] = spins;', f'if (q == {w_} && lane == 0) cs[14] = spins;')
    assert re.search('me == (288|320)', src) is None, re.findall('.*me == (?:288|320).*', src)[:3]
    n = len(re.findall('for \\(int j_ = 0; j_ < (\\d+); \\+\\+j_\\) asm volatile\\(\\"discard\\.global\\.L2', src))
    src = re.sub('for \\(int j_ = 0; j_ < (\\d+); \\+\\+j_\\) asm volatile\\(\\"discard\\.global\\.L2', lambda m: f'for (int j_ = (int)(threadIdx.x & 31); j_ < {m.group(1)}; j_ += 32) asm volatile("discard.global.L2', src)
    assert n == 2 and src.count('j_ += 32) asm volatile("discard') == 2
    src = src.replace('// persistent trips', f'// sm_80 lowering (cp.async warp-cooperative feed, {npw} producer warp(s), roles {roles}, {nt} threads) applied to:\n// persistent trips', 1)
    return src + f'#define SM80_NT {nt}\n'

def fold(src, wmap=None, waiter=0, ablate='', rpos='mid', dst=0):
    """npw = 0: NO producer warps.  Input: lower(src, 3, {'S': 8, 'X': 9, 'O': 10}).  The producer loop's per-beat role block (stores,
    stream loads, XP/YH loads) is moved into the compute loop right after each middle barrier and its epilogue after the compute loop;
    roles run on COMPUTE warps given by wmap: F (field stream job), C (coefficient + history job), X (XP/YH polls + loads + discards),
    O (stores + publish).  256 threads, __launch_bounds__(256, 1): up to 255 registers (sm_80 assigns warps round robin to the 4 SM
    sub-partitions, 16,384 registers each, so 9..12 warps cap a thread at 168 -- measured: npw >= 1 builds spill, 3x slower).
    The data wait of slot b (one thread, `waiter`) precedes the middle barrier, as in the producer form."""
    w = dict(F=3, C=4, X=1, O=2)
    w.update(wmap or {})
    assert '__launch_bounds__(352, 1)' in src and '#define SNPW 3' in src
    i0 = src.index('  if (q >= 8) {\n')
    i1 = src.index('  } else {\n', i0)
    prod = src[i0:i1]
    a = prod.index('        if (cf) {\n') + len('        if (cf) {\n')
    b = prod.index('        }\n        __syncwarp();\n      }\n', a)
    roles = prod[a:b]
    e0 = prod.index('      if (q == 10) {\n', b)
    e1 = prod.index('      asm volatile("bar.sync 1, 352;" ::: "memory");\n', e0)
    epi = prod[e0:e1]
    wl = re.search('if \\(me == 256\\) (mb_wait\\(mbs \\+ 8 \\* \\(\\(unsigned\\)b & \\d+u\\), \\(\\(unsigned\\)b >> \\d+\\) & 1u\\);)', prod).group(1)
    cj = '              { const int n_ = b + '
    k1 = roles.index('          if (q == 8) {\n')
    k2 = roles.index(cj, roles.index(cj, k1) + 1)
    roles = roles[:k2] + '          }\n          if (q == @C@) {\n' + roles[k2:]
    for tok, hdr in (('noO', '          if (q == 10) {\n'), ('noF', '          if (q == 8) {\n'), ('noC', '          if (q == @C@) {\n'), ('noX', '          if (q == 9) {\n')):
        pass
    rm = lambda t: t.replace('if (q == 8) {', f"if (q == {w['F']}) {{").replace('if (q == 9) {', f"if (q == {w['X']}) {{").replace('if (q == 10) {', f"if (q == {w['O']}) {{").replace('@C@', str(w['C'])).replace('if (q == 9 && lane == 0) cs[14]', f"if (q == {w['X']} && lane == 0) cs[14]")
    roles, epi = (rm(roles), rm(epi))
    pre = src[:i0]
    p1 = pre.index('  if (q == 8) {\n')
    p2 = pre.index('  if (q == 8) {\n', p1 + 1)
    pre = pre[:p1] + f"  if (q == {w['F']}) {{\n" + pre[p1 + 16:p2] + f"  if (q == {w['C']}) {{\n" + pre[p2 + 16:]
    pre = subn(pre, '  if (q == 9) {\n', f"  if (q == {w['X']}) {{\n")
    pre = subn(pre, 'const unsigned cf = ((q >= 8 && q < 11) ? 1u : 0u) + opq();', 'const unsigned cf = 1u + opq();      // sm80 fold: roles on compute warps')
    rest = src[i1 + len('  } else {\n'):]
    mid = '      asm volatile("bar.sync 1, 352;" ::: "memory");      // pw: middle barrier with the producer warps (data of slot b has landed)\n'
    n = rest.count(mid)
    assert n == 2, n
    pr6 = '      if (me == 0) { const u64 c_ = (u64)clock64(); cs[22] += c_ - cs[15]; cs[15] = c_; }      // prof 6: data wait\n' if 'cs[15] = c_; }' in rest else ''
    roles = '        u64* const cq_ = cs + opq();\n' + roles.replace('cs[', 'cq_[')
    if rpos == 'end':
        rest = rest.replace(mid, f'      if (me == {waiter}) {wl}      // sm80 fold: data of slot b landed\n{pr6}      __syncthreads();\n      if (cf) {{\n      }}\n')
        lh = '      if (q == 7) { const unsigned* p_ = reinterpret_cast<const unsigned*>(rin8'
        assert rest.count(lh) == 2
        rest = rest.replace(lh, f'      if (cf) {{      // sm80 fold rpos=end\n{roles}      }}\n' + lh)
    end = '  asm volatile("bar.sync 1, 352;" ::: "memory");\n  asm volatile("bar.sync 1, 352;" ::: "memory");\n'
    rest = subn(rest, end, '  __syncthreads();\n' + epi + '  __syncthreads();\n')
    tail = '  }\n  __syncthreads();\n  }      // trip\n'
    rest = subn(rest, tail, '  __syncthreads();\n  }      // trip\n')
    src = pre + rest
    src = subn(src, '__launch_bounds__(352, 1)', '__launch_bounds__(256, 1)')
    src = subn(src, '#define SNTHR 352', '#define SNTHR 256')
    src = subn(src, '#define SNPW 3', '#define SNPW 0')
    src = subn(src, '#define SM80_NT 352', '#define SM80_NT 256')
    src = re.sub('(for \\(int i = me; i < \\d+; i \\+= )352(\\) smem\\[i\\] = 0;)', '\\g<1>256\\2', src)
    assert 'bar.sync 1, 352' not in src and 'i += 352' not in src and ('q >= 8' not in src)
    return src.replace('// sm_80 lowering', f'// sm_80 fold (no producer warps: roles {w} on compute warps, 256 threads)\n// sm_80 lowering', 1)
HELPERS_CTA = '// sm80 spread: CTA-cooperative copies (all 256 threads; chunk i = thread, thread + 256, ...)\n__device__ __forceinline__ void tma_ld_a(unsigned dst, const void* src, unsigned bytes, unsigned) {\n  _Pragma("unroll 1") for (unsigned o = threadIdx.x * 16u; o < bytes; o += 4096u)\n    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + o), "l"((const char*)src + o) : "memory"); }\n__device__ __forceinline__ void tma_ld_h_a(unsigned dst, const void* src, unsigned bytes, unsigned, u64 pol) {\n  _Pragma("unroll 1") for (unsigned o = threadIdx.x * 16u; o < bytes; o += 4096u)\n    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst + o), "l"((const char*)src + o), "l"(pol) : "memory"); }\n__device__ __forceinline__ void box_ld_a(unsigned dst, const void* src, unsigned) {\n  const unsigned i = (threadIdx.x + 148u) & 255u;      // threads 108..215 (a different set than the field chunks\' tail)\n  if (i < 9u * (SGH * 24u / 16u)) { const unsigned p = i / (SGH * 24u / 16u), c = i % (SGH * 24u / 16u);\n    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + p * (SGH * 24u) + 16u * c), "l"((const char*)src + p * (SS * 24u) + 16u * c) : "memory"); } }\n__device__ __forceinline__ void tma_st_h_a(void* dst, unsigned src, unsigned bytes, u64 pol) {\n  _Pragma("unroll 1") for (unsigned o = threadIdx.x * 16u; o < bytes; o += 4096u) { unsigned a, b, c, d;\n    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(src + o) : "memory");\n    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;" ::"l"((char*)dst + o), "r"(a), "r"(b), "r"(c), "r"(d), "l"(pol) : "memory"); } }\n'
HELPERS_P = '// sm80 split2: a job shared by two warps; pt = 0 / 1 selects the half (16-B chunk i of the job: i % 64 in [32 pt, 32 pt + 32))\n__device__ __forceinline__ void tma_ld_p(unsigned pt, unsigned dst, const void* src, unsigned bytes, unsigned) {\n  _Pragma("unroll 1") for (unsigned o = ((threadIdx.x & 31u) + 32u * pt) * 16u; o < bytes; o += 1024u)\n    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + o), "l"((const char*)src + o) : "memory"); }\n__device__ __forceinline__ void tma_ld_h_p(unsigned pt, unsigned dst, const void* src, unsigned bytes, unsigned, u64 pol) {\n  _Pragma("unroll 1") for (unsigned o = ((threadIdx.x & 31u) + 32u * pt) * 16u; o < bytes; o += 1024u)\n    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst + o), "l"((const char*)src + o), "l"(pol) : "memory"); }\n__device__ __forceinline__ void tma_st_h_p(unsigned pt, void* dst, unsigned src, unsigned bytes, u64 pol) {\n  _Pragma("unroll 1") for (unsigned o = ((threadIdx.x & 31u) + 32u * pt) * 16u; o < bytes; o += 1024u) { unsigned a, b, c, d;\n    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(src + o) : "memory");\n    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;" ::"l"((char*)dst + o), "r"(a), "r"(b), "r"(c), "r"(d), "l"(pol) : "memory"); } }\n__device__ __forceinline__ void tma_st_p(unsigned pt, void* dst, unsigned src, unsigned bytes) {\n  _Pragma("unroll 1") for (unsigned o = ((threadIdx.x & 31u) + 32u * pt) * 16u; o < bytes; o += 1024u) { unsigned a, b, c, d;\n    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(src + o) : "memory");\n    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"((char*)dst + o), "r"(a), "r"(b), "r"(c), "r"(d) : "memory"); } }\n__device__ __forceinline__ void box_ld_p(unsigned pt, unsigned dst, const void* src, unsigned) {\n  _Pragma("unroll 1") for (unsigned i = (threadIdx.x & 31u) + 32u * pt; i < 9u * (SGH * 24u / 16u); i += 64u) { const unsigned p = i / (SGH * 24u / 16u), c = i % (SGH * 24u / 16u);\n    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + p * (SGH * 24u) + 16u * c), "l"((const char*)src + p * (SS * 24u) + 16u * c) : "memory"); } }\n'

def split2(src, pairs):
    """pairs: {first warp: second warp} for the F and C jobs of a fold() kernel (e.g. {3: 5, 4: 6}): each job is shared by two warps."""
    a = src.index('__device__ __forceinline__ void mb_wait(')
    src = src[:a] + HELPERS_P + src[a:]
    g0, g1 = src.split('__global__', 1)
    for w1, w2 in pairs.items():
        for ind in ('  ', '          '):
            hdr = f'{ind}if (q == {w1}) {{\n'
            pos = 0
            while True:
                i = g1.find('\n' + hdr, pos)
                if i < 0:
                    break
                i += 1
                j = g1.index(f'\n{ind}}}\n', i)
                blk = g1[i + len(hdr):j]
                assert 'tma_ld' in blk or 'box_ld' in blk or 'tma_st' in blk, blk[:200]
                if 'tma_st' in blk:
                    assert 'stv(' not in blk, 'split O only with the spread publish (publish at the middle barrier)'
                blk = blk.replace('tma_st_h(', 'tma_st_h_p(pt_, ').replace('tma_st(', 'tma_st_p(pt_, ').replace('tma_ld_h(', 'tma_ld_h_p(pt_, ').replace('tma_ld(', 'tma_ld_p(pt_, ').replace('box_ld(', 'box_ld_p(pt_, ')
                new = f'{ind}if (q == {w1} || q == {w2}) {{ const unsigned pt_ = (q == {w2}) ? 1u : 0u;      // sm80 split2\n' + blk
                g1 = g1[:i] + new + g1[j:]
                pos = i + len(new)
    assert g1.count('], 96;') == 1
    nl = sum((1 for w1 in pairs if f'if (q == {w1} || q == {pairs[w1]}) {{ const unsigned pt_ = (q == {pairs[w1]}) ? 1u : 0u;      // sm80 split2\n' in g1 and 'tma_ld' in g1.split(f'if (q == {w1} || q == {pairs[w1]})')[1][:600]))
    g1 = g1.replace('], 96;', f'], {96 + 32 * nl};')
    return g0 + '__global__' + g1

def pubmid(src):
    """Move the progress publish out of the store warp (so stores can be shared by several warps): v = b is published right after
    the middle barrier of beat b by thread 0 with fence.acq_rel.gpu (every store of beats <= b - 1 was issued before that barrier);
    DONE after the final barrier.  Same values as the TMA form (see spread())."""
    g0, g1 = src.split('__global__', 1)
    pm = int(re.search('if \\(b >= 1 && \\(b & (\\d+)\\) == 0\\) \\{\\n                __threadfence\\(\\); __syncwarp\\(\\); if \\(\\(threadIdx.x & 31\\) == 0\\) stv\\(\\(unsigned\\*\\)cq_\\[11\\]', g1).group(1))
    old_pub = f"              if (b >= 1 && (b & {pm}) == 0) {{\n                __threadfence(); __syncwarp(); if ((threadIdx.x & 31) == 0) stv((unsigned*)cq_[11], (unsigned)(b - 1 + 1)); }}      // sm80: every lane's stores first\n"
    n = g1.count(old_pub)
    assert n == 2, n
    g1 = g1.replace(old_pub, '')
    newpub = f'      if (b >= 1 && (b & {pm}) == 0 && me == 0)' + ' { asm volatile("fence.acq_rel.gpu;" ::: "memory"); stv((unsigned*)cs[11], (unsigned)b); }      // sm80 pubmid: publish v = b\n'
    g1 = g1.replace('      __syncthreads();\n      if (cf) {\n', '      __syncthreads();\n' + newpub + '      if (cf) {\n')
    assert g1.count(newpub) == 2
    old_done = "        __threadfence(); __syncwarp(); if ((threadIdx.x & 31) == 0) stv((unsigned*)cs[11], 0x3fffffffu);      // sm80: every lane's stores first\n"
    assert g1.count(old_done) == 1
    g1 = g1.replace(old_done, '')
    tail = '  __syncthreads();\n  if (me == 0) { long long* t_'
    assert g1.count(tail) == 1
    g1 = g1.replace(tail, '  __syncthreads();\n  if (me == 0) { asm volatile("fence.acq_rel.gpu;" ::: "memory"); stv((unsigned*)cs[11], 0x3fffffffu); }      // sm80 pubmid: DONE after every thread\'s final stores\n  if (me == 0) { long long* t_')
    return g0 + '__global__' + g1
