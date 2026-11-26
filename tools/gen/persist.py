# SPDX-License-Identifier: MIT
"""persist: turn a generated kernel text into a persistent kernel (source transform, every edit asserted to apply once).

  * the kernel body becomes the body of `for (;;)`: each trip takes one block index (one pillar-pass), either
    blockIdx.x + it * gridDim.x (P.pmode 0) or a global ticket atomicAdd(P.tick, 1) (P.pmode 1);
  * the producer and compute warpgroups return to the launch register split at the end of each trip;
  * mbarriers are invalidated and re-initialised for each trip;
  * progress is published every `pub` beats, optionally by a dedicated publisher warp (pubw);
  * two more guard words ord[12], ord[13] (DONE waits before a block starts);
  * L2 cache hints on the stream loads and stores (policy l2h).
Deadlock freedom: every wait of block b is on a block < b (host-checked) and every CTA runs its blocks in increasing order,
so the smallest unfinished block always has its producers done or running.
"""

def sub1(src, old, new):
    n = src.count(old)
    assert n == 1, (n, old[:120])
    return src.replace(old, new)

def persist(src, pub=4, pubw=False, bpw=0, instr=False, ppoll=False, pfence='sc', l2h=0):
    src = sub1(src, 'struct KP { Rec* fld;', 'struct KP { int nblk, pmode; unsigned* tick; Rec* fld;')
    src = sub1(src, '  extern __shared__ __align__(128) unsigned smem[];\n', '  extern __shared__ __align__(128) unsigned smem[];\n  __shared__ int bid_s;\n#pragma unroll 1\n  for (int it_ = 0; ; ++it_) {      // persistent trip: one block index per trip\n  if (threadIdx.x == 0) bid_s = P.pmode == 1 ? (int)atomicAdd(P.tick, 1u) : (int)blockIdx.x + it_ * (int)gridDim.x;\n  __syncthreads();\n  const int bid = bid_s;\n  if (bid >= P.nblk) break;\n')
    assert src.count('(long)blockIdx.x') == 2, src.count('(long)blockIdx.x')
    src = src.replace('(long)blockIdx.x', '(long)bid')
    assert 'blockIdx' not in src.split('bid_s = ')[1].split('\n', 1)[1], 'blockIdx left in the body'
    import re as _re0
    nmb = _re0.search('for \\(unsigned j = 0; j < (\\d+); \\+\\+j\\) asm volatile\\(\\"mbarrier.init.shared::cta.b64 \\[%0\\], 1;\\"', src).group(1)
    src = sub1(src, f'    for (unsigned j = 0; j < {nmb}; ++j) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"', f'    if (it_ > 0) for (unsigned j = 0; j < {nmb}; ++j) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mbs + 8 * j) : "memory");\n    for (unsigned j = 0; j < {nmb}; ++j) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"')
    src = sub1(src, 'for (int j = 9; j < 12; ++j) while (ldv(P.prog + ob[j]', 'for (int j = 9; j < 14; ++j) while (ldv(P.prog + ob[j]')
    src = sub1(src, 'if (b >= 1 && (b & 63) == 0) {', f'if (b >= 1 && (b & {pub - 1}) == 0) {{')
    src = sub1(src, '    return;\n  }\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 240;"', '    asm volatile("setmaxnreg.inc.sync.aligned.u32 168;" ::: "memory");      // back to the launch split\n  } else {\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 240;"')
    if pubw:
        src = sub1(src, '    cs[38] = (u64)(I > 0);', '    cs[44] = 0; cs[45] = 0;      // pubw: posted beat, publisher ack\n    cs[38] = (u64)(I > 0);')
        src = sub1(src, 'asm volatile("cp.async.bulk.wait_group 1;" ::: "memory"); __threadfence(); stv((unsigned*)cs[11], (unsigned)(b - 1 + 1)); }', 'asm volatile("cp.async.bulk.wait_group 1;" ::: "memory"); asm volatile("fence.acq_rel.cta;" ::: "memory"); *(volatile u64*)&cs[44] = (u64)b; }')
        src = sub1(src, '        __threadfence(); stv((unsigned*)cs[11], 0x3fffffffu);', '        *(volatile u64*)&cs[44] = 0xffffffffull; while (*(volatile u64*)&cs[45] == 0ull) __nanosleep(64);      // stop the publisher first\n        __threadfence(); stv((unsigned*)cs[11], 0x3fffffffu);')
        src = sub1(src, '    asm volatile("setmaxnreg.inc.sync.aligned.u32 168;" ::: "memory");      // back to the launch split\n', '    else if (q == 11 && lane == 0) {      // publisher warp\n      u64 last_ = 0;\n      for (;;) { const u64 v_ = *(volatile u64*)&cs[44];\n        if (v_ == 0xffffffffull) break;\n        if (v_ != last_) { asm volatile("fence.acq_rel.cta;" ::: "memory"); __threadfence(); stv((unsigned*)cs[11], (unsigned)v_); last_ = v_; } else __nanosleep(128); }\n      *(volatile u64*)&cs[45] = 1ull;\n    }\n    asm volatile("setmaxnreg.inc.sync.aligned.u32 168;" ::: "memory");      // back to the launch split\n')
    assert src.endswith('}\n')
    src = src[:-2] + '  asm volatile("setmaxnreg.dec.sync.aligned.u32 168;" ::: "memory");      // back to the launch split\n  }\n  __syncthreads();\n  }      // trip\n}\n'
    if l2h:
        import re
        src = sub1(src, '__device__ __forceinline__ void mb_wait(', '__device__ __forceinline__ u64 pol_last() { u64 p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p; }\n__device__ __forceinline__ u64 pol_first() { u64 p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }\n__device__ __forceinline__ void tma_st_h(void* dst, unsigned src, unsigned bytes, u64 pol) {\n  asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;" ::"l"(dst), "r"(src), "r"(bytes), "l"(pol) : "memory"); }\n__device__ __forceinline__ void tma_ld_h(unsigned dst, const void* src, unsigned bytes, unsigned mb, u64 pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst), "l"(src), "r"(bytes), "r"(mb), "l"(pol) : "memory"); }\n__device__ __forceinline__ void mb_wait(')
        src, n = re.subn('tma_st\\(\\(void\\*\\)\\(cs\\[([68])\\]([^;]*)\\);', 'tma_st_h((void*)(cs[\\1]\\2, pol_last());', src)
        assert n == 3, n
        if l2h >= 2:
            src, n = re.subn('tma_st\\(\\(void\\*\\)\\(cs\\[3\\]([^;]*)\\);', 'tma_st_h((void*)(cs[3]\\1, pol_first());', src)
            assert n == 1, n
            old = 'tma_ld(sb_sa + 39680 + s_ * 13824u, (const void*)(cs[32] + zi_ * (long)cs[36]), 5760u, m_);'
            n = src.count(old)
            assert n >= 2, n
            src = src.replace(old, 'tma_ld_h(sb_sa + 39680 + s_ * 13824u, (const void*)(cs[32] + zi_ * (long)cs[36]), 5760u, m_, pol_first());')
        if l2h >= 3:
            old = 'tma_ld(d_ + 1728u, (const void*)(cs[0] + zi_ * (P.PLREC * 24)), 5184u, m_);'
            n = src.count(old)
            assert n >= 2, n
            src = src.replace(old, 'tma_ld_h(d_ + 1728u, (const void*)(cs[0] + zi_ * (P.PLREC * 24)), 5184u, m_, pol_first());')
    src = src.replace('// AUTO-GENERATED', f'// persistent trips (publish every {pub} beats) applied to:\n// AUTO-GENERATED', 1)
    return src + f'#define SPUB {pub}\n#define SPERSIST 1\n#define SPUBW {int(pubw)}\n#define SBPW {bpw}\n'
