# SPDX-License-Identifier: MIT
"""gen3: the IR of one pillar trip of the z-stream engine.

A block owns an x pair per thread and 24 y lanes per warp-row and streams along z, holding a rolling window of k = 7 time
levels in registers (the "half-pair" engine: one cell crosses the x seam per level).  The IR is a list of operations per beat
(loads, shuffles, mailbox exchanges, the H and E half steps, x take-over, y halo); cudagen6 turns it into CUDA.  With the `cx`
flag the coefficients are not carried in the x hand-off record but re-read from the coefficient array (history ring).
"""
import sys, collections
F6 = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
C6 = ['Cax', 'Cay', 'Caz', 'Cbx', 'Cby', 'Cbz']
GOFF = 8

class Gen:

    def __init__(self, k=7, mode='shift', S=0, ctl='', D=4, DX=2, DY=2, WGP=2, NSI=8, G=1, ctlw=0, sph=0, split=0, wc=4):
        self.sph, self.split, self.wc = (sph, split, wc)
        self.k, self.R, self.mode, self.ctl = (k, 2 * k + 2, mode, ctl)
        if not S:
            S = 32 - k & ~1
        self.S, self.Gh = (S, 32 - S)
        assert S % 2 == 0 and 32 - S >= k and (S > 32 - S), 'S even, ghost lanes >= k'
        self.D, self.DX, self.DY, self.WGP, self.NSI, self.G, self.ctlw = (D, DX, DY, WGP, NSI, G, ctlw)
        assert NSI in (4, 8) and 0 <= D <= NSI - 2 and (1 <= DX <= 2) and (0 <= WGP <= 6) and (G >= 1)
        self.ncopy = self.R if mode == 'rot' else 2
        self.w0 = -self.ncopy if mode == 'rot' else -2
        self.nt = 0
        assert mode == 'shift' and k in (3, 5, 7, 9), 'half-pair hand-off: shifted window, odd k'
        self.SH = (k - 1) // 2
        self.cx = 'cx' in ctl.split('+')
        self.XREC = 12 if self.cx else 24

    def t(self):
        self.nt += 1
        return f't{self.nt}'

    def sr(self, f, cls, j):
        return f'{f}_{cls}{j}'

    def sidx(self, cls, j):
        return (0 if cls == 'A' else self.k + 1) + j

    def state_regs(self):
        return [self.sr(f, c, j) for c in 'AB' for j in range(self.k + 1) for f in F6 + C6]

    def qswap(self, r):
        return 7 - r // 2

    def qtop(self, r):
        return (15 - r) // 2

    def qbot(self, r):
        return (8 - r // 2) % 8 if r % 2 == 0 else 8 - (r + 1) // 2

    def hhalf(self, o, r, E, Ep, Hs, Hd, mbe_i, mbh_i):
        Ex, Ey, Ez = E
        Exp, Eyp = Ep
        rec = self.t()
        ld = ('lds', rec, 'MBE', mbe_i, +1)
        if r == 0:
            o.append(('ifq', 7, [('prmt', rec, 'X0B', 'X0A', 21520)], [ld]))
        else:
            o.append(ld)
        PEz, PEy, SEz, SEx = (self.t(), self.t(), self.t(), self.t())
        norm = [('prmt', PEz, Ez, rec, 21554), ('prmt', PEy, Ey, rec, 30258)]
        if r % 2 == 1:
            T = self.t()
            o.append(('ifq', self.qswap(r), [('xown', T, r - 1, 0, -2), ('prmt', PEz, T, rec, 21520), ('prmt', PEy, T, rec, 30258)], norm))
        else:
            o += norm
        o += [('shfl', SEz, Ez, +1), ('shfl', SEx, Ex, +1)]
        for h, p, pm, m, mm in ((0, SEz, Ez, Eyp, Ey), (1, Exp, Ex, PEz, Ez), (2, PEy, Ey, SEx, Ex)):
            a, b, c = (self.t(), self.t(), self.t())
            o += [('sub', a, p, pm), ('sub', b, m, mm), ('sub', c, a, b), ('sub', Hd[h], Hs[h], c)]
        rh = self.t()
        o += [('prmt', rh, Hd[2], Hd[1], 30258), ('sts', rh, 'MBH', mbh_i)]
        o.append(('yput', r, Hd[0], Hd[2]))

    def ehalf(self, o, r, Es, Ed, H, Hm, C, mbh_i, mbe_i):
        Hx, Hy, Hz = H
        Hxm, Hym = Hm
        rec = self.t()
        ld = ('lds', rec, 'MBH', mbh_i, -1)
        hw = 2 if r < self.k - 1 else 0
        MHz, MHy, SHz, SHx = (self.t(), self.t(), self.t(), self.t())
        norm = [('prmt', MHz, rec, Hz, 21520), ('prmt', MHy, rec, Hy, 21554)]
        if r % 2 == 0:
            o.append(('ifq', self.qbot(r), [('xld', [rec], r, hw, 0)], [ld]))
            o += norm
        else:
            HAL = self.t()
            o.append(ld)
            o.append(('ifq', self.qbot(r), [('xld', [HAL], r, hw, 0), ('prmt', MHz, rec, HAL, 21520), ('prmt', MHy, rec, HAL, 30258)], norm))
        o += [('shfl', SHz, Hz, -1), ('shfl', SHx, Hx, -1)]
        o.append(('yget', r, SHz, SHx))
        for e, p, pm, m, mm in ((0, Hz, SHz, Hy, Hym), (1, Hx, Hxm, Hz, MHz), (2, Hy, MHy, Hx, SHx)):
            a, b, c, d = (self.t(), self.t(), self.t(), self.t())
            o += [('sub', a, p, pm), ('sub', b, m, mm), ('sub', c, a, b), ('mul', d, C[e], Es[e]), ('fma', Ed[e], C[3 + e], c, d)]
        re = self.t()
        o += [('prmt', re, Ed[2], Ed[1], 21520), ('sts', re, 'MBE', mbe_i)]

    def endblock(self, o, loc_cur, loc_prev):
        """half swaps.  loc_cur(r) -> (F regs, C regs) of plane w-2r after this beat; loc_prev(r) -> F regs of plane w-1-2r."""
        k = self.k
        for r in range(k - 1):
            Fc, Cc = loc_cur(r)
            Fp = loc_prev(r)
            Ex, Ey, Ez, Hx, Hy, Hz = Fc
            hi = r % 2 == 0
            ext = 30258 if hi else 21520
            in1 = 21520 if hi else 12884
            in2 = 30224 if hi else 12918
            if self.cx:
                inc = 30224 if hi else 12884
                w = [self.t() for _ in range(3)]
                ops = [('prmt', w[0], Ez, Ey, ext), ('prmt', w[1], Ex, Hx, ext), ('prmt', w[2], Hz, Hy, ext), ('xst', w, r)]
                a = [self.t() for _ in range(2)]
                cc = [self.t() for _ in range(6)]
                ops += [('xld', a, r, 0, 0), ('cld', cc, r), ('prmt', Ez, Ez, a[0], in1), ('prmt', Ey, Ey, a[0], in2), ('prmt', Ex, Ex, a[1], in1)]
                ops += [('prmt', Cc[i], Cc[i], cc[i], inc) for i in range(6)]
                h2 = [self.t(), self.t()]
                ops += [('xld', h2, r, 1, -1), ('prmt', Fp[5], Fp[5], h2[1], in1), ('prmt', Fp[4], Fp[4], h2[1], in2), ('prmt', Fp[3], Fp[3], h2[0], in2)]
                o.append(('ifq', self.qswap(r), ops, []))
                continue
            w = [self.t() for _ in range(6)]
            ops = [('prmt', w[0], Ez, Ey, ext), ('prmt', w[1], Ex, Cc[0], ext), ('prmt', w[2], Hz, Hy, ext), ('prmt', w[3], Hx, Cc[1], ext), ('prmt', w[4], Cc[2], Cc[3], ext), ('prmt', w[5], Cc[4], Cc[5], ext), ('xst', w, r)]
            a = [self.t() for _ in range(6)]
            ops += [('xld', a[0:4], r, 0, 0), ('xld', a[4:6], r, 4, 0), ('prmt', Ez, Ez, a[0], in1), ('prmt', Ey, Ey, a[0], in2), ('prmt', Ex, Ex, a[1], in1), ('prmt', Cc[0], Cc[0], a[1], in2), ('prmt', Cc[1], Cc[1], a[3], in2), ('prmt', Cc[2], Cc[2], a[4], in1), ('prmt', Cc[3], Cc[3], a[4], in2), ('prmt', Cc[4], Cc[4], a[5], in1), ('prmt', Cc[5], Cc[5], a[5], in2)]
            h2 = [self.t(), self.t()]
            ops += [('xld', h2, r, 2, -1), ('prmt', Fp[5], Fp[5], h2[0], in1), ('prmt', Fp[4], Fp[4], h2[0], in2), ('prmt', Fp[3], Fp[3], h2[1], in1)]
            o.append(('ifq', self.qswap(r), ops, []))
        Fc, _ = loc_cur(k - 1)
        th = self.t()
        o.append(('ifq', self.qtop(k - 1), [('prmt', th, Fc[5], Fc[4], 30258), ('xst', [th], k - 1)], []))
        o.append(('ldhalo',))

    def copy(self, c):
        k, R = (self.k, self.R)
        o = []
        act, rest = ('A', 'B') if c == 0 else ('B', 'A')
        S = lambda cl, j: ([self.sr(f, cl, j) for f in F6], [self.sr(f, cl, j) for f in C6])
        Fo, Co = S(act, k)
        Fn, Cn = S(rest, 0)
        o += [('stout', Fo + Co)]
        for r in reversed(range(k)):
            F, _ = S(act, r)
            Fd, _ = S(act, r + 1)
            Fp, _ = S(rest, r)
            self.hhalf(o, r, F[0:3], (Fp[0], Fp[1]), F[3:6], Fd[3:6], self.sidx(act, r), self.sidx(act, r + 1))
        r0 = self.t()
        o += [('prmt', r0, Fn[2], Fn[1], 21520), ('sts', r0, 'MBE', self.sidx(rest, 0))]
        o.append(('bar',))
        for r in reversed(range(k)):
            F, C = S(act, r)
            Fd, Cd = S(act, r + 1)
            Fm, _ = S(rest, r + 1)
            self.ehalf(o, r, F[0:3], Fd[0:3], Fd[3:6], (Fm[3], Fm[4]), C, self.sidx(act, r + 1), self.sidx(act, r + 1))
            o += [('mov', Cd[i], C[i]) for i in range(6)]
        o.append(('bar',))
        self.endblock(o, lambda r: S(act, r + 1), lambda r: S(rest, r + 1)[0])
        F0, C0 = S(act, 0)
        o.append(('ldin', F0 + C0))
        return o

    def body(self):
        """The IR of every copy (beat) of the trip."""
        ctl = set(self.ctl.split('+')) if self.ctl else set()
        out = []

        def filt(ops):
            res = []
            for op in ops:
                kk = op[0]
                if kk == 'ifq':
                    res.append(('ifq', op[1], filt(op[2]), filt(op[3])))
                    continue
                res.append(op)
            return res
        for c in range(self.ncopy):
            out.append(filt(self.copy(c)))
        return out

def parse_cfg(s):
    cfg = dict((x.split('=') for x in s.split(',') if x))
    return Gen(k=int(cfg.get('k', 7)), mode=cfg.get('mode', 'shift'), S=int(cfg.get('S', 0)), ctl=cfg.get('ctl', ''), D=int(cfg.get('D', 4)), DX=int(cfg.get('DX', 2)), DY=int(cfg.get('DY', 2)), WGP=int(cfg.get('WGP', 2)), NSI=int(cfg.get('NSI', 8)), G=int(cfg.get('G', 1)), ctlw=int(cfg.get('ctlw', 0)), sph=int(cfg.get('sph', 0)), split=int(cfg.get('split', 0)), wc=int(cfg.get('wc', 4)))
