"""Minimal OpenCL MLP engine (forward, backward, Adam) for the aim/tap heads.

Targets the Mesa *rusticl* OpenCL runtime on the gfx902 (Vega 11) iGPU.

Everything is done with a single tiled GEMM kernel plus tiny elementwise
kernels, so we only rely on add/mul/max (avoiding libclc math bugs).  The
layout is plain row-major float32; a linear layer stores ``W`` as
``(in, out)`` with leading dimension ``out`` and ``b`` as ``(out,)``.

    W = ocl.MLP([indim, 128, 128, 2], ["relu", "relu", "none"], seed=0)
    W.forward(X)          # X: (M, indim) float32 buffer
    W.backward(dY)        # dY: (M, outdim) float32 buffer
    W.step(lr)            # Adam update on all params
"""
from __future__ import annotations

import numpy as np
import pyopencl as cl

TS = 16  # GEMM tile

_KERNELS = r"""
#define TS 16
__kernel void gemm(const int M, const int N, const int K, const float alpha,
        __global const float* A, const int lda, const int ta,
        __global const float* B, const int ldb, const int tb,
        const float beta, __global float* C, const int ldc)
{
    __local float As[TS][TS];
    __local float Bs[TS][TS];
    int gx = get_group_id(0), gy = get_group_id(1);
    int tx = get_local_id(0),  ty = get_local_id(1);
    int row = gy * TS + ty, col = gx * TS + tx;
    float acc = 0.f;
    int nt = (K + TS - 1) / TS;
    for (int t = 0; t < nt; ++t) {
        int ar = row, ac = t * TS + tx;
        float a = 0.f;
        if (ar < M && ac < K) a = ta ? A[ac * lda + ar] : A[ar * lda + ac];
        As[ty][tx] = a;
        int br = t * TS + ty, bc = col;
        float b = 0.f;
        if (br < K && bc < N) b = tb ? B[bc * ldb + br] : B[br * ldb + bc];
        Bs[ty][tx] = b;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TS; ++k) acc += As[ty][k] * Bs[k][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        float c = (beta == 0.f) ? 0.f : beta * C[row * ldc + col];
        C[row * ldc + col] = alpha * acc + c;
    }
}

__kernel void add_bias(__global float* Y, const int MN, const int N,
        __global const float* b)
{
    int i = get_global_id(0);
    if (i < MN) Y[i] += b[i % N];
}

__kernel void colsum(__global const float* Z, __global float* db,
        const int M, const int N)
{
    int j = get_global_id(0);
    if (j < N) {
        float s = 0.f;
        for (int i = 0; i < M; ++i) s += Z[i * N + j];
        db[j] = s;
    }
}

__kernel void colsum_part(__global const float* Z, __global float* part,
        const int M, const int N, const int P)
{
    int j = get_global_id(0);
    int p = get_global_id(1);
    if (j >= N || p >= P) return;
    int lo = (int)((long)M * p / P), hi = (int)((long)M * (p + 1) / P);
    float s = 0.f;
    for (int i = lo; i < hi; ++i) s += Z[i * N + j];
    part[p * N + j] = s;
}

__kernel void colsum_fin(__global const float* part, __global float* db,
        const int N, const int P)
{
    int j = get_global_id(0);
    if (j >= N) return;
    float s = 0.f;
    for (int p = 0; p < P; ++p) s += part[p * N + j];
    db[j] = s;
}

__kernel void relu_fwd(__global const float* x, __global float* y, const int n)
{
    int i = get_global_id(0);
    if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}

__kernel void relu_bwd(__global const float* z, __global const float* dA,
        __global float* dZ, const int n)
{
    int i = get_global_id(0);
    if (i < n) dZ[i] = z[i] > 0.f ? dA[i] : 0.f;
}

__kernel void smoothl1_grad(__global const float* pred, __global const float* tgt,
        __global float* grad, const int n, const float beta, const float scale)
{
    int i = get_global_id(0);
    if (i < n) {
        float d = pred[i] - tgt[i];
        float g = (fabs(d) < beta) ? d / beta : (d > 0.f ? 1.f : -1.f);
        grad[i] = g * scale;
    }
}

__kernel void adam(__global float* W, __global const float* g,
        __global float* m, __global float* v, const float lr,
        const float b1, const float b2, const float eps,
        const float bc1, const float bc2, const float wd, const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        float gi = g[i] + wd * W[i];
        m[i] = b1 * m[i] + (1.f - b1) * gi;
        v[i] = b2 * v[i] + (1.f - b2) * gi * gi;
        W[i] -= lr * (m[i] / bc1) / (sqrt(v[i] / bc2) + eps);
    }
}
"""

_ctx = None
_q = None
_prog = None
_K = None


def _setup():
    global _ctx, _q, _prog, _K
    if _ctx is not None:
        return
    dev = None
    for p in cl.get_platforms():
        for d in p.get_devices():
            if d.type & cl.device_type.GPU:
                dev = d
                break
        if dev is not None:
            break
    _ctx = cl.Context([dev]) if dev is not None else cl.create_some_context(interactive=False)
    _q = cl.CommandQueue(_ctx)
    _prog = cl.Program(_ctx, _KERNELS).build()
    _K = {n: cl.Kernel(_prog, n) for n in
          ("gemm", "add_bias", "colsum", "colsum_part", "colsum_fin",
           "relu_fwd", "relu_bwd", "smoothl1_grad", "adam")}


MF = cl.mem_flags


def device_name() -> str:
    _setup()
    d = _ctx.devices[0]
    return f"{d.name}  CUs={d.max_compute_units}"


def up(a: np.ndarray) -> cl.Buffer:
    """Upload a float32 numpy array (read-only device buffer)."""
    _setup()
    a = np.ascontiguousarray(a, dtype=np.float32)
    return cl.Buffer(_ctx, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=a)


def empty(n: int) -> cl.Buffer:
    _setup()
    return cl.Buffer(_ctx, MF.READ_WRITE, n * 4)


def down(b: cl.Buffer, shape) -> np.ndarray:
    out = np.empty(int(np.prod(shape)), dtype=np.float32)
    cl.enqueue_copy(_q, out, b)
    _q.finish()
    return out.reshape(shape)


def _gemm(M, N, K, alpha, A, lda, ta, B, ldb, tb, beta, C, ldc):
    gs = (((N + TS - 1) // TS) * TS, ((M + TS - 1) // TS) * TS)
    _K['gemm'](_q, gs, (TS, TS),
               np.int32(M), np.int32(N), np.int32(K), np.float32(alpha),
               A, np.int32(lda), np.int32(ta),
               B, np.int32(ldb), np.int32(tb),
               np.float32(beta), C, np.int32(ldc))


def _relu_fwd(x, y, n):
    _K['relu_fwd'](_q, ((n + 255) // 256 * 256,), None, x, y, np.int32(n))


def _relu_bwd(z, dA, dZ, n):
    _K['relu_bwd'](_q, ((n + 255) // 256 * 256,), None, z, dA, dZ, np.int32(n))


def _add_bias(Y, M, N, b):
    mn = M * N
    _K['add_bias'](_q, ((mn + 255) // 256 * 256,), None, Y, np.int32(mn), np.int32(N), b)


def _colsum(Z, db, M, N):
    P = 32
    part = empty(N * P)
    _K['colsum_part'](_q, (N, P), None, Z, part, np.int32(M), np.int32(N), np.int32(P))
    _K['colsum_fin'](_q, ((N + 255) // 256 * 256,), None, part, db, np.int32(N), np.int32(P))


def smoothl1_grad(pred, tgt, grad, n, beta=1.0, scale=1.0):
    _setup()
    _K['smoothl1_grad'](_q, ((n + 255) // 256 * 256,), None,
                        pred, tgt, grad, np.int32(n),
                        np.float32(beta), np.float32(scale))


def _adam(W, g, m, v, lr, t, wd, n):
    b1, b2, eps = 0.9, 0.999, 1e-8
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    _K['adam'](_q, ((n + 255) // 256 * 256,), None,
               W, g, m, v, np.float32(lr), np.float32(b1), np.float32(b2),
               np.float32(eps), np.float32(bc1), np.float32(bc2),
               np.float32(wd), np.int32(n))


class MLP:
    """Fully-connected net, row-major, trained entirely on the OpenCL device."""

    def __init__(self, dims: list[int], acts: list[str], seed: int = 0):
        _setup()
        assert len(acts) == len(dims) - 1
        self.dims = list(dims)
        self.acts = list(acts)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        self.gW, self.gb = [], []
        self.mW, self.vW, self.mb, self.vb = [], [], [], []
        self._A = None
        self._Z = None
        self.t = 0
        for l in range(len(acts)):
            fin, fout = dims[l], dims[l + 1]
            scale = np.sqrt(2.0 / fin) if acts[l] == "relu" else np.sqrt(1.0 / fin)
            w = (rng.standard_normal((fin, fout)) * scale).astype(np.float32)
            bb = np.zeros(fout, np.float32)
            self.W.append(up(w))
            self.b.append(up(bb))
            self.gW.append(empty(fin * fout))
            self.gb.append(empty(fout))
            self.mW.append(empty(fin * fout))
            self.vW.append(empty(fin * fout))
            self.mb.append(empty(fout))
            self.vb.append(empty(fout))

    # -- forward ---------------------------------------------------------
    def forward(self, X: cl.Buffer, M: int) -> cl.Buffer:
        self._A = [X]
        self._Z = []
        A = X
        for l in range(len(self.acts)):
            fin, fout = self.dims[l], self.dims[l + 1]
            Y = empty(M * fout)
            _gemm(M, fout, fin, 1.0, A, fin, 0, self.W[l], fout, 0, 0.0, Y, fout)
            _add_bias(Y, M, fout, self.b[l])
            self._Z.append(Y)
            if self.acts[l] == "relu":
                Y2 = empty(M * fout)
                _relu_fwd(Y, Y2, M * fout)
                A = Y2
            else:
                A = Y
            self._A.append(A)
        return A

    # -- backward --------------------------------------------------------
    def backward(self, dY: cl.Buffer, M: int) -> None:
        dA = dY
        for l in range(len(self.acts) - 1, -1, -1):
            fin, fout = self.dims[l], self.dims[l + 1]
            if self.acts[l] == "relu":
                dZ = empty(M * fout)
                _relu_bwd(self._Z[l], dA, dZ, M * fout)
            else:
                dZ = dA
            # dW = A_prev^T @ dZ
            _gemm(fin, fout, M, 1.0, self._A[l], fin, 1,
                  dZ, fout, 0, 0.0, self.gW[l], fout)
            # db = sum_rows(dZ)
            _colsum(dZ, self.gb[l], M, fout)
            if l > 0:
                # dA_prev = dZ @ W^T
                dA = empty(M * fin)
                _gemm(M, fin, fout, 1.0, dZ, fout, 0,
                      self.W[l], fout, 1, 0.0, dA, fin)

    # -- optim -----------------------------------------------------------
    def step(self, lr: float, wd: float = 0.0) -> None:
        self.t += 1
        for l in range(len(self.acts)):
            n = self.dims[l] * self.dims[l + 1]
            _adam(self.W[l], self.gW[l], self.mW[l], self.vW[l], lr, self.t, wd, n)
            _adam(self.b[l], self.gb[l], self.mb[l], self.vb[l], lr, self.t, wd, self.dims[l + 1])

    def state(self) -> dict:
        return {"W": [down(w, (self.dims[l], self.dims[l + 1]))
                      for l, w in enumerate(self.W)],
                "b": [down(b, (self.dims[l + 1],)) for l, b in enumerate(self.b)]}


if __name__ == "__main__":
    _setup()
    print("device:", device_name())

    # ---- GEMM check vs numpy
    rng = np.random.default_rng(0)
    M, N, K = 37, 21, 19
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C = empty(M * N)
    _gemm(M, N, K, 1.0, up(A), K, 0, up(B), N, 0, 0.0, C, N)
    got = down(C, (M, N))
    print("gemm fwd  :", np.allclose(got, A @ B, atol=1e-3))

    # ---- MLP forward vs numpy
    dims = [K, 16, 12, 2]
    acts = ["relu", "relu", "none"]
    net = MLP(dims, acts, seed=1)
    st = net.state()
    X = rng.standard_normal((M, dims[0])).astype(np.float32)
    out = net.forward(up(X), M)
    got = down(out, (M, dims[-1]))
    ref = X
    for l, a in enumerate(acts):
        ref = ref @ st["W"][l] + st["b"][l]
        if a == "relu":
            ref = np.maximum(ref, 0)
    print("mlp fwd   :", np.allclose(got, ref, atol=1e-3))

    # ---- backward grad check vs numpy (SmoothL1 grad, scale 1/M)
    D = dims[-1]
    pred = got
    tgt = rng.standard_normal((M, D)).astype(np.float32)
    n = M * D
    gpred = empty(n)
    smoothl1_grad(out, up(tgt), gpred, n, 1.0, 1.0 / M)
    dpred = down(gpred, (M, D))
    net.backward(gpred, M)
    gW = [down(g, (dims[l], dims[l + 1])) for l, g in enumerate(net.gW)]
    gb = [down(b, (dims[l + 1],)) for l, b in enumerate(net.gb)]

    def fwd_np(X, W, b):
        A = X
        Zs = []
        for l, a in enumerate(acts):
            Z = A @ W[l] + b[l]
            Zs.append(Z)
            A = np.maximum(Z, 0) if a == "relu" else Z
        return A, Zs

    def bwd_np(X, tgt, W, b):
        A, Zs = fwd_np(X, W, b)
        d = A - tgt
        dA = np.where(np.abs(d) < 1, d, np.sign(d)) / M
        gW = [None] * len(W)
        gb = [None] * len(b)
        for l in range(len(W) - 1, -1, -1):
            dZ = dA * (Zs[l] > 0) if acts[l] == "relu" else dA
            Ain = X if l == 0 else (np.maximum(Zs[l - 1], 0) if acts[l - 1] == "relu" else Zs[l - 1])
            gW[l] = Ain.T @ dZ
            gb[l] = dZ.sum(0)
            dA = dZ @ W[l].T
        return gW, gb

    rW = st["W"]; rB = st["b"]
    nW, nB = bwd_np(X, tgt, rW, rB)
    ok = all(np.allclose(gW[l], nW[l], atol=1e-3) for l in range(len(rW)))
    print("backward  :", ok, "" if ok else [float(np.abs(gW[l]-nW[l]).max()) for l in range(len(rW))])
    okb = all(np.allclose(gb[l], nB[l], atol=1e-3) for l in range(len(rB)))
    print("backward b:", okb, "" if okb else [float(np.abs(gb[l]-nB[l]).max()) for l in range(len(rB))])
