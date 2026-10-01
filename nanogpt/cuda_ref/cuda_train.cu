// Hand-written CUDA reference for the nanoGPT training step benchmarked in this repo.
//
// Same model as the Bend / PyTorch versions: 1 layer, 1 head, pre-LN, tied embeddings, tanh-GELU,
// C=16, F=64, V=32, T=14; loss = mean cross-entropy; AdamW(weight_decay=0). fp32, no cuBLAS.
// It exists only to calibrate how fast this step CAN run on the GPU (and so how far the Bend
// result is from the hardware), not as a recommended implementation.
//
// Layout: one block (256 threads) walks over several samples; per sample the activations live in
// shared memory, the 4048 parameters are read through L1/L2, and the gradient is accumulated in
// shared memory. Each block writes one partial gradient; a second kernel sums them and runs Adam.
//
//   nvcc -O3 -arch=sm_80 -o cuda_train cuda_train.cu
//   ./cuda_train params.bin B steps [verify]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define C 16
#define T 14
#define F 64
#define V 32
#define NTHREADS 256

// parameter layout (identical to nanogpt/bench_train.py::field_sizes)
#define O_WTE 0
#define O_WPE 512
#define O_LN1G 736
#define O_LN1B 752
#define O_WQ 768
#define O_BQ 1024
#define O_WK 1040
#define O_BK 1296
#define O_WV 1312
#define O_BV 1568
#define O_WO 1584
#define O_BO 1840
#define O_LN2G 1856
#define O_LN2B 1872
#define O_WFC 1888
#define O_BFC 2912
#define O_WMP 2976
#define O_BMP 4000
#define O_LNFG 4016
#define O_LNFB 4032
#define NPARAM 4048

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__constant__ int c_x[4][T];
__constant__ int c_y[4][T];

__device__ __forceinline__ float gelu_f(float x) {
  const float c = 0.79788456f;
  float t = tanhf(c * (x + 0.044715f * x * x * x));
  return 0.5f * x * (1.f + t);
}
__device__ __forceinline__ float gelu_grad(float x) {
  const float c = 0.79788456f;
  float x2 = x * x;
  float t = tanhf(c * (x + 0.044715f * x * x2));
  return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * c * (1.f + 0.134145f * x2);
}
// sum over the 16 lanes of a token (tid = t*16 + c, so a token is a half-warp)
__device__ __forceinline__ float s16(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 16);
  return v;
}

// shared memory carve-up (floats)
struct Smem {
  float e[T * C], xh1[T * C], n1[T * C], q[T * C], k[T * C], v[T * C], ctx[T * C], r1[T * C];
  float xh2[T * C], n2[T * C], xhf[T * C], nf[T * C];
  float rs1[16], rs2[16], rsf[16];
  float A[T * 16], DA[T * 16];
  float f[T * F], h[T * F], df[T * F];
  float P[T * V];
  float dnf[T * C], dr2[T * C], dn2[T * C], dr1[T * C], dctx[T * C], dq[T * C], dk[T * C], dv[T * C], dn1[T * C], de[T * C];
  float G[NPARAM];
  int sx[T], sy[T];
  float loss;
};

__global__ void __launch_bounds__(NTHREADS) fwd_bwd(const float* __restrict__ P, float* __restrict__ part,
                                                    float* __restrict__ lossPart, int B) {
  extern __shared__ float sm_raw[];
  Smem& S = *reinterpret_cast<Smem*>(sm_raw);
  const int tid = threadIdx.x;
  const int t = tid >> 4, c = tid & 15;
  const bool valid = t < T;
  const float scale = 0.25f;  // 1/sqrt(16)

  for (int i = tid; i < NPARAM; i += NTHREADS) S.G[i] = 0.f;
  if (tid == 0) S.loss = 0.f;
  __syncthreads();

  for (int s = blockIdx.x; s < B; s += gridDim.x) {
    const int ph = s & 3;
    if (tid < T) { S.sx[tid] = c_x[ph][tid]; S.sy[tid] = c_y[ph][tid]; }
    __syncthreads();

    // ---- embedding + LN1
    {
      float val = valid ? P[O_WTE + S.sx[t] * C + c] + P[O_WPE + t * C + c] : 0.f;
      float mean = s16(val) * (1.f / C);
      float d = val - mean;
      float var = s16(d * d) * (1.f / C);
      float rs = 1.f / sqrtf(var + 1e-5f);
      float xh = d * rs;
      if (valid) {
        S.e[t * C + c] = val;
        S.xh1[t * C + c] = xh;
        S.n1[t * C + c] = xh * P[O_LN1G + c] + P[O_LN1B + c];
        if (c == 0) S.rs1[t] = rs;
      }
    }
    __syncthreads();
    // ---- q, k, v
    if (valid) {
      float aq = P[O_BQ + c], ak = P[O_BK + c], av = P[O_BV + c];
#pragma unroll
      for (int i = 0; i < C; i++) {
        float x = S.n1[t * C + i];
        aq += P[O_WQ + c * C + i] * x;
        ak += P[O_WK + c * C + i] * x;
        av += P[O_WV + c * C + i] * x;
      }
      S.q[t * C + c] = aq; S.k[t * C + c] = ak; S.v[t * C + c] = av;
    }
    __syncthreads();
    // ---- attention probabilities (one thread per query row)
    if (tid < T) {
      float sc[T];
      float mx = -1e30f;
      for (int j = 0; j <= tid; j++) {
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < C; i++) a += S.q[tid * C + i] * S.k[j * C + i];
        sc[j] = a * scale;
        mx = fmaxf(mx, sc[j]);
      }
      float sum = 0.f;
      for (int j = 0; j <= tid; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
      float inv = 1.f / sum;
      for (int j = 0; j < 16; j++) S.A[tid * 16 + j] = (j <= tid) ? sc[j] * inv : 0.f;
    }
    __syncthreads();
    // ---- context
    if (valid) {
      float a = 0.f;
      for (int j = 0; j <= t; j++) a += S.A[t * 16 + j] * S.v[j * C + c];
      S.ctx[t * C + c] = a;
    }
    __syncthreads();
    // ---- o-proj, residual, LN2
    {
      float o = 0.f;
      if (valid) {
        o = P[O_BO + c];
#pragma unroll
        for (int i = 0; i < C; i++) o += P[O_WO + c * C + i] * S.ctx[t * C + i];
      }
      float r1 = valid ? S.e[t * C + c] + o : 0.f;
      float mean = s16(r1) * (1.f / C);
      float d = r1 - mean;
      float var = s16(d * d) * (1.f / C);
      float rs = 1.f / sqrtf(var + 1e-5f);
      float xh = d * rs;
      if (valid) {
        S.r1[t * C + c] = r1;
        S.xh2[t * C + c] = xh;
        S.n2[t * C + c] = xh * P[O_LN2G + c] + P[O_LN2B + c];
        if (c == 0) S.rs2[t] = rs;
      }
    }
    __syncthreads();
    // ---- MLP up
    for (int idx = tid; idx < T * F; idx += NTHREADS) {
      int tt = idx >> 6, m = idx & 63;
      float a = P[O_BFC + m];
#pragma unroll
      for (int i = 0; i < C; i++) a += P[O_WFC + m * C + i] * S.n2[tt * C + i];
      S.f[idx] = a;
      S.h[idx] = gelu_f(a);
    }
    __syncthreads();
    // ---- MLP down, residual, LN_f
    {
      float mo = 0.f;
      if (valid) {
        mo = P[O_BMP + c];
#pragma unroll 8
        for (int m = 0; m < F; m++) mo += P[O_WMP + c * F + m] * S.h[t * F + m];
      }
      float r2 = valid ? S.r1[t * C + c] + mo : 0.f;
      float mean = s16(r2) * (1.f / C);
      float d = r2 - mean;
      float var = s16(d * d) * (1.f / C);
      float rs = 1.f / sqrtf(var + 1e-5f);
      float xh = d * rs;
      if (valid) {
        S.xhf[t * C + c] = xh;
        S.nf[t * C + c] = xh * P[O_LNFG + c] + P[O_LNFB + c];
        if (c == 0) S.rsf[t] = rs;
      }
    }
    __syncthreads();
    // ---- logits
    for (int idx = tid; idx < T * V; idx += NTHREADS) {
      int tt = idx >> 5, v = idx & 31;
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < C; i++) a += P[O_WTE + v * C + i] * S.nf[tt * C + i];
      S.P[idx] = a;
    }
    __syncthreads();
    // ---- softmax + loss + d logits
    if (tid < T) {
      float mx = -1e30f;
      for (int v = 0; v < V; v++) mx = fmaxf(mx, S.P[tid * V + v]);
      float sum = 0.f;
      for (int v = 0; v < V; v++) { float e = expf(S.P[tid * V + v] - mx); S.P[tid * V + v] = e; sum += e; }
      float inv = 1.f / sum;
      int y = S.sy[tid];
      float py = S.P[tid * V + y] * inv;
      atomicAdd(&S.loss, -logf(fmaxf(py, 1e-30f)));
      for (int v = 0; v < V; v++) S.P[tid * V + v] = S.P[tid * V + v] * inv - (v == y ? 1.f : 0.f);
    }
    __syncthreads();

    // ================= backward =================
    // d nf, d wte (head), LN_f backward
    {
      float dnf = 0.f;
      if (valid) {
#pragma unroll
        for (int v = 0; v < V; v++) dnf += S.P[t * V + v] * P[O_WTE + v * C + c];
      }
      float dxh = dnf * P[O_LNFG + c];
      float xh = valid ? S.xhf[t * C + c] : 0.f;
      float m1 = s16(dxh) * (1.f / C);
      float m2 = s16(dxh * xh) * (1.f / C);
      if (valid) {
        S.dnf[t * C + c] = dnf;
        S.dr2[t * C + c] = S.rsf[t] * (dxh - m1 - xh * m2);
      }
    }
    for (int idx = tid; idx < V * C; idx += NTHREADS) {
      int v = idx >> 4, cc = idx & 15;
      float a = 0.f;
      for (int tt = 0; tt < T; tt++) a += S.P[tt * V + v] * S.nf[tt * C + cc];
      S.G[O_WTE + idx] += a;
    }
    __syncthreads();
    // MLP backward: dWmp, dh -> df ; LN_f params
    for (int idx = tid; idx < C * F; idx += NTHREADS) {
      int cc = idx >> 6, m = idx & 63;
      float a = 0.f;
      for (int tt = 0; tt < T; tt++) a += S.dr2[tt * C + cc] * S.h[tt * F + m];
      S.G[O_WMP + idx] += a;
    }
    for (int idx = tid; idx < T * F; idx += NTHREADS) {
      int tt = idx >> 6, m = idx & 63;
      float a = 0.f;
#pragma unroll
      for (int cc = 0; cc < C; cc++) a += P[O_WMP + cc * F + m] * S.dr2[tt * C + cc];
      S.df[idx] = a * gelu_grad(S.f[idx]);
    }
    if (tid < 16) {
      float g = 0.f, b = 0.f, bm = 0.f;
      for (int tt = 0; tt < T; tt++) {
        g += S.dnf[tt * C + tid] * S.xhf[tt * C + tid];
        b += S.dnf[tt * C + tid];
        bm += S.dr2[tt * C + tid];
      }
      S.G[O_LNFG + tid] += g; S.G[O_LNFB + tid] += b; S.G[O_BMP + tid] += bm;
    }
    __syncthreads();
    // dWfc, dbfc, dn2, LN2 backward
    for (int idx = tid; idx < F * C; idx += NTHREADS) {
      int m = idx >> 4, i = idx & 15;
      float a = 0.f;
      for (int tt = 0; tt < T; tt++) a += S.df[tt * F + m] * S.n2[tt * C + i];
      S.G[O_WFC + idx] += a;
    }
    if (tid < F) {
      float b = 0.f;
      for (int tt = 0; tt < T; tt++) b += S.df[tt * F + tid];
      S.G[O_BFC + tid] += b;
    }
    {
      float dn2 = 0.f;
      if (valid) {
#pragma unroll 8
        for (int m = 0; m < F; m++) dn2 += P[O_WFC + m * C + c] * S.df[t * F + m];
      }
      float dxh = dn2 * P[O_LN2G + c];
      float xh = valid ? S.xh2[t * C + c] : 0.f;
      float m1 = s16(dxh) * (1.f / C);
      float m2 = s16(dxh * xh) * (1.f / C);
      if (valid) {
        S.dn2[t * C + c] = dn2;
        S.dr1[t * C + c] = S.dr2[t * C + c] + S.rs2[t] * (dxh - m1 - xh * m2);
      }
    }
    __syncthreads();
    // dWo, dbo, dctx, LN2 params
    {
      int o = tid >> 4, i = tid & 15;
      float a = 0.f;
      for (int tt = 0; tt < T; tt++) a += S.dr1[tt * C + o] * S.ctx[tt * C + i];
      S.G[O_WO + tid] += a;
    }
    if (valid) {
      float a = 0.f;
#pragma unroll
      for (int o = 0; o < C; o++) a += P[O_WO + o * C + c] * S.dr1[t * C + o];
      S.dctx[t * C + c] = a;
    }
    if (tid < 16) {
      float g = 0.f, b = 0.f, bo = 0.f;
      for (int tt = 0; tt < T; tt++) {
        g += S.dn2[tt * C + tid] * S.xh2[tt * C + tid];
        b += S.dn2[tt * C + tid];
        bo += S.dr1[tt * C + tid];
      }
      S.G[O_LN2G + tid] += g; S.G[O_LN2B + tid] += b; S.G[O_BO + tid] += bo;
    }
    __syncthreads();
    // attention backward
    for (int idx = tid; idx < T * 16; idx += NTHREADS) {
      int tt = idx >> 4, j = idx & 15;
      float a = 0.f;
      if (j <= tt) {
#pragma unroll
        for (int i = 0; i < C; i++) a += S.dctx[tt * C + i] * S.v[j * C + i];
      }
      S.DA[idx] = a;
    }
    __syncthreads();
    if (tid < T) {
      float dot = 0.f;
      for (int j = 0; j <= tid; j++) dot += S.A[tid * 16 + j] * S.DA[tid * 16 + j];
      for (int j = 0; j < 16; j++) S.DA[tid * 16 + j] = S.A[tid * 16 + j] * (S.DA[tid * 16 + j] - dot);
    }
    __syncthreads();
    if (valid) {
      float aq = 0.f, ak = 0.f, av = 0.f;
      for (int j = 0; j <= t; j++) aq += S.DA[t * 16 + j] * S.k[j * C + c];
      for (int tt = t; tt < T; tt++) {
        ak += S.DA[tt * 16 + t] * S.q[tt * C + c];
        av += S.A[tt * 16 + t] * S.dctx[tt * C + c];
      }
      S.dq[t * C + c] = aq * scale; S.dk[t * C + c] = ak * scale; S.dv[t * C + c] = av;
    }
    __syncthreads();
    // dWq/dWk/dWv, dbq.., dn1, LN1 backward
    {
      int o = tid >> 4, i = tid & 15;
      float aq = 0.f, ak = 0.f, av = 0.f;
      for (int tt = 0; tt < T; tt++) {
        float x = S.n1[tt * C + i];
        aq += S.dq[tt * C + o] * x; ak += S.dk[tt * C + o] * x; av += S.dv[tt * C + o] * x;
      }
      S.G[O_WQ + tid] += aq; S.G[O_WK + tid] += ak; S.G[O_WV + tid] += av;
    }
    if (tid < 16) {
      float bq = 0.f, bk = 0.f, bv = 0.f;
      for (int tt = 0; tt < T; tt++) { bq += S.dq[tt * C + tid]; bk += S.dk[tt * C + tid]; bv += S.dv[tt * C + tid]; }
      S.G[O_BQ + tid] += bq; S.G[O_BK + tid] += bk; S.G[O_BV + tid] += bv;
    }
    {
      float dn1 = 0.f;
      if (valid) {
#pragma unroll
        for (int o = 0; o < C; o++)
          dn1 += P[O_WQ + o * C + c] * S.dq[t * C + o] + P[O_WK + o * C + c] * S.dk[t * C + o] + P[O_WV + o * C + c] * S.dv[t * C + o];
      }
      float dxh = dn1 * P[O_LN1G + c];
      float xh = valid ? S.xh1[t * C + c] : 0.f;
      float m1 = s16(dxh) * (1.f / C);
      float m2 = s16(dxh * xh) * (1.f / C);
      if (valid) {
        S.dn1[t * C + c] = dn1;
        S.de[t * C + c] = S.dr1[t * C + c] + S.rs1[t] * (dxh - m1 - xh * m2);
      }
    }
    __syncthreads();
    // embeddings and LN1 params
    if (valid) {
      S.G[O_WPE + t * C + c] += S.de[t * C + c];
      atomicAdd(&S.G[O_WTE + S.sx[t] * C + c], S.de[t * C + c]);
    }
    if (tid < 16) {
      float g = 0.f, b = 0.f;
      for (int tt = 0; tt < T; tt++) { g += S.dn1[tt * C + tid] * S.xh1[tt * C + tid]; b += S.dn1[tt * C + tid]; }
      S.G[O_LN1G + tid] += g; S.G[O_LN1B + tid] += b;
    }
    __syncthreads();
  }
  for (int i = tid; i < NPARAM; i += NTHREADS) part[(size_t)blockIdx.x * NPARAM + i] = S.G[i];
  if (tid == 0) lossPart[blockIdx.x] = S.loss;
}

__global__ void adam_kernel(float* __restrict__ p, float* __restrict__ m, float* __restrict__ v,
                            const float* __restrict__ part, const float* __restrict__ lossPart, float* __restrict__ lossOut,
                            int nblocks, float inv_bt, float lr, float b1, float b2, float eps, float c1, float c2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NPARAM) {
    float g = 0.f;
    for (int b = 0; b < nblocks; b++) g += part[(size_t)b * NPARAM + i];
    g *= inv_bt;
    float mm = b1 * m[i] + (1.f - b1) * g;
    float vv = b2 * v[i] + (1.f - b2) * g * g;
    m[i] = mm; v[i] = vv;
    p[i] -= lr * (mm / c1) / (sqrtf(vv / c2) + eps);
  }
  if (i == 0) {
    float l = 0.f;
    for (int b = 0; b < nblocks; b++) l += lossPart[b];
    *lossOut = l * inv_bt;
  }
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s params.bin B steps [verify]\n", argv[0]); return 1; }
  const int B = atoi(argv[2]), steps = atoi(argv[3]);
  const bool verify = argc > 4;
  std::vector<float> h(NPARAM);
  FILE* f = fopen(argv[1], "rb");
  if (!f || fread(h.data(), 4, NPARAM, f) != NPARAM) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  fclose(f);
  // data: the four phrases of the Bend / PyTorch runs, tokens = " ABCDEFGHIJKLMNOPQRSTUVWXYZ!.:01"
  const char* chars = " ABCDEFGHIJKLMNOPQRSTUVWXYZ!.:01";
  const char* phrases[4] = {"BEND IS FAST!  ", "BEND ON GPU!   ", "BEND RUNS AI!  ", "BEND PARALLEL! "};
  int hx[4][T], hy[4][T];
  for (int p = 0; p < 4; p++) {
    int ids[T + 1];
    for (int i = 0; i < T + 1; i++) { const char* q = strchr(chars, phrases[p][i]); ids[i] = (int)(q - chars); }
    for (int i = 0; i < T; i++) { hx[p][i] = ids[i]; hy[p][i] = ids[i + 1]; }
  }
  CHECK(cudaMemcpyToSymbol(c_x, hx, sizeof hx));
  CHECK(cudaMemcpyToSymbol(c_y, hy, sizeof hy));

  int dev = 0, nsm = 0;
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  const size_t smem = sizeof(Smem);
  CHECK(cudaFuncSetAttribute(fwd_bwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int bps = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, fwd_bwd, NTHREADS, smem));
  const int grid = std::min(B, nsm * (bps > 0 ? bps : 1));

  float *dp, *dm, *dv, *dpart, *dlp, *dloss;
  CHECK(cudaMalloc(&dp, NPARAM * 4)); CHECK(cudaMalloc(&dm, NPARAM * 4)); CHECK(cudaMalloc(&dv, NPARAM * 4));
  CHECK(cudaMalloc(&dpart, (size_t)grid * NPARAM * 4)); CHECK(cudaMalloc(&dlp, grid * 4)); CHECK(cudaMalloc(&dloss, 4));
  CHECK(cudaMemcpy(dp, h.data(), NPARAM * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dm, 0, NPARAM * 4)); CHECK(cudaMemset(dv, 0, NPARAM * 4));

  const float lr = 0.01f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, inv_bt = 1.f / ((float)B * T);
  auto step = [&](int s) {
    fwd_bwd<<<grid, NTHREADS, smem>>>(dp, dpart, dlp, B);
    float c1 = 1.f - powf(b1, (float)(s + 1)), c2 = 1.f - powf(b2, (float)(s + 1));
    adam_kernel<<<(NPARAM + 255) / 256, 256>>>(dp, dm, dv, dpart, dlp, dloss, grid, inv_bt, lr, b1, b2, eps, c1, c2);
  };
  if (verify) {
    for (int s = 0; s < steps; s++) {
      step(s);
      float l; CHECK(cudaMemcpy(&l, dloss, 4, cudaMemcpyDeviceToHost));
      printf("step %d loss %.7f\n", s, l);
    }
    return 0;
  }
  const int warm = 5;
  for (int s = 0; s < warm; s++) step(s);
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
  CHECK(cudaEventRecord(e0));
  for (int s = 0; s < steps; s++) step(warm + s);
  CHECK(cudaEventRecord(e1));
  CHECK(cudaEventSynchronize(e1));
  float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
  CHECK(cudaGetLastError());
  printf("ms_per_step %.4f grid %d smem %zu\n", ms / steps, grid, smem);
  return 0;
}
