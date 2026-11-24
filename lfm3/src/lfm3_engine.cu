// lfm3_engine : moteur sur mesure de LFM3 1.7B (LiquidAI), GGUF QAD Q4_0, une seule source CUDA et HIP.
//
//   lfm3-engine --model F.gguf (--serve | --tokens IDS | --tokens-file F) [--n N] [--ctx N] [--dump-logits F]
//
// Le décodage est un graphe de ~120 noyaux par token (llama.cpp : 381) :
// - les poids Q4_0 sont réorganisés au chargement (échelles à part, 16 octets de quartets alignés) ; un warp lit
//   ses blocs en 128 bits, l'activation (quantifiée en q8 comme chez llama.cpp) reste dans ses registres ;
// - chaque produit matrice-vecteur commence par la norme dont il a besoin (prologue) : la norme du bloc
//   précédent, le résidu, puis la norme d'entrée, recalculés par chaque bloc de la grille (2 x 8 Ko lus) ;
// - et finit par ce qui suit (épilogue) : SwiGLU (lignes gate/up entrelacées), ou la convolution courte entière
//   (lignes b/c/x entrelacées par canal : b·x, fenêtre de 7, c·conv, état mis à jour) ;
// - l'attention : un noyau par (morceau de positions, tête KV), les 4 têtes de requête d'une tête KV ensemble,
//   norme q/k et RoPE faites sur place, puis un noyau qui combine les morceaux et applique la porte sigmoïde.
// Le protocole --serve est celui de strata-ink2 (READY, GEN, T, PP, DONE, STOP, QUIT) : même façade.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemset hipMemset
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaGraphInstantiateWithFlags hipGraphInstantiateWithFlags
#define cudaGraphLaunch hipGraphLaunch
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemGetInfo hipMemGetInfo
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define SHFL_XOR(v, m) __shfl_xor(v, m, 32)
#define SHFL(v, l) __shfl(v, l, 32)
#define WARP_SYNC() do { __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront"); __builtin_amdgcn_wave_barrier(); } while (0)
#include <hipblas/hipblas.h>
#include <rocwmma/rocwmma.hpp>
namespace wm = rocwmma;
using wf16 = rocwmma::hfloat16_t;   // = __half
#define blasHandle_t hipblasHandle_t
#define blasCreate hipblasCreate
#define blasSetStream hipblasSetStream
static void gemm_f16(blasHandle_t b, int N, int T, int K, const __half* W, const __half* X, float* Y) {
    const float one = 1.f, zero = 0.f;
    hipblasGemmEx(b, HIPBLAS_OP_T, HIPBLAS_OP_N, N, T, K, &one, W, HIP_R_16F, K, X, HIP_R_16F, K, &zero, Y, HIP_R_32F, N,
                  HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
}
__device__ __forceinline__ int dp4a(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define SHFL_XOR(v, m) __shfl_xor_sync(0xffffffffu, v, m, 32)
#define SHFL(v, l) __shfl_sync(0xffffffffu, v, l, 32)
#define WARP_SYNC() __syncwarp()
#include <cublas_v2.h>
#include <mma.h>
namespace wm = nvcuda::wmma;
using wf16 = __half;
#define blasHandle_t cublasHandle_t
#define blasCreate cublasCreate
#define blasSetStream cublasSetStream
static void gemm_f16(blasHandle_t b, int N, int T, int K, const __half* W, const __half* X, float* Y) {
    const float one = 1.f, zero = 0.f;
    cublasGemmEx(b, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16F, K, X, CUDA_R_16F, K, &zero, Y, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
__device__ __forceinline__ int dp4a(int a, int b, int c) { return __dp4a(a, b, c); }
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "GPU %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

// ── Architecture (vérifiée contre le GGUF au chargement) ─────────────────────────────────────
constexpr int H = 2048, NL = 24, NH = 16, NKV = 4, HD = 128, FF = 7680, VOCAB = 128000, CONVK = 7, SWA = 256;
constexpr int QKVG = NH * HD + 2 * NKV * HD + NH;   // 3088 lignes : q, k, v, porte
constexpr float EPS = 1e-5f;
constexpr int S_SPLIT = 128;                        // morceaux de positions par tête KV (décodage), au plus
constexpr int THREADS = 256, WARPS = THREADS / 32;
#ifndef ATT_SWITCH
#define ATT_SWITCH 1024
#endif
#ifndef GRID_MULT
#define GRID_MULT 4
#endif
#ifndef PREFETCH
#define PREFETCH 0
#endif
#ifndef R_GU
#define R_GU 4
#endif
#ifndef R_DOWN
#define R_DOWN 2
#endif
#ifndef R_OUT
#define R_OUT 2
#endif
#ifndef R_CONV
#define R_CONV 3
#endif

// ── GGUF ─────────────────────────────────────────────────────────────────────────────────────
struct GTensor { std::string name; std::vector<uint64_t> ne; uint32_t type; uint64_t off; const uint8_t* data; };
struct Gguf {
    const uint8_t* base = nullptr; size_t size = 0;
    std::unordered_map<std::string, GTensor> t;
    std::unordered_map<std::string, std::vector<int64_t>> ints;   // clés entières (et tableaux d'entiers / booléens)
    std::unordered_map<std::string, double> reals;
};
static Gguf load_gguf(const char* path) {
    Gguf g;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { std::perror(path); std::exit(1); }
    struct stat st; fstat(fd, &st); g.size = st.st_size;
    g.base = (const uint8_t*) mmap(nullptr, g.size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    const uint8_t* p = g.base;
    auto rd = [&](auto& v) { std::memcpy(&v, p, sizeof(v)); p += sizeof(v); };
    auto str = [&]() { uint64_t n; rd(n); std::string s((const char*) p, n); p += n; return s; };
    uint32_t magic, ver; rd(magic); rd(ver);
    if (magic != 0x46554747u) { std::fprintf(stderr, "pas un GGUF\n"); std::exit(1); }
    uint64_t nt, nkv; rd(nt); rd(nkv);
    const size_t tsz[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};   // u8 i8 u16 i16 u32 i32 f32 bool str arr u64 i64 f64
    uint64_t align = 32;
    std::function<void(uint32_t, std::vector<int64_t>*, double*)> skip = [&](uint32_t ty, std::vector<int64_t>* iv, double* fv) {
        if (ty == 8) { str(); return; }
        if (ty == 9) { uint32_t et; uint64_t n; rd(et); rd(n); for (uint64_t i = 0; i < n; ++i) skip(et, iv, nullptr); return; }
        int64_t v = 0; double f = 0;
        switch (ty) {
            case 0: case 7: v = *p; break; case 1: v = (int8_t) *p; break;
            case 2: { uint16_t x; std::memcpy(&x, p, 2); v = x; } break; case 3: { int16_t x; std::memcpy(&x, p, 2); v = x; } break;
            case 4: { uint32_t x; std::memcpy(&x, p, 4); v = x; } break; case 5: { int32_t x; std::memcpy(&x, p, 4); v = x; } break;
            case 6: { float x; std::memcpy(&x, p, 4); f = x; v = (int64_t) x; } break;
            case 10: { uint64_t x; std::memcpy(&x, p, 8); v = (int64_t) x; } break; case 11: { int64_t x; std::memcpy(&x, p, 8); v = x; } break;
            case 12: { double x; std::memcpy(&x, p, 8); f = x; v = (int64_t) x; } break;
        }
        if (iv) iv->push_back(v);
        if (fv) *fv = (ty == 6 || ty == 12) ? f : (double) v;
        p += tsz[ty];
    };
    for (uint64_t i = 0; i < nkv; ++i) {
        std::string k = str(); uint32_t ty; rd(ty);
        std::vector<int64_t> iv; double fv = 0;
        skip(ty, &iv, &fv);
        if (ty != 8) { g.ints[k] = iv; g.reals[k] = fv; }
        if (k == "general.alignment") align = (uint64_t) iv[0];
    }
    std::vector<GTensor> ts;
    for (uint64_t i = 0; i < nt; ++i) {
        GTensor x; x.name = str(); uint32_t nd; rd(nd); x.ne.resize(nd);
        for (auto& d : x.ne) rd(d);
        rd(x.type); rd(x.off); ts.push_back(x);
    }
    size_t data = ((size_t) (p - g.base) + align - 1) / align * align;
    for (auto& x : ts) { x.data = g.base + data + x.off; g.t[x.name] = x; }
    return g;
}
static const GTensor& need(const Gguf& g, const std::string& n) {
    auto it = g.t.find(n);
    if (it == g.t.end()) { std::fprintf(stderr, "tenseur absent : %s\n", n.c_str()); std::exit(1); }
    return it->second;
}
static float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t f;
    if (e == 0) { if (m == 0) f = s << 31; else { int ee = -1; uint32_t mm = m; do { ++ee; mm <<= 1; } while (!(mm & 1024)); f = (s << 31) | ((uint32_t) (127 - 15 - ee) << 23) | ((mm & 1023) << 13); } }
    else if (e == 31) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float r; std::memcpy(&r, &f, 4); return r;
}

// ── Poids réorganisés ───────────────────────────────────────────────────────────────────────
// Q4W : N lignes de K valeurs ; qs[N][K/32][16 octets] (le format Q4_0 : octet b = valeur b en bas, b+16 en haut),
// d[N][K/32] en fp16. Q6W (sortie liée) : par groupe de 32, lo[16 octets] comme Q4_0, hi[8 octets] (mot k, octet bb,
// paire de bits s : valeur k·16 + s·4 + bb), sc[2] (échelles int8 des deux moitiés), et d par super-bloc de 256.
struct Q4W { const uint4* qs; const __half* d; int N, K; };
struct Q6W { const uint4* lo; const uint2* hi; const int8_t* sc; const float* d; int N; };

static std::vector<uint8_t> g_q4_qs, g_q4_d;   // tampons hôte de la réorganisation (réutilisés)
template <typename T> static T* upload(const void* src, size_t bytes) {
    T* p; CK(cudaMalloc((void**) &p, bytes)); CK(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice)); return p;
}
// lignes (en ordre) prises à travers plusieurs tenseurs Q4_0 : rows[i] = {tenseur, ligne}
static Q4W pack_q4(const std::vector<std::pair<const GTensor*, int>>& rows, int K) {
    const int NB = K / 32, N = (int) rows.size();
    std::vector<uint8_t> qs((size_t) N * NB * 16);
    std::vector<uint16_t> d((size_t) N * NB);
    for (int r = 0; r < N; ++r) {
        const GTensor* t = rows[r].first;
        if (t->type != 2 || (int) t->ne[0] != K) { std::fprintf(stderr, "%s : Q4_0 de %d colonnes attendu\n", t->name.c_str(), K); std::exit(1); }
        const uint8_t* src = t->data + (size_t) rows[r].second * NB * 18;
        for (int b = 0; b < NB; ++b) {
            std::memcpy(&d[(size_t) r * NB + b], src + b * 18, 2);
            std::memcpy(&qs[((size_t) r * NB + b) * 16], src + b * 18 + 2, 16);
        }
    }
    return {upload<uint4>(qs.data(), qs.size()), upload<__half>(d.data(), d.size() * 2), N, K};
}
static Q4W pack_q4_whole(const GTensor& t) {
    std::vector<std::pair<const GTensor*, int>> rows;
    for (int r = 0; r < (int) t.ne[1]; ++r) rows.push_back({&t, r});
    return pack_q4(rows, (int) t.ne[0]);
}
static Q6W pack_q6(const GTensor& t) {
    if (t.type != 14 || t.ne[0] != H) { std::fprintf(stderr, "%s : Q6_K de %d colonnes attendu\n", t.name.c_str(), H); std::exit(1); }
    const int N = (int) t.ne[1], NSB = H / 256, NG = H / 32;
    std::vector<uint8_t> lo((size_t) N * NG * 16), hi((size_t) N * NG * 8);
    std::vector<int8_t> sc((size_t) N * NG * 2);
    std::vector<float> d((size_t) N * NSB);
    for (int r = 0; r < N; ++r) {
        for (int sb = 0; sb < NSB; ++sb) {
            const uint8_t* B = t.data + ((size_t) r * NSB + sb) * 210;   // ql[128] qh[64] scales[16] d
            const uint8_t *ql = B, *qh = B + 128; const int8_t* scs = (const int8_t*) (B + 192);
            uint16_t dh; std::memcpy(&dh, B + 208, 2);
            d[(size_t) r * NSB + sb] = h2f(dh);
            uint8_t q[256]; int8_t s16[16];
            for (int e = 0; e < 256; ++e) {
                const int n = e / 128, rr = e % 128, qd = rr / 32, l = rr % 32;
                const uint8_t lb = ql[n * 64 + l + (qd & 1) * 32];
                const int lo4 = (qd < 2) ? (lb & 15) : (lb >> 4);
                const int hi2 = (qh[n * 32 + l] >> (2 * qd)) & 3;
                q[e] = (uint8_t) (lo4 | (hi2 << 4));
                if (l % 16 == 0) s16[e / 16] = scs[n * 8 + qd * 2 + l / 16];
            }
            for (int gi = 0; gi < 8; ++gi) {
                const int G = sb * 8 + gi;
                uint8_t* L = &lo[((size_t) r * NG + G) * 16];
                uint8_t* Hh = &hi[((size_t) r * NG + G) * 8];
                std::memset(Hh, 0, 8);
                for (int b = 0; b < 16; ++b) L[b] = (q[gi * 32 + b] & 15) | ((q[gi * 32 + b + 16] & 15) << 4);
                for (int k = 0; k < 2; ++k) for (int s = 0; s < 4; ++s) for (int bb = 0; bb < 4; ++bb) {
                    const int e = k * 16 + s * 4 + bb;
                    Hh[k * 4 + bb] |= (uint8_t) (((q[gi * 32 + e] >> 4) & 3) << (2 * s));
                }
                sc[((size_t) r * NG + G) * 2 + 0] = s16[gi * 2];
                sc[((size_t) r * NG + G) * 2 + 1] = s16[gi * 2 + 1];
            }
        }
    }
    return {upload<uint4>(lo.data(), lo.size()), upload<uint2>(hi.data(), hi.size()), upload<int8_t>(sc.data(), sc.size()),
            upload<float>(d.data(), d.size() * 4), N};
}
static float* upload_f32(const GTensor& t) {
    if (t.type != 0) { std::fprintf(stderr, "%s : F32 attendu\n", t.name.c_str()); std::exit(1); }
    size_t n = 1; for (auto x : t.ne) n *= x;
    return upload<float>(t.data, n * 4);
}

// ── Noyaux ──────────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_sum(float v) { for (int m = 16; m; m >>= 1) v += SHFL_XOR(v, m); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int m = 16; m; m >>= 1) v = fmaxf(v, SHFL_XOR(v, m)); return v; }
__device__ float block_sum(float v, float* red) {   // THREADS fils ; red[WARPS + 1]
    v = warp_sum(v);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    if (w == 0) { float x = l < WARPS ? red[l] : 0.f; x = warp_sum(x); if (l == 0) red[WARPS] = x; }
    __syncthreads();
    return red[WARPS];
}

enum { PRO_NORM = 0, PRO_RESNORM = 1, PRO_RAW = 2 };
enum { EPI_STORE = 0, EPI_SWIGLU = 1, EPI_CONV = 2, EPI_LOGITS = 3 };
struct Pro {
    const float* h;        // NORM, RESNORM : le résidu entrant
    const float* d;        // RESNORM : la sortie brute du bloc précédent ; RAW : l'entrée
    const float* wpost;    // RESNORM : la norme d'après bloc
    const float* wpre;     // NORM, RESNORM : la norme d'entrée
    float* hout;           // RESNORM : le résidu mis à jour (écrit par le bloc 0)
};
struct Epi {
    float* y;                       // STORE, SWIGLU, CONV : la sortie ; LOGITS : les logits
    float* conv_state;              // CONV : [6][H]
    const float* conv_w;            // CONV : [H][7]
    unsigned long long* argmax;     // LOGITS : (logit ordonné << 32) | ~indice
};

// Le prologue : x (fp32, K valeurs) puis sa version q8 par blocs de 32 (xq, dx, sommes), dans les registres :
// le fil t tient les éléments t + 256k, donc chaque warp tient exactement le bloc de 32 valeurs w + 8k.
template <int K, int PRO>
__device__ void prologue(const Pro& P, int8_t* xq, float* dx, float* s8, float* s16, float* red, float* xs) {
    static_assert(K % THREADS == 0, "K multiple de 256");
    constexpr int PER = K / THREADS;
    const int t = threadIdx.x, lane = t % 32;
    float v[PER];
    if (PRO == PRO_RAW) {
#pragma unroll
        for (int k = 0; k < PER; ++k) v[k] = P.d[t + THREADS * k];
    } else if (PRO == PRO_NORM) {
        float ss = 0;
#pragma unroll
        for (int k = 0; k < PER; ++k) { v[k] = P.h[t + THREADS * k]; ss += v[k] * v[k]; }
        const float r = rsqrtf(block_sum(ss, red) / K + EPS);
#pragma unroll
        for (int k = 0; k < PER; ++k) v[k] *= r * P.wpre[t + THREADS * k];
    } else {
        float ss = 0;
#pragma unroll
        for (int k = 0; k < PER; ++k) { v[k] = P.d[t + THREADS * k]; ss += v[k] * v[k]; }
        const float rd = rsqrtf(block_sum(ss, red) / K + EPS);
        float ss2 = 0;
#pragma unroll
        for (int k = 0; k < PER; ++k) { const int i = t + THREADS * k; v[k] = P.h[i] + v[k] * rd * P.wpost[i]; ss2 += v[k] * v[k]; }
        if (blockIdx.x == 0)
#pragma unroll
            for (int k = 0; k < PER; ++k) P.hout[t + THREADS * k] = v[k];
        const float r = rsqrtf(block_sum(ss2, red) / K + EPS);
#pragma unroll
        for (int k = 0; k < PER; ++k) v[k] *= r * P.wpre[t + THREADS * k];
    }
    if constexpr (PER > 8) {   // K = 7680 : 30 blocs par warp, les échanges coûteraient plus que la mémoire partagée
#pragma unroll
        for (int k = 0; k < PER; ++k) xs[t + THREADS * k] = v[k];
        __syncthreads();
        for (int b = t; b < K / 32; b += THREADS) {
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xs[b * 32 + j]));
            const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
            int s0 = 0, s1 = 0;
            for (int j = 0; j < 32; ++j) { const int q = (int) rintf(xs[b * 32 + j] * id); xq[b * 32 + j] = (int8_t) q; (j < 16 ? s0 : s1) += q; }
            dx[b] = d; s8[b] = 8.f * d * (float) (s0 + s1); s16[2 * b] = 32.f * d * (float) s0; s16[2 * b + 1] = 32.f * d * (float) s1;
        }
        __syncthreads();
        return;
    }
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = t + THREADS * k, b = i / 32;
        const float amax = warp_max(fabsf(v[k]));
        const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
        const int q = (int) rintf(v[k] * id);
        xq[i] = (int8_t) q;
        const float s0 = warp_sum(lane < 16 ? (float) q : 0.f), s1 = warp_sum(lane < 16 ? 0.f : (float) q);
        if (lane == 0) { dx[b] = d; s8[b] = 8.f * d * (s0 + s1); s16[2 * b] = 32.f * d * s0; s16[2 * b + 1] = 32.f * d * s1; }
    }
    __syncthreads();
}

__device__ __forceinline__ int q4_dot(const uint4 q, const int* xr) {
    const int m = 0x0F0F0F0F;
    int s = dp4a((int) (q.x & m), xr[0], 0);
    s = dp4a((int) (q.y & m), xr[1], s);
    s = dp4a((int) (q.z & m), xr[2], s);
    s = dp4a((int) (q.w & m), xr[3], s);
    s = dp4a((int) ((q.x >> 4) & m), xr[4], s);
    s = dp4a((int) ((q.y >> 4) & m), xr[5], s);
    s = dp4a((int) ((q.z >> 4) & m), xr[6], s);
    s = dp4a((int) ((q.w >> 4) & m), xr[7], s);
    return s;
}

template <int K, int R, int PRO, int EPI>
__global__ void __launch_bounds__(THREADS) k_gemv_q4(Q4W W, Pro P, Epi E) {
    constexpr int NB = K / 32, NBL = (NB + 31) / 32;
    __shared__ __align__(16) int8_t xq[K];
    __shared__ float dxs[NB], s8s[NB], s16s[2 * NB], red[WARPS + 1];
    __shared__ float xs[K > 2048 ? K : 1];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int nwarps = gridDim.x * WARPS;
    int r0 = (blockIdx.x * WARPS + warp) * R;
    // Les poids ne dépendent pas de x : la première itération se lit pendant le prologue.
    uint4 q[R][NBL]; float dw[R][NBL];
    auto load = [&](int rb) {
#pragma unroll
        for (int rr = 0; rr < R; ++rr)
#pragma unroll
            for (int i = 0; i < NBL; ++i) {
                const int j = lane + 32 * i;
                if (j < NB) { q[rr][i] = W.qs[(size_t) (rb + rr) * NB + j]; dw[rr][i] = __half2float(W.d[(size_t) (rb + rr) * NB + j]); }
                else { q[rr][i] = make_uint4(0, 0, 0, 0); dw[rr][i] = 0; }
            }
    };
#if PREFETCH
    if (r0 < W.N) load(r0);
#endif
    prologue<K, PRO>(P, xq, dxs, s8s, s16s, red, xs);
    int xr[NBL][8]; float dxr[NBL], s8r[NBL];
#pragma unroll
    for (int i = 0; i < NBL; ++i) {
        const int j = lane + 32 * i;
        if (j < NB) {
#pragma unroll
            for (int k = 0; k < 8; ++k) xr[i][k] = ((const int*) xq)[j * 8 + k];
            dxr[i] = dxs[j]; s8r[i] = s8s[j];
        } else { dxr[i] = 0; s8r[i] = 0; for (int k = 0; k < 8; ++k) xr[i][k] = 0; }
    }
    for (; r0 < W.N; r0 += nwarps * R) {
#if !PREFETCH
        load(r0);
#endif
        float acc[R];
        float a[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            a[rr] = 0;
#pragma unroll
            for (int i = 0; i < NBL; ++i) a[rr] += dw[rr][i] * (dxr[i] * (float) q4_dot(q[rr][i], xr[i]) - s8r[i]);
        }
#if PREFETCH
        if (r0 + nwarps * R < W.N) load(r0 + nwarps * R);   // la suite part pendant les réductions
#endif
#pragma unroll
        for (int rr = 0; rr < R; ++rr) acc[rr] = warp_sum(a[rr]);
        if (lane == 0) {
            if (EPI == EPI_STORE) {
#pragma unroll
                for (int rr = 0; rr < R; ++rr) E.y[r0 + rr] = acc[rr];
            } else if (EPI == EPI_SWIGLU) {
#pragma unroll
                for (int pp = 0; pp < R / 2; ++pp) {
                    const float g = acc[2 * pp], u = acc[2 * pp + 1];
                    E.y[r0 / 2 + pp] = g / (1.f + __expf(-g)) * u;
                }
            } else if (EPI == EPI_CONV) {
#pragma unroll
              for (int pp = 0; pp < R / 3; ++pp) {
                const int ch = r0 / 3 + pp;
                const float b = acc[3 * pp], c = acc[3 * pp + 1], xv = acc[3 * pp + 2], bx = b * xv;
                const float* w = E.conv_w + ch * CONVK;
                float s[CONVK - 1];
#pragma unroll
                for (int k = 0; k < CONVK - 1; ++k) s[k] = E.conv_state[k * H + ch];
                float o = w[CONVK - 1] * bx;
#pragma unroll
                for (int k = 0; k < CONVK - 1; ++k) o += w[k] * s[k];
#pragma unroll
                for (int k = 0; k < CONVK - 2; ++k) E.conv_state[k * H + ch] = s[k + 1];
                E.conv_state[(CONVK - 2) * H + ch] = bx;
                E.y[ch] = c * o;
              }
            }
        }
    }
}

// La sortie liée (Q6_K réorganisé), avec l'arg-max du bloc par atomique 64 bits.
__device__ __forceinline__ unsigned long long pack_max(float v, int idx) {
    uint32_t u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // ordre des flottants = ordre des entiers
    return ((unsigned long long) u << 32) | (uint32_t) (~(uint32_t) idx);
}
template <int R>
__global__ void __launch_bounds__(THREADS) k_gemv_q6_logits(Q6W W, Pro P, Epi E) {
    constexpr int K = H, NB = K / 32, NBL = NB / 32;   // 2 groupes par fil
    __shared__ __align__(16) int8_t xq[K];
    __shared__ float dxs[NB], s8s[NB], s16s[2 * NB], red[WARPS + 1];
    prologue<K, PRO_RESNORM>(P, xq, dxs, s8s, s16s, red, nullptr);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int xr[NBL][8]; float dxr[NBL], c0[NBL], c1[NBL];
#pragma unroll
    for (int i = 0; i < NBL; ++i) {
        const int j = lane + 32 * i;
#pragma unroll
        for (int k = 0; k < 8; ++k) xr[i][k] = ((const int*) xq)[j * 8 + k];
        dxr[i] = dxs[j]; c0[i] = s16s[2 * j]; c1[i] = s16s[2 * j + 1];
    }
    float best = -INFINITY; int besti = 0;
    const int nwarps = gridDim.x * WARPS;
    const int m = 0x0F0F0F0F, m2 = 0x03030303;
    for (int r0 = (blockIdx.x * WARPS + warp) * R; r0 < W.N; r0 += nwarps * R) {
        float acc[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            const int row = r0 + rr;
            float a = 0;
#pragma unroll
            for (int i = 0; i < NBL; ++i) {
                const int j = lane + 32 * i;
                const size_t g = (size_t) row * NB + j;
                const uint4 lo = W.lo[g]; const uint2 hi = W.hi[g];
                const int sc0 = W.sc[2 * g], sc1 = W.sc[2 * g + 1];
                const float d = W.d[(size_t) row * (K / 256) + j / 8];
                const int l[4] = {(int) lo.x, (int) lo.y, (int) lo.z, (int) lo.w};
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int qd = 0; qd < 4; ++qd) {   // valeurs 4qd..4qd+3 (moitié 0) et 16+4qd.. (moitié 1)
                    const int v0 = (l[qd] & m) | ((((int) hi.x >> (2 * qd)) & m2) << 4);
                    const int v1 = ((l[qd] >> 4) & m) | ((((int) hi.y >> (2 * qd)) & m2) << 4);
                    s0 = dp4a(v0, xr[i][qd], s0);
                    s1 = dp4a(v1, xr[i][4 + qd], s1);
                }
                a += d * ((float) sc0 * (dxr[i] * (float) s0 - c0[i]) + (float) sc1 * (dxr[i] * (float) s1 - c1[i]));
            }
            acc[rr] = warp_sum(a);
        }
        if (lane == 0)
#pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                E.y[r0 + rr] = acc[rr];
                if (acc[rr] > best) { best = acc[rr]; besti = r0 + rr; }
            }
    }
    if (lane == 0 && best > -INFINITY) atomicMax(E.argmax, pack_max(best, besti));
}

// La ligne d'embedding (même table Q6, réorganisée) du token d_tok.
__global__ void k_embed(Q6W W, const int* tok, float* h, unsigned long long* argmax) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;   // H fils
    const int row = *tok;
    const int g = e / 32, w = e % 32;
    const size_t gi = (size_t) row * (H / 32) + g;
    const uint8_t* lo = (const uint8_t*) &W.lo[gi];
    const uint8_t* hi = (const uint8_t*) &W.hi[gi];
    const int lo4 = w < 16 ? (lo[w] & 15) : (lo[w - 16] >> 4);
    const int k = w / 16, s = (w % 16) / 4, bb = w % 4;
    const int hi2 = (hi[k * 4 + bb] >> (2 * s)) & 3;
    const int q = (lo4 | (hi2 << 4)) - 32;
    h[e] = W.d[(size_t) row * (H / 256) + g / 8] * (float) W.sc[2 * gi + w / 16] * (float) q;
    if (e == 0) *argmax = 0ull;
}

// Attention, décodage. Bloc (morceau s, tête KV g), 4 warps : la tête de requête 4g + warp.
// Le fil l tient les dimensions l, l+32, l+64, l+96 : la RoPE NEOX tourne (l, l+64) et (l+32, l+96) sur place.
struct AttnArgs {
    const float* qkvg;          // sortie du produit q|k|v|porte
    const float *qn, *kn;       // normes q, k [128]
    __half *kc, *vc;            // cache [cap][NKV][HD]
    int cap;                    // positions du cache (SWA : 256, anneau)
    int swa;                    // 1 : fenêtre glissante
    float base;                 // base de la RoPE
    int rot_pairs;              // paires qui tournent (globale : 32, glissante : 64)
    const int* pos;
    float* part;                // [S][NH][HD + 2] : acc, m, l
    int* ticket;                // [NKV] : blocs finis (remis à 0 par le dernier)
    float* x;                   // sortie combinée, porte appliquée [NH * HD]
};
__device__ __forceinline__ void rope4(float* v, int lane, int pos, float base, int rot_pairs) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int i = lane + 32 * k;   // paire (i, i + 64)
        if (i < rot_pairs) {
            const float th = (float) pos * powf(base, -2.f * (float) i / (float) HD);
            float sn, cs; sincosf(th, &sn, &cs);
            const float a = v[k], b = v[k + 2];
            v[k] = a * cs - b * sn;
            v[k + 2] = a * sn + b * cs;
        }
    }
}
__global__ void __launch_bounds__(128) k_attn(AttnArgs A) {
    const int s = blockIdx.x, g = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32, h = g * 4 + warp;
    const int pos = *A.pos;
    const int start = A.swa ? max(0, pos - SWA + 1) : 0;
    const int n = pos - start + 1;
    const bool tiled = n > ATT_SWITCH;   // court : lecture directe, morceaux de 32 ; long : tuiles partagées, morceaux de 128
    const int amin = tiled ? 128 : 32, S = min(S_SPLIT, (n + amin - 1) / amin), cl = (n + S - 1) / S;
    if (s >= S) return;
    const int p0 = start + s * cl, p1 = min(pos + 1, p0 + cl);
    float* out = A.part + ((size_t) s * NH + h) * (HD + 2);
    __shared__ int last;
    __shared__ float qs[4][HD];
    // q de la tête h : norme, RoPE, échelle
    {
        float q[4];
        for (int k = 0; k < 4; ++k) q[k] = A.qkvg[h * HD + lane + 32 * k];
        const float r = rsqrtf(warp_sum(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) / HD + EPS);
        for (int k = 0; k < 4; ++k) q[k] *= r * A.qn[lane + 32 * k];
        rope4(q, lane, pos, A.base, A.rot_pairs);
        for (int k = 0; k < 4; ++k) qs[warp][lane + 32 * k] = q[k] * rsqrtf((float) HD);
    }
    // k, v courants : le bloc qui contient pos les écrit dans le cache avant de lire (norme et RoPE de k)
    if (p1 == pos + 1 && warp == 0) {
        float kc[4], vcur[4];
        for (int k = 0; k < 4; ++k) { kc[k] = A.qkvg[NH * HD + g * HD + lane + 32 * k]; vcur[k] = A.qkvg[NH * HD + NKV * HD + g * HD + lane + 32 * k]; }
        const float r = rsqrtf(warp_sum(kc[0] * kc[0] + kc[1] * kc[1] + kc[2] * kc[2] + kc[3] * kc[3]) / HD + EPS);
        for (int k = 0; k < 4; ++k) kc[k] *= r * A.kn[lane + 32 * k];
        rope4(kc, lane, pos, A.base, A.rot_pairs);
        const size_t slot = (size_t) (A.swa ? pos % A.cap : pos) * NKV * HD + g * HD;
        for (int k = 0; k < 4; ++k) { A.kc[slot + lane + 32 * k] = __float2half(kc[k]); A.vc[slot + lane + 32 * k] = __float2half(vcur[k]); }
        __threadfence_block();
    }
    __syncthreads();
    // Tuiles de 32 positions, chargées une fois en mémoire partagée pour les 4 têtes : le fil l calcule le score de
    // la position t + l (sa clé lue en mémoire partagée), puis chaque fil cumule ses 4 dimensions.
    __shared__ __align__(16) __half kt_[32][HD + 8];
    __shared__ __align__(16) __half vt_[32][HD];
    float m = -INFINITY, l = 0, acc[4] = {0, 0, 0, 0};
    if (!tiled) for (int t0 = p0; t0 < p1; t0 += 32) {
        const int p = t0 + lane;
        float sc = -INFINITY;
        if (p < p1) {
            const uint4* kr = (const uint4*) (A.kc + (size_t) (A.swa ? p % A.cap : p) * NKV * HD + g * HD);
            float d = 0;
#pragma unroll
            for (int c = 0; c < HD / 8; ++c) {
                const uint4 u = kr[c];
                const __half* hh = (const __half*) &u;
#pragma unroll
                for (int e = 0; e < 8; ++e) d += qs[warp][c * 8 + e] * __half2float(hh[e]);
            }
            sc = d;
        }
        const float mn = fmaxf(m, warp_max(sc)), corr = __expf(m - mn);
        const float e = p < p1 ? __expf(sc - mn) : 0.f;
        l = l * corr + warp_sum(e);
        for (int k = 0; k < 4; ++k) acc[k] *= corr;
        const int cnt = min(32, p1 - t0);
        for (int j = 0; j < cnt; ++j) {
            const float pj = SHFL(e, j);
            const __half* vr = A.vc + (size_t) (A.swa ? (t0 + j) % A.cap : t0 + j) * NKV * HD + g * HD;
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[k] += pj * __half2float(vr[lane + 32 * k]);
        }
        m = mn;
    }
    else for (int t0 = p0; t0 < p1; t0 += 32) {
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * HD / 8; i += 128) {
            const int j = i / (HD / 8), c = (i % (HD / 8)) * 8, p = t0 + j;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
            if (p < p1) {
                const size_t sl = (size_t) (A.swa ? p % A.cap : p) * NKV * HD + g * HD;
                kv = *(const uint4*) (A.kc + sl + c); vv = *(const uint4*) (A.vc + sl + c);
            }
            *(uint4*) &kt_[j][c] = kv; *(uint4*) &vt_[j][c] = vv;
        }
        __syncthreads();
        const int p = t0 + lane;
        float sc = -INFINITY;
        if (p < p1) {
            float d = 0;
#pragma unroll
            for (int c = 0; c < HD / 8; ++c) {
                const uint4 u = *(const uint4*) &kt_[lane][c * 8];
                const __half* hh = (const __half*) &u;
#pragma unroll
                for (int e = 0; e < 8; ++e) d += qs[warp][c * 8 + e] * __half2float(hh[e]);
            }
            sc = d;
        }
        const float mn = fmaxf(m, warp_max(sc)), corr = __expf(m - mn);
        const float e = p < p1 ? __expf(sc - mn) : 0.f;
        l = l * corr + warp_sum(e);
        for (int k = 0; k < 4; ++k) acc[k] *= corr;
        const int cnt = min(32, p1 - t0);
        for (int j = 0; j < cnt; ++j) {
            const float pj = SHFL(e, j);
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[k] += pj * __half2float(vt_[j][lane + 32 * k]);
        }
        m = mn;
    }
    for (int k = 0; k < 4; ++k) out[lane + 32 * k] = acc[k];
    if (lane == 0) { out[HD] = m; out[HD + 1] = l; }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&A.ticket[g], 1) == S - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    float M = -INFINITY;
    for (int t = 0; t < S; ++t) M = fmaxf(M, A.part[((size_t) t * NH + h) * (HD + 2) + HD]);
    float L = 0, o[4] = {0, 0, 0, 0};
    for (int t = 0; t < S; ++t) {
        const float* pp = A.part + ((size_t) t * NH + h) * (HD + 2);
        const float mm = pp[HD];
        if (mm == -INFINITY) continue;
        const float w = __expf(mm - M);
        L += w * pp[HD + 1];
        for (int k = 0; k < 4; ++k) o[k] += w * pp[lane + 32 * k];
    }
    const float gte = A.qkvg[NH * HD + 2 * NKV * HD + h], sg = 1.f / (1.f + __expf(-gte));
    for (int k = 0; k < 4; ++k) A.x[h * HD + lane + 32 * k] = o[k] / L * sg;
    if (threadIdx.x == 0) A.ticket[g] = 0;
}
// Combine les morceaux et applique la porte : x[h·128 + d] = sigmoid(porte_h) · Σ acc / Σ l
__global__ void k_attn_combine(const float* part, const float* qkvg, float* x) {
    const int h = blockIdx.x, d = threadIdx.x;   // 128 fils
    float M = -INFINITY;
    for (int s = 0; s < S_SPLIT; ++s) M = fmaxf(M, part[((size_t) s * NH + h) * (HD + 2) + HD]);
    float L = 0, o = 0;
    for (int s = 0; s < S_SPLIT; ++s) {
        const float* p = part + ((size_t) s * NH + h) * (HD + 2);
        if (p[HD] == -INFINITY) continue;
        const float w = __expf(p[HD] - M);
        L += w * p[HD + 1];
        o += w * p[d];
    }
    const float gte = qkvg[NH * HD + 2 * NKV * HD + h];
    x[h * HD + d] = o / L / (1.f + __expf(-gte));
}

// ── Préremplissage groupé : T tokens à la fois, poids en fp16 (copie déquantifiée), produits par hipBLAS / cuBLAS ──
#ifndef TP_CHUNK
#define TP_CHUNK 2048
#endif
constexpr int TP = TP_CHUNK;   // tokens par paquet
__global__ void k_dequant_q4(Q4W W, __half* out) {
    const size_t NB = W.K / 32, b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (size_t) W.N * NB) return;
    const float d = __half2float(W.d[b]);
    const uint8_t* q = (const uint8_t*) &W.qs[b];
    __half* o = out + b * 32;
    for (int j = 0; j < 16; ++j) { o[j] = __float2half(d * (float) ((q[j] & 15) - 8)); o[j + 16] = __float2half(d * (float) ((q[j] >> 4) - 8)); }
}
__global__ void k_embed_batch(Q6W W, const int* toks, float* Hres) {
    const int t = blockIdx.x;
    const int row = toks[t];
    for (int e = threadIdx.x; e < H; e += blockDim.x) {
        const int g = e / 32, w = e % 32;
        const size_t gi = (size_t) row * (H / 32) + g;
        const uint8_t* lo = (const uint8_t*) &W.lo[gi];
        const uint8_t* hi = (const uint8_t*) &W.hi[gi];
        const int lo4 = w < 16 ? (lo[w] & 15) : (lo[w - 16] >> 4);
        const int k = w / 16, s = (w % 16) / 4, bb = w % 4;
        const int q = (lo4 | (((hi[k * 4 + bb] >> (2 * s)) & 3) << 4)) - 32;
        Hres[(size_t) t * H + e] = W.d[(size_t) row * (H / 256) + g / 8] * (float) W.sc[2 * gi + w / 16] * (float) q;
    }
}
// un bloc par token : (RESNORM) h += norme(d)·wpost, puis xn = fp16(norme(h)·wpre)
template <int PRO>
__global__ void __launch_bounds__(THREADS) k_norm_batch(float* Hres, const float* D, const float* wpost, const float* wpre, __half* Xn) {
    __shared__ float red[WARPS + 1];
    const int t = blockIdx.x;
    float* h = Hres + (size_t) t * H;
    float v[H / THREADS];
    if (PRO == PRO_RESNORM) {
        const float* d = D + (size_t) t * H;
        float ss = 0;
        for (int k = 0; k < H / THREADS; ++k) { v[k] = d[threadIdx.x + THREADS * k]; ss += v[k] * v[k]; }
        const float rd = rsqrtf(block_sum(ss, red) / H + EPS);
        for (int k = 0; k < H / THREADS; ++k) { const int i = threadIdx.x + THREADS * k; v[k] = h[i] + v[k] * rd * wpost[i]; h[i] = v[k]; }
    } else {
        for (int k = 0; k < H / THREADS; ++k) v[k] = h[threadIdx.x + THREADS * k];
    }
    float ss = 0;
    for (int k = 0; k < H / THREADS; ++k) ss += v[k] * v[k];
    const float r = rsqrtf(block_sum(ss, red) / H + EPS);
    for (int k = 0; k < H / THREADS; ++k) { const int i = threadIdx.x + THREADS * k; Xn[(size_t) t * H + i] = __float2half(v[k] * r * wpre[i]); }
}
// convolution courte, un fil par canal, le temps en séquence ; Y : [T][3H] (b, c, x entrelacés par canal)
__global__ void k_conv_batch(const float* Y, int T, float* state, const float* w, __half* X) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= H) return;
    float s[CONVK - 1], wk[CONVK];
    for (int k = 0; k < CONVK - 1; ++k) s[k] = state[k * H + ch];
    for (int k = 0; k < CONVK; ++k) wk[k] = w[ch * CONVK + k];
    for (int t = 0; t < T; ++t) {
        const float* y = Y + (size_t) t * 3 * H + 3 * ch;
        const float bx = y[0] * y[2];
        float o = wk[CONVK - 1] * bx;
        for (int k = 0; k < CONVK - 1; ++k) o += wk[k] * s[k];
        for (int k = 0; k < CONVK - 2; ++k) s[k] = s[k + 1];
        s[CONVK - 2] = bx;
        X[(size_t) t * H + ch] = __float2half(y[1] * o);
    }
    for (int k = 0; k < CONVK - 1; ++k) state[k * H + ch] = s[k];
}
__global__ void k_swiglu_batch(const float* Y, __half* A, int n) {   // n = T * FF
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = Y[2 * (size_t) i], u = Y[2 * (size_t) i + 1];
    A[i] = __float2half(g / (1.f + __expf(-g)) * u);
}
// q : norme, RoPE, échelle -> Q ; k : norme, RoPE ; k et v vers le cache (globale) ou le tampon du paquet (glissante)
struct PrepArgs { const float* Y; int p0; const float *qn, *kn; float base; int rot_pairs; __half *kc, *vc; float* Q; __half* Qh; };
__global__ void k_qkv_prep(PrepArgs A) {
    const int t = blockIdx.x, hh = blockIdx.y, lane = threadIdx.x, pos = A.p0 + t;   // 32 fils
    const float* y = A.Y + (size_t) t * QKVG;
    float v[4];
    if (hh < NH) {
        for (int k = 0; k < 4; ++k) v[k] = y[hh * HD + lane + 32 * k];
        const float r = rsqrtf(warp_sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]) / HD + EPS);
        for (int k = 0; k < 4; ++k) v[k] *= r * A.qn[lane + 32 * k];
        rope4(v, lane, pos, A.base, A.rot_pairs);
        for (int k = 0; k < 4; ++k) {
            const float x = v[k] * rsqrtf((float) HD);
            A.Q[((size_t) t * NH + hh) * HD + lane + 32 * k] = x;
            A.Qh[((size_t) t * NH + hh) * HD + lane + 32 * k] = __float2half(x);
        }
    } else {
        const int g = hh - NH;
        for (int k = 0; k < 4; ++k) v[k] = y[NH * HD + g * HD + lane + 32 * k];
        const float r = rsqrtf(warp_sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]) / HD + EPS);
        for (int k = 0; k < 4; ++k) v[k] *= r * A.kn[lane + 32 * k];
        rope4(v, lane, pos, A.base, A.rot_pairs);
        const size_t slot = (size_t) t * NKV * HD + g * HD;   // kc, vc pointent déjà sur la position p0 (globale) ou le tampon
        for (int k = 0; k < 4; ++k) {
            A.kc[slot + lane + 32 * k] = __float2half(v[k]);
            A.vc[slot + lane + 32 * k] = __float2half(y[NH * HD + NKV * HD + g * HD + lane + 32 * k]);
        }
    }
}
struct PAttnArgs { const float *Q, *Y; int p0; int swa; const __half *kc, *vc; int cap; const __half *kt, *vt; __half* X; const __half* Qh; };
// Attention du paquet sur les unités matricielles (WMMA 16x16x16, fp16 -> fp32). Bloc (16 tokens, tête KV g),
// 4 warps : le warp w tient les 16 lignes de la tête 4g + w. Par tuile de 32 clés : S = Q·Kᵀ (2 x 8 mma), le
// softmax en ligne en mémoire partagée (masque causal et fenêtre), O = O·corr + P·V (O en mémoire partagée).
#ifndef ATTN_KT
#define ATTN_KT 32
#endif
constexpr int AQT = 16, AKT = ATTN_KT;   // clés par tuile (Radeon : 32, 64 Ko de mémoire partagée ; NVIDIA : 64 possible)
constexpr int ATTN_SMEM = 2 * AKT * HD * 2 + 4 * AQT * AKT * 4 + 4 * AQT * AKT * 2 + 4 * AQT * HD * 4 + 2 * 4 * AQT * 4;
__global__ void __launch_bounds__(128) k_attn_prefill_mma(PAttnArgs A, int T) {
    const int tb = blockIdx.x * AQT, g = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32, h = g * 4 + warp;
    // mémoire partagée dynamique (CUDA : 48 Ko au plus en statique ; ici ~61 Ko)
    extern __shared__ __align__(32) unsigned char smem[];
    auto ks = (__half(*)[HD]) smem;                                       // [AKT][HD]
    auto vs = (__half(*)[HD]) (smem + AKT * HD * 2);                      // [AKT][HD]
    auto ss = (float(*)[AQT][AKT]) (smem + 2 * AKT * HD * 2);             // [4][AQT][AKT]
    auto ps = (__half(*)[AQT][AKT]) ((unsigned char*) ss + 4 * AQT * AKT * 4);
    auto os = (float(*)[AQT][HD]) ((unsigned char*) ps + 4 * AQT * AKT * 2);
    auto mrow = (float(*)[AQT]) ((unsigned char*) os + 4 * AQT * HD * 4);
    auto lrow = (float(*)[AQT]) ((unsigned char*) mrow + 4 * AQT * 4);
    const int ntok = min(AQT, T - tb);
    wm::fragment<wm::accumulator, 16, 16, 16, float> o[HD / 16];
    for (int n = 0; n < HD / 16; ++n) wm::fill_fragment(o[n], 0.f);
    float mrow_r = -INFINITY, lrow_r = 0;
    const int half = lane % 2;
    const wf16* qg = (const wf16*) (A.Qh + ((size_t) tb * NH + h) * HD);   // 16 lignes, pas NH·HD (tampon de TP lignes)
    const int pos_first = A.p0 + tb, pos_last = A.p0 + tb + ntok - 1;
    const int kstart = A.swa ? max(0, pos_first - SWA + 1) : 0;
    for (int k0 = kstart; k0 <= pos_last; k0 += AKT) {
        __syncthreads();
        for (int i = threadIdx.x; i < AKT * HD / 8; i += blockDim.x) {   // 32 clés x 128 dims, par 8 demis
            const int j = i / (HD / 8), c = (i % (HD / 8)) * 8, p = k0 + j;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
            if (p <= pos_last) {
                const __half *kr, *vr;
                if (A.swa && p >= A.p0) { kr = A.kt + (size_t) (p - A.p0) * NKV * HD + g * HD; vr = A.vt + (size_t) (p - A.p0) * NKV * HD + g * HD; }
                else { const size_t sl = (size_t) (A.swa ? p % A.cap : p) * NKV * HD + g * HD; kr = A.kc + sl; vr = A.vc + sl; }
                kv = *(const uint4*) (kr + c); vv = *(const uint4*) (vr + c);
            }
            *(uint4*) &ks[j][c] = kv; *(uint4*) &vs[j][c] = vv;
        }
        __syncthreads();
        // S = Q · Kᵀ
        for (int n = 0; n < AKT / 16; ++n) {
            wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
            wm::fill_fragment(acc, 0.f);
            for (int kk = 0; kk < HD / 16; ++kk) {
                wm::fragment<wm::matrix_a, 16, 16, 16, wf16, wm::row_major> fa;
                wm::fragment<wm::matrix_b, 16, 16, 16, wf16, wm::col_major> fb;
                wm::load_matrix_sync(fa, qg + kk * 16, NH * HD);
                wm::load_matrix_sync(fb, (const wf16*) &ks[n * 16][kk * 16], HD);
                wm::mma_sync(acc, fa, fb, acc);
            }
            wm::store_matrix_sync(&ss[warp][0][n * 16], acc, AKT, wm::mem_row_major);
        }
        // softmax en ligne : la paire de fils (2r, 2r+1) tient la ligne r, chaque fil 16 clés
        WARP_SYNC();
        const int r = lane / 2;
        const int pos = A.p0 + tb + r, st = A.swa ? max(0, pos - SWA + 1) : 0;
        constexpr int KH = AKT / 2;
        float sc[KH], mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < KH; ++j) {
            const int key = half * KH + j, p = k0 + key;
            const bool ok = r < ntok && p >= st && p <= pos;
            sc[j] = ok ? ss[warp][r][key] : -INFINITY;
            mx = fmaxf(mx, sc[j]);
        }
        mx = fmaxf(mx, SHFL_XOR(mx, 1));
        const float mo = mrow_r, mn = fmaxf(mo, mx);
        float sum = 0;
#pragma unroll
        for (int j = 0; j < KH; ++j) {
            const float e = sc[j] == -INFINITY ? 0.f : __expf(sc[j] - mn);
            ps[warp][r][half * KH + j] = __float2half(e);
            sum += e;
        }
        sum += SHFL_XOR(sum, 1);
        const float corr = (mo == -INFINITY || mn == mo) ? 1.f : __expf(mo - mn);
        lrow_r = lrow_r * corr + sum;
        mrow_r = mn;
        // O (dans les registres) corrigé seulement si le maximum d'une ligne a monté
        if (warp_max(corr < 1.f ? 1.f : 0.f) > 0.f) {
            for (int n = 0; n < HD / 16; ++n) wm::store_matrix_sync(&os[warp][0][n * 16], o[n], HD, wm::mem_row_major);
            WARP_SYNC();
            for (int d = half * 64; d < half * 64 + 64; ++d) os[warp][r][d] *= corr;
            WARP_SYNC();
            for (int n = 0; n < HD / 16; ++n) wm::load_matrix_sync(o[n], &os[warp][0][n * 16], HD, wm::mem_row_major);
        }
        WARP_SYNC();
        // O += P · V
        for (int kk = 0; kk < AKT / 16; ++kk) {
            wm::fragment<wm::matrix_a, 16, 16, 16, wf16, wm::row_major> fa;
            wm::load_matrix_sync(fa, (const wf16*) &ps[warp][0][kk * 16], AKT);
            for (int n = 0; n < HD / 16; ++n) {
                wm::fragment<wm::matrix_b, 16, 16, 16, wf16, wm::row_major> fb;
                wm::load_matrix_sync(fb, (const wf16*) &vs[kk * 16][n * 16], HD);
                wm::mma_sync(o[n], fa, fb, o[n]);
            }
        }
    }
    for (int n = 0; n < HD / 16; ++n) wm::store_matrix_sync(&os[warp][0][n * 16], o[n], HD, wm::mem_row_major);
    if (half == 0) { mrow[warp][lane / 2] = mrow_r; lrow[warp][lane / 2] = lrow_r; }
    WARP_SYNC();
    for (int r = 0; r < ntok; ++r) {
        const int t = tb + r;
        const float gte = A.Y[(size_t) t * QKVG + NH * HD + 2 * NKV * HD + h], sg = 1.f / (1.f + __expf(-gte));
        const float il = 1.f / lrow[warp][r];
        for (int d = lane; d < HD; d += 32) A.X[(size_t) t * H + h * HD + d] = __float2half(os[warp][r][d] * il * sg);
    }
}
// glissante : les dernières positions du paquet entrent dans l'anneau
__global__ void k_swa_commit(const __half* kt, const __half* vt, int p0, int T, __half* kc, __half* vc) {
    const int t = blockIdx.x + max(0, T - SWA), i = threadIdx.x;   // NKV * HD fils
    const size_t dst = (size_t) ((p0 + t) % SWA) * NKV * HD + i, src = (size_t) t * NKV * HD + i;
    kc[dst] = kt[src]; vc[dst] = vt[src];
}

// ── Le modèle ───────────────────────────────────────────────────────────────────────────────
struct Layer {
    bool attn, swa;
    const float *attn_norm, *attn_post, *ffn_norm, *ffn_post;
    Q4W first;          // conv : in_proj entrelacé b|c|x par canal ; attention : q|k|v|porte
    Q4W out;            // conv : out_proj ; attention : attn_output
    Q4W gateup;         // gate/up entrelacés
    Q4W down;
    const float *conv_w, *qn, *kn;
    float* conv_state;
    __half *kc, *vc; int cap;
    __half *f_first, *f_out, *f_gateup, *f_down;   // copies fp16 pour le préremplissage
};
struct Model {
    Layer L[NL];
    Q6W emb;
    const float* out_norm;
    int ctx = 0;
    // tampons
    float *hA, *hB, *d1, *d2, *y, *ffn, *qkvg, *xattn, *part, *logits;
    int *d_tok, *d_pos, *d_ticket;
    unsigned long long* d_argmax;
    int cus = 64;
    // préremplissage
    float *pH, *pD, *pY, *pQ; __half *pX, *pX2, *pKt, *pVt, *pQh; int* pTok;
    blasHandle_t blas;
};
static Model M;
static cudaStream_t g_st;

static int grid_for(int N, int R) { return std::max(1, std::min((N + WARPS * R - 1) / (WARPS * R), M.cus * GRID_MULT)); }

template <int K, int R, int PRO, int EPI>
static void gemv(const Q4W& W, const Pro& P, const Epi& E) {
    k_gemv_q4<K, R, PRO, EPI><<<grid_for(W.N, R), THREADS, 0, g_st>>>(W, P, E);
}

// Un token : d_tok et d_pos déjà sur la carte. logits : avec la tête de sortie (sinon le préremplissage s'arrête
// au résidu de la dernière couche, inutile ici : seule la dernière position demande des logits).
static void forward(bool logits) {
    k_embed<<<H / 256, 256, 0, g_st>>>(M.emb, M.d_tok, M.hA, M.d_argmax);
    float *h = M.hA, *hn = M.hB;
    const float* prev_post = nullptr; const float* prev_d = nullptr;
    for (int il = 0; il < NL; ++il) {
        Layer& Y = M.L[il];
        // 1. entrée du bloc : norme (couche 0) ou résidu + norme d'après FFN + norme d'entrée
        Pro p1{h, prev_d, prev_post, Y.attn_norm, hn};
        if (!Y.attn) {
            Epi e{M.y, Y.conv_state, Y.conv_w, nullptr};
            if (il == 0) gemv<H, R_CONV, PRO_NORM, EPI_CONV>(Y.first, p1, e); else gemv<H, R_CONV, PRO_RESNORM, EPI_CONV>(Y.first, p1, e);
            gemv<H, R_OUT, PRO_RAW, EPI_STORE>(Y.out, Pro{nullptr, M.y, nullptr, nullptr, nullptr}, Epi{M.d1, nullptr, nullptr, nullptr});
        } else {
            Epi e{M.qkvg, nullptr, nullptr, nullptr};
            gemv<H, R_OUT, PRO_RESNORM, EPI_STORE>(Y.first, p1, e);
            AttnArgs a{M.qkvg, Y.qn, Y.kn, Y.kc, Y.vc, Y.cap, Y.swa ? 1 : 0, Y.swa ? 10000.f : 1000000.f, Y.swa ? 64 : 32, M.d_pos, M.part, M.d_ticket, M.xattn};
            k_attn<<<dim3(S_SPLIT, NKV), 128, 0, g_st>>>(a);
            gemv<H, R_OUT, PRO_RAW, EPI_STORE>(Y.out, Pro{nullptr, M.xattn, nullptr, nullptr, nullptr}, Epi{M.d1, nullptr, nullptr, nullptr});
        }
        if (il > 0) std::swap(h, hn);   // le résidu mis à jour par le prologue
        // 2. FFN : résidu + norme d'après bloc + norme FFN, gate/up -> SwiGLU, down
        gemv<H, R_GU, PRO_RESNORM, EPI_SWIGLU>(Y.gateup, Pro{h, M.d1, Y.attn_post, Y.ffn_norm, hn}, Epi{M.ffn, nullptr, nullptr, nullptr});
        std::swap(h, hn);
        gemv<FF, R_DOWN, PRO_RAW, EPI_STORE>(Y.down, Pro{nullptr, M.ffn, nullptr, nullptr, nullptr}, Epi{M.d2, nullptr, nullptr, nullptr});
        prev_post = Y.ffn_post; prev_d = M.d2;
    }
    if (logits)
        k_gemv_q6_logits<2><<<grid_for(VOCAB, 2), THREADS, 0, g_st>>>(M.emb, Pro{h, prev_d, prev_post, M.out_norm, hn}, Epi{M.logits, nullptr, nullptr, M.d_argmax});
}

static void load_model(const char* path, int ctx) {
    Gguf g = load_gguf(path);
    auto I = [&](const char* k) { auto it = g.ints.find(k); if (it == g.ints.end()) { std::fprintf(stderr, "clé absente : %s\n", k); std::exit(1); } return it->second; };
    if (I("lfm3.embedding_length")[0] != H || I("lfm3.block_count")[0] != NL || I("lfm3.feed_forward_length")[0] != FF ||
        I("lfm3.attention.head_count")[0] != NH || I("lfm3.attention.sliding_window")[0] != SWA || I("lfm3.shortconv.l_cache")[0] != CONVK) {
        std::fprintf(stderr, "ce GGUF n'est pas un LFM3 1.7B\n"); std::exit(1);
    }
    const auto kvh = I("lfm3.attention.head_count_kv");
    const auto swa = I("lfm3.attention.sliding_window_pattern");
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); M.cus = pr.multiProcessorCount;
    M.ctx = ctx;
    for (int il = 0; il < NL; ++il) {
        Layer& Y = M.L[il];
        const std::string b = "blk." + std::to_string(il) + ".";
        Y.attn = kvh[il] != 0; Y.swa = swa[il] != 0;
        if (Y.attn && kvh[il] != NKV) { std::fprintf(stderr, "têtes KV inattendues\n"); std::exit(1); }
        Y.attn_norm = upload_f32(need(g, b + "attn_norm.weight"));
        Y.attn_post = upload_f32(need(g, b + "post_attention_norm.weight"));
        Y.ffn_norm = upload_f32(need(g, b + "ffn_norm.weight"));
        Y.ffn_post = upload_f32(need(g, b + "post_ffw_norm.weight"));
        if (!Y.attn) {
            const GTensor& ip = need(g, b + "shortconv.in_proj.weight");
            std::vector<std::pair<const GTensor*, int>> rows;
            for (int c = 0; c < H; ++c) for (int k = 0; k < 3; ++k) rows.push_back({&ip, k * H + c});
            Y.first = pack_q4(rows, H);
            Y.out = pack_q4_whole(need(g, b + "shortconv.out_proj.weight"));
            Y.conv_w = upload_f32(need(g, b + "shortconv.conv.weight"));
            CK(cudaMalloc((void**) &Y.conv_state, (CONVK - 1) * H * 4));
        } else {
            const GTensor &q = need(g, b + "attn_q.weight"), &k = need(g, b + "attn_k.weight"), &v = need(g, b + "attn_v.weight"), &gt = need(g, b + "attn_gate.weight");
            std::vector<std::pair<const GTensor*, int>> rows;
            for (int r = 0; r < NH * HD; ++r) rows.push_back({&q, r});
            for (int r = 0; r < NKV * HD; ++r) rows.push_back({&k, r});
            for (int r = 0; r < NKV * HD; ++r) rows.push_back({&v, r});
            for (int r = 0; r < NH; ++r) rows.push_back({&gt, r});
            Y.first = pack_q4(rows, H);
            Y.out = pack_q4_whole(need(g, b + "attn_output.weight"));
            Y.qn = upload_f32(need(g, b + "attn_q_norm.weight"));
            Y.kn = upload_f32(need(g, b + "attn_k_norm.weight"));
            Y.cap = Y.swa ? SWA : ctx;
            CK(cudaMalloc((void**) &Y.kc, (size_t) Y.cap * NKV * HD * 2));
            CK(cudaMalloc((void**) &Y.vc, (size_t) Y.cap * NKV * HD * 2));
        }
        const GTensor &gt = need(g, b + "ffn_gate.weight"), &up = need(g, b + "ffn_up.weight");
        std::vector<std::pair<const GTensor*, int>> rows;
        for (int r = 0; r < FF; ++r) { rows.push_back({&gt, r}); rows.push_back({&up, r}); }
        Y.gateup = pack_q4(rows, H);
        Y.down = pack_q4_whole(need(g, b + "ffn_down.weight"));
        std::fprintf(stderr, "\rcouche %d/%d", il + 1, NL);
    }
    auto f16 = [](const Q4W& W) {
        __half* p; CK(cudaMalloc((void**) &p, (size_t) W.N * W.K * 2));
        const size_t nb = (size_t) W.N * (W.K / 32);
        k_dequant_q4<<<(unsigned) ((nb + 255) / 256), 256>>>(W, p);
        return p;
    };
    for (auto& Y : M.L) { Y.f_first = f16(Y.first); Y.f_out = f16(Y.out); Y.f_gateup = f16(Y.gateup); Y.f_down = f16(Y.down); }
    CK(cudaDeviceSynchronize());
    M.emb = pack_q6(need(g, "token_embd.weight"));
    M.out_norm = upload_f32(need(g, "output_norm.weight"));
    auto fl = [](size_t n) { float* p; CK(cudaMalloc((void**) &p, n * 4)); CK(cudaMemset(p, 0, n * 4)); return p; };
    M.hA = fl(H); M.hB = fl(H); M.d1 = fl(H); M.d2 = fl(H); M.y = fl(H); M.ffn = fl(FF); M.qkvg = fl(QKVG); M.xattn = fl(H);
    M.part = fl((size_t) S_SPLIT * NH * (HD + 2)); M.logits = fl(VOCAB);
    CK(cudaMalloc((void**) &M.d_tok, 4)); CK(cudaMalloc((void**) &M.d_pos, 4)); CK(cudaMalloc((void**) &M.d_argmax, 8));
    CK(cudaMalloc((void**) &M.d_ticket, NKV * 4)); CK(cudaMemset(M.d_ticket, 0, NKV * 4));
    M.pH = fl((size_t) TP * H); M.pD = fl((size_t) TP * H); M.pY = fl((size_t) TP * 2 * FF); M.pQ = fl((size_t) TP * NH * HD);
    auto hf = [](size_t n) { __half* p; CK(cudaMalloc((void**) &p, n * 2)); return p; };
    M.pX = hf((size_t) TP * FF); M.pX2 = hf((size_t) TP * H); M.pKt = hf((size_t) TP * NKV * HD); M.pVt = hf((size_t) TP * NKV * HD);
    M.pQh = hf((size_t) (TP + AQT) * NH * HD); CK(cudaMemset(M.pQh, 0, (size_t) (TP + AQT) * NH * HD * 2));
    CK(cudaMalloc((void**) &M.pTok, TP * 4));
    blasCreate(&M.blas); blasSetStream(M.blas, g_st);
    CK(cudaFuncSetAttribute((const void*) k_attn_prefill_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM));
    munmap((void*) g.base, g.size);
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    std::fprintf(stderr, "\rchargé : %.2f Gio utilisés sur la carte (ctx %d)\n", (tot - fr) / 1073741824.0, ctx);
}
static void reset_state() {
    for (auto& Y : M.L) if (!Y.attn) CK(cudaMemsetAsync(Y.conv_state, 0, (CONVK - 1) * H * 4, g_st));
}

// Un paquet de T <= TP tokens aux positions p0.. ; logits : ceux de la dernière position (dans M.logits, arg-max).
static void prefill(const int* toks, int T, int p0, bool logits) {
    CK(cudaMemcpyAsync(M.pTok, toks, T * 4, cudaMemcpyHostToDevice, g_st));
    k_embed_batch<<<T, 256, 0, g_st>>>(M.emb, M.pTok, M.pH);
    const float* prev_post = nullptr;
    for (int il = 0; il < NL; ++il) {
        Layer& Y = M.L[il];
        if (il == 0) k_norm_batch<PRO_NORM><<<T, THREADS, 0, g_st>>>(M.pH, nullptr, nullptr, Y.attn_norm, M.pX);
        else k_norm_batch<PRO_RESNORM><<<T, THREADS, 0, g_st>>>(M.pH, M.pD, prev_post, Y.attn_norm, M.pX);
        if (!Y.attn) {
            gemm_f16(M.blas, 3 * H, T, H, Y.f_first, M.pX, M.pY);
            k_conv_batch<<<H / 128, 128, 0, g_st>>>(M.pY, T, Y.conv_state, Y.conv_w, M.pX2);
        } else {
            gemm_f16(M.blas, QKVG, T, H, Y.f_first, M.pX, M.pY);
            __half* kdst = Y.swa ? M.pKt : Y.kc + (size_t) p0 * NKV * HD;
            __half* vdst = Y.swa ? M.pVt : Y.vc + (size_t) p0 * NKV * HD;
            const float base = Y.swa ? 10000.f : 1000000.f; const int rot = Y.swa ? 64 : 32;
            k_qkv_prep<<<dim3(T, NH + NKV), 32, 0, g_st>>>(PrepArgs{M.pY, p0, Y.qn, Y.kn, base, rot, kdst, vdst, M.pQ, M.pQh});
            const PAttnArgs pa{M.pQ, M.pY, p0, Y.swa ? 1 : 0, Y.kc, Y.vc, Y.cap, M.pKt, M.pVt, M.pX2, M.pQh};
            k_attn_prefill_mma<<<dim3((T + AQT - 1) / AQT, NKV), 128, ATTN_SMEM, g_st>>>(pa, T);
            if (const char* dbg = std::getenv("LFM3_DUMP_ATTN")) if (p0 == (int) std::min(512, TP) && (il == 2 || il == 4)) {
                std::vector<__half> hx((size_t) T * H); CK(cudaMemcpyAsync(hx.data(), M.pX2, hx.size() * 2, cudaMemcpyDeviceToHost, g_st)); CK(cudaStreamSynchronize(g_st));
                FILE* f = std::fopen((std::string(dbg) + "." + std::to_string(il)).c_str(), "wb"); std::fwrite(hx.data(), 2, hx.size(), f); std::fclose(f);
            }
            if (Y.swa) k_swa_commit<<<std::min(T, SWA), NKV * HD, 0, g_st>>>(M.pKt, M.pVt, p0, T, Y.kc, Y.vc);
        }
        gemm_f16(M.blas, H, T, H, Y.f_out, M.pX2, M.pD);
        k_norm_batch<PRO_RESNORM><<<T, THREADS, 0, g_st>>>(M.pH, M.pD, Y.attn_post, Y.ffn_norm, M.pX);
        gemm_f16(M.blas, 2 * FF, T, H, Y.f_gateup, M.pX, M.pY);
        k_swiglu_batch<<<(T * FF + 255) / 256, 256, 0, g_st>>>(M.pY, M.pX, T * FF);
        gemm_f16(M.blas, H, T, FF, Y.f_down, M.pX, M.pD);
        prev_post = Y.ffn_post;
    }
    if (logits) {
        CK(cudaMemsetAsync(M.d_argmax, 0, 8, g_st));
        k_gemv_q6_logits<2><<<grid_for(VOCAB, 2), THREADS, 0, g_st>>>(M.emb, Pro{M.pH + (size_t) (T - 1) * H, M.pD + (size_t) (T - 1) * H, prev_post, M.out_norm, M.hB},
                                                                     Epi{M.logits, nullptr, nullptr, M.d_argmax});
    }
    CK(cudaStreamSynchronize(g_st));   // pTok resservira
}
static double now_ms();
// tout un prompt (hors cache), par paquets ; le dernier token donne les logits
static void prefill_all(const std::vector<int>& ids, size_t from, const std::function<void(size_t)>& progress = nullptr) {
    for (size_t i = from; i < ids.size(); i += TP) {
        const int T = (int) std::min((size_t) TP, ids.size() - i);
        const double t0 = std::getenv("LFM3_TIME_CHUNKS") ? now_ms() : 0;
        prefill(&ids[i], T, (int) i, i + T == ids.size());
        if (t0 > 0) std::fprintf(stderr, "  paquet %zu (+%d) : %.0f ms\n", i, T, now_ms() - t0);
        if (progress) progress(i + T);
    }
}

// ── Graphes et pas ──────────────────────────────────────────────────────────────────────────
static cudaGraphExec_t g_full = nullptr, g_noout = nullptr;
static int* g_host_tp = nullptr;   // [2] épinglé : token, position
static void capture() {
    for (int k = 0; k < 2; ++k) {
        cudaGraph_t gr;
        CK(cudaStreamBeginCapture(g_st, cudaStreamCaptureModeGlobal));
        forward(k == 0);
        CK(cudaStreamEndCapture(g_st, &gr));
        CK(cudaGraphInstantiateWithFlags(k == 0 ? &g_full : &g_noout, gr, 0));
    }
}
static void step(int tok, int pos, bool logits) {
    // copie synchrone vers la carte, puis le graphe : le tampon épinglé est réutilisé au pas suivant
    g_host_tp[0] = tok; g_host_tp[1] = pos;
    CK(cudaMemcpyAsync(M.d_tok, &g_host_tp[0], 4, cudaMemcpyHostToDevice, g_st));
    CK(cudaMemcpyAsync(M.d_pos, &g_host_tp[1], 4, cudaMemcpyHostToDevice, g_st));
    CK(cudaGraphLaunch(logits ? g_full : g_noout, g_st));
    if (!logits) CK(cudaStreamSynchronize(g_st));   // le tampon épinglé resservira
}
static int argmax_result() {
    unsigned long long v; CK(cudaMemcpyAsync(&v, M.d_argmax, 8, cudaMemcpyDeviceToHost, g_st)); CK(cudaStreamSynchronize(g_st));
    return (int) ~(uint32_t) (v & 0xffffffffu);
}

// ── Échantillonnage (sur l'hôte, à partir des logits ; glouton par l'arg-max de la carte) ─────
struct Sampler { float temp = 0, top_p = 1, min_p = 0, rep = 1; int top_k = 0, rep_n = 64; std::mt19937_64 rng{42}; };
static std::vector<float> g_logits(VOCAB);
static int sample(const Sampler& s0, Sampler& s, const std::vector<int>& recent) {
    (void) s0;
    if (s.temp <= 0 && s.rep == 1.f) return argmax_result();
    CK(cudaMemcpyAsync(g_logits.data(), M.logits, VOCAB * 4, cudaMemcpyDeviceToHost, g_st)); CK(cudaStreamSynchronize(g_st));
    std::vector<float>& L = g_logits;
    if (s.rep != 1.f) {
        const size_t from = recent.size() > (size_t) s.rep_n ? recent.size() - s.rep_n : 0;
        std::vector<int> seen;
        for (size_t i = from; i < recent.size(); ++i) {
            const int t = recent[i];
            if (t < 0 || t >= VOCAB || std::find(seen.begin(), seen.end(), t) != seen.end()) continue;
            seen.push_back(t);
            L[t] = L[t] > 0 ? L[t] / s.rep : L[t] * s.rep;
        }
    }
    if (s.temp <= 0) return (int) (std::max_element(L.begin(), L.end()) - L.begin());
    const int k = s.top_k > 0 ? std::min(s.top_k, VOCAB) : VOCAB;
    std::vector<int> idx(VOCAB);
    for (int i = 0; i < VOCAB; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return L[a] > L[b]; });
    idx.resize(k);
    std::vector<double> p(k);
    const double mx = L[idx[0]];
    double z = 0;
    for (int i = 0; i < k; ++i) { p[i] = std::exp((L[idx[i]] - mx) / s.temp); z += p[i]; }
    int keep = k;
    if (s.min_p > 0) for (int i = 0; i < k; ++i) if (p[i] / p[0] < s.min_p) { keep = i; break; }
    if (s.top_p < 1) { double c = 0; for (int i = 0; i < keep; ++i) { c += p[i] / z; if (c >= s.top_p) { keep = i + 1; break; } } }
    double zz = 0; for (int i = 0; i < keep; ++i) zz += p[i];
    double u = std::uniform_real_distribution<double>(0, zz)(s.rng);
    for (int i = 0; i < keep; ++i) { u -= p[i]; if (u <= 0) return idx[i]; }
    return idx[keep - 1];
}

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<int> parse_ids(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) { char* e = nullptr; const long x = std::strtol(p, &e, 10); if (e == p) break; v.push_back((int) x); p = *e == ',' ? e + 1 : e; }
    return v;
}

int main(int argc, char** argv) {
    std::string model, dump; std::vector<int> ids; int n_gen = 64, ctx = 32768, dump_from = 0; bool serve = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto v = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "%s : valeur attendue\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--model") model = v();
        else if (a == "--serve") serve = true;
        else if (a == "--tokens") ids = parse_ids(v());
        else if (a == "--tokens-file") { FILE* f = std::fopen(v(), "r"); std::string t; char buf[65536]; size_t n; while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) t.append(buf, n); std::fclose(f); ids = parse_ids(t.c_str()); }
        else if (a == "--n") n_gen = std::atoi(v());
        else if (a == "--ctx") ctx = std::atoi(v());
        else if (a == "--dump-logits") dump = v();
        else if (a == "--dump-from") dump_from = std::atoi(v());
        else { std::fprintf(stderr, "option inconnue : %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || (ids.empty() && !serve)) {
        std::fprintf(stderr, "usage : lfm3-engine --model F.gguf (--serve | --tokens IDS | --tokens-file F) [--n N] [--ctx N] [--dump-logits F]\n");
        return 2;
    }
    CK(cudaStreamCreate(&g_st));
    CK(cudaMallocHost((void**) &g_host_tp, 8));
    const double t0 = now_ms();
    load_model(model.c_str(), ctx);
    capture();
    {   // échauffement : hipBLAS / cuBLAS charge ses noyaux au premier appel de chaque forme
        std::vector<int> dummy(TP, 1);
        for (int T : {TP, TP / 2, TP / 4, 64, 17, 1}) prefill(dummy.data(), T, 0, true);
        reset_state();
        CK(cudaStreamSynchronize(g_st));
    }
    std::fprintf(stderr, "prêt en %.1f s\n", (now_ms() - t0) / 1000);

    if (!serve) {   // essai : prompt, puis n tokens gloutons ; --dump-logits : les logits de chaque position générée
        if ((int) ids.size() + n_gen > ctx) { std::fprintf(stderr, "prompt + n dépasse --ctx\n"); return 2; }
        reset_state();
        FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
        if (n_gen == 0) {   // notation : les logits de chaque position du texte donné (validation contre une référence)
            size_t first = 0;
            if (dump_from > 0 && !std::getenv("LFM3_PP_STEP")) {   // le prompt par le préremplissage groupé, comme en service
                std::vector<int> head(ids.begin(), ids.begin() + dump_from + 1);
                prefill_all(head, 0);
                argmax_result();
                if (df) { CK(cudaMemcpy(g_logits.data(), M.logits, VOCAB * 4, cudaMemcpyDeviceToHost)); std::fwrite(g_logits.data(), 4, VOCAB, df); }
                first = dump_from + 1;
            }
            for (size_t i = first; i < ids.size(); ++i) {
                const bool want = (int) i >= dump_from;
                step(ids[i], (int) i, want);
                if (!want) continue;
                argmax_result();
                if (df) { CK(cudaMemcpy(g_logits.data(), M.logits, VOCAB * 4, cudaMemcpyDeviceToHost)); std::fwrite(g_logits.data(), 4, VOCAB, df); }
            }
            if (df) std::fclose(df);
            return 0;
        }
        const double a = now_ms();
        if (std::getenv("LFM3_PP_STEP")) { for (size_t i = 0; i + 1 < ids.size(); ++i) step(ids[i], (int) i, false); step(ids.back(), (int) ids.size() - 1, true); }
        else prefill_all(ids, 0);
        int tok = argmax_result();
        double b = now_ms();
        if (const char* rep = std::getenv("LFM3_REPEAT")) {   // mesure : le préremplissage refait, le meilleur temps
            double best = b - a;
            for (int k = 0; k < std::atoi(rep); ++k) {
                reset_state();
                const double x = now_ms();
                prefill_all(ids, 0);
                tok = argmax_result();
                best = std::min(best, now_ms() - x);
            }
            std::fprintf(stderr, "préremplissage, meilleur de %d : %.0f ms (%.0f tok/s)\n", std::atoi(rep) + 1, best, 1000.0 * ids.size() / best);
            b = now_ms(); (void) b;
        }
        std::vector<int> out;
        for (int k = 0; k < n_gen; ++k) {
            if (df) { CK(cudaMemcpy(g_logits.data(), M.logits, VOCAB * 4, cudaMemcpyDeviceToHost)); std::fwrite(g_logits.data(), 4, VOCAB, df); }
            out.push_back(tok);
            if (k + 1 == n_gen) break;
            step(tok, (int) ids.size() + k, true);
            tok = argmax_result();
        }
        const double c = now_ms();
        if (df) std::fclose(df);
        std::printf("gen:");
        for (int t : out) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "prompt %zu tokens : %.0f ms (%.0f tok/s) ; décodage %d tokens : %.1f tok/s\n", ids.size(), b - a,
                     1000.0 * ids.size() / (b - a), n_gen, 1000.0 * (n_gen - 1) / (c - b));
        return 0;
    }

    // --serve : le protocole de strata-ink2
    std::atomic<bool> stop{false};
    std::mutex mu; std::condition_variable cv; std::deque<std::string> q; bool eof = false;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "STOP") { stop.store(true); continue; }
            { std::lock_guard<std::mutex> lk(mu); q.push_back(line); }
            cv.notify_one();
        }
        { std::lock_guard<std::mutex> lk(mu); eof = true; }
        cv.notify_one();
    });
    reader.detach();
    std::printf("READY %d stop\n", ctx); std::fflush(stdout);
    std::vector<int> hist;   // tokens dont l'état (cache, convolution) est sur la carte
    Sampler smp;
    for (;;) {
        std::string line;
        { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return eof || !q.empty(); }); if (q.empty()) break; line = q.front(); q.pop_front(); }
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
        stop.store(false);
        std::vector<std::string> f;
        for (size_t p = 0; p < line.size();) { const size_t s2 = line.find(' ', p); f.push_back(line.substr(p, s2 == std::string::npos ? std::string::npos : s2 - p)); if (s2 == std::string::npos) break; p = s2 + 1; }
        if (f.size() < 3) { std::printf("ERR malformed GEN\n"); std::fflush(stdout); continue; }
        int max_new = std::atoi(f[1].c_str());
        std::vector<int> pr = parse_ids(f.back().c_str());
        Sampler s; s.rng = smp.rng;
        for (size_t k = 2; k + 1 < f.size(); ++k) {
            const size_t eq = f[k].find('='); if (eq == std::string::npos) continue;
            const std::string key = f[k].substr(0, eq), val = f[k].substr(eq + 1);
            if (key == "temperature") s.temp = std::strtof(val.c_str(), nullptr);
            else if (key == "top_p") s.top_p = std::strtof(val.c_str(), nullptr);
            else if (key == "top_k") s.top_k = std::atoi(val.c_str());
            else if (key == "min_p") s.min_p = std::strtof(val.c_str(), nullptr);
            else if (key == "repeat_penalty") s.rep = std::strtof(val.c_str(), nullptr);
            else if (key == "seed") s.rng.seed(std::strtoull(val.c_str(), nullptr, 10));
        }
        const auto bad = std::find_if(pr.begin(), pr.end(), [](int t) { return t < 0 || t >= VOCAB; });
        if (pr.empty() || bad != pr.end()) { std::printf("ERR empty prompt or unknown token\n"); std::fflush(stdout); continue; }
        if ((int) pr.size() >= ctx) { std::printf("ERR prompt of %zu tokens exceeds the context %d\n", pr.size(), ctx); std::fflush(stdout); continue; }
        max_new = std::max(0, std::min(max_new, ctx - (int) pr.size()));
        // L'état de la convolution ne se rembobine pas : on ne reprend que si le prompt prolonge hist.
        const bool reuse = !hist.empty() && pr.size() >= hist.size() && std::equal(hist.begin(), hist.end(), pr.begin());
        size_t start = reuse ? hist.size() : 0;
        if (!reuse) reset_state();
        const double t0 = now_ms();
        std::vector<int> recent(pr.begin(), pr.end());
        std::vector<int> fed(pr.begin(), pr.end());   // ce que l'état de la carte contient à la fin
        int next;
        if (start == pr.size()) {
            // même prompt : on refait le dernier token (son état est déjà écrit : on repart d'un cran)
            reset_state(); start = 0;
        }
        prefill_all(pr, start, [&](size_t done) {
            const double ms = now_ms() - t0;
            std::printf("PP %zu %zu %.0f %.1f\n", done, pr.size(), ms, 1000.0 * (done - start) / std::max(ms, 1.0)); std::fflush(stdout);
        });
        next = sample(s, s, recent);
        const double t1 = now_ms();
        std::vector<int> gen; const char* finish = "length"; int pos = (int) pr.size();
        for (int k = 0; k < max_new; ++k) {
            if (stop.load()) { finish = "cancel"; break; }
            gen.push_back(next); recent.push_back(next);
            std::printf("T %d\n", next); std::fflush(stdout);
            if (next == 124900 || next == 124895) { finish = "stop"; break; }   // <|im_end|>, <|endoftext|>
            if (k + 1 == max_new) break;
            step(next, pos++, true);
            fed.push_back(next);
            next = sample(s, s, recent);
        }
        const double t2 = now_ms();
        hist = fed;
        smp.rng = s.rng;
        std::printf("DONE %zu %zu %.0f %.0f %s\n", gen.size(), pr.size(), t1 - t0, t2 - t1, finish); std::fflush(stdout);
    }
    return 0;
}
