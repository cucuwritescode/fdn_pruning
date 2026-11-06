//radix-4 staged orthogonal FDN feedback matrix
//
/*
constant-geometry O(N·L) feedback for real-time reverb, b=4.

same idea as the b=2 kernel but four lanes at a time, so full mixing
needs half the layers. each layer grabs lanes b, b+N/4, b+N/2, b+3N/4
and writes them back contiguous (pease dataflow). the permutation costs
nothing, it lives in the load addressing, same way the perfect shuffle
rides along in vst2q for StagedOrtho.

for N <= 64 the whole state fits in neon registers so the kernel just
keeps it there across layers. no stores in between, the gather is
register naming and the interleave is an in-register 4x4 transpose.
on an m2, against the b=2 kernel at the layer counts we actually
shipped, this lands around 2.6x at N=16 and 32, 1.5x at N=64, parity
from 128 up.

author: facundo franchino, 2026
license: MIT
*/

#ifndef RADIX4_FDN_H_INCLUDED
#define RADIX4_FDN_H_INCLUDED

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define R4FDN_HAS_NEON 1
#endif

namespace sfdn {

#if R4FDN_HAS_NEON

//4x4 register transpose, the vst4 interleave without touching memory
#define SFDN_R4_TRANSPOSE(o0,o1,o2,o3, a0,a1,a2,a3) { \
    float32x4_t T0_ = vtrn1q_f32(a0, a1), T1_ = vtrn2q_f32(a0, a1); \
    float32x4_t T2_ = vtrn1q_f32(a2, a3), T3_ = vtrn2q_f32(a2, a3); \
    o0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(T0_), vreinterpretq_f64_f32(T2_))); \
    o1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(T1_), vreinterpretq_f64_f32(T3_))); \
    o2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(T0_), vreinterpretq_f64_f32(T2_))); \
    o3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(T1_), vreinterpretq_f64_f32(T3_))); }

//one output row for four blocks at once, tree-reduced to depth three
//instead of a serial chain of four
#define SFDN_R4_ROW(dst, W, x0, x1, x2, x3, r) { \
    float32x4_t p_ = vfmaq_f32(vmulq_f32(vld1q_f32(W[0][r]), x0), vld1q_f32(W[1][r]), x1); \
    float32x4_t q_ = vfmaq_f32(vmulq_f32(vld1q_f32(W[2][r]), x2), vld1q_f32(W[3][r]), x3); \
    dst = vaddq_f32(p_, q_); }

#endif

/*
Radix4Ortho<N, L>

constant-geometry radix-4 staged orthogonal matrix. weights are stored
vectorised across the four blocks of a group:

    wv_[l][g][j][r][b] = W_l[4g+b][r][j]

so a column of four weights loads as one vector whose lanes are the
four blocks of the group.
*/

template <int N, int L>
class Radix4Ortho {
    static_assert(N >= 16 && N % 16 == 0, "N must be a multiple of 16");
    static_assert(L >= 1, "need at least one layer");

    static constexpr int kGroups = N / 16;
    static constexpr int kBlocks = N / 4;

public:
    Radix4Ortho() noexcept {
        std::memset(wv_, 0, sizeof(wv_));
    }

    //blocks laid out (L, N/4, 4, 4) row-major, blocks[l][k][r][j],
    //the layout produced by staged_fdn.radix4 (numpy default order)
    void setBlocks(const float* blocks) noexcept {
        for (int l = 0; l < L; ++l)
            for (int g = 0; g < kGroups; ++g)
                for (int b = 0; b < 4; ++b)
                    for (int r = 0; r < 4; ++r)
                        for (int j = 0; j < 4; ++j)
                            wv_[l][g][j][r][b] =
                                blocks[((l * kBlocks + 4 * g + b) * 4 + r) * 4 + j];
    }

    inline void forward(const float* __restrict__ x,
                        float* __restrict__ y) const noexcept {
#if R4FDN_HAS_NEON
        if constexpr (N == 16)      forwardReg16(x, y);
        else if constexpr (N == 32) forwardReg32(x, y);
        else if constexpr (N == 64) forwardReg64(x, y);
        else                        forwardGeneric(x, y);
#else
        forwardScalar(x, y);
#endif
    }

    static constexpr int size()       noexcept { return N; }
    static constexpr int layers()     noexcept { return L; }
    static constexpr int blocks()     noexcept { return kBlocks * L; }
    static constexpr int paramCount() noexcept { return 6 * kBlocks * L; }
    static constexpr int flops()      noexcept { return 8 * N * L; }

private:
    alignas(16) float wv_[L][kGroups][4][4][4];

#if R4FDN_HAS_NEON

    //n=16, whole state is v0..v3, one group per layer
    inline void forwardReg16(const float* __restrict__ x,
                             float* __restrict__ y) const noexcept {
        float32x4_t v0 = vld1q_f32(x), v1 = vld1q_f32(x + 4),
                    v2 = vld1q_f32(x + 8), v3 = vld1q_f32(x + 12);
#pragma clang loop unroll(full)
        for (int l = 0; l < L; ++l) {
            const float (*W)[4][4] = wv_[l][0];
            float32x4_t a0, a1, a2, a3;
            SFDN_R4_ROW(a0, W, v0, v1, v2, v3, 0)
            SFDN_R4_ROW(a1, W, v0, v1, v2, v3, 1)
            SFDN_R4_ROW(a2, W, v0, v1, v2, v3, 2)
            SFDN_R4_ROW(a3, W, v0, v1, v2, v3, 3)
            SFDN_R4_TRANSPOSE(v0, v1, v2, v3, a0, a1, a2, a3)
        }
        vst1q_f32(y, v0); vst1q_f32(y + 4, v1);
        vst1q_f32(y + 8, v2); vst1q_f32(y + 12, v3);
    }

    //n=32, state in v0..v7. blocks 0..3 read lanes of v0,v2,v4,v6 and
    //blocks 4..7 of v1,v3,v5,v7, the strided gather is register naming
    inline void forwardReg32(const float* __restrict__ x,
                             float* __restrict__ y) const noexcept {
        float32x4_t v0 = vld1q_f32(x),      v1 = vld1q_f32(x + 4),
                    v2 = vld1q_f32(x + 8),  v3 = vld1q_f32(x + 12),
                    v4 = vld1q_f32(x + 16), v5 = vld1q_f32(x + 20),
                    v6 = vld1q_f32(x + 24), v7 = vld1q_f32(x + 28);
#pragma clang loop unroll(full)
        for (int l = 0; l < L; ++l) {
            const float (*WA)[4][4] = wv_[l][0];
            const float (*WB)[4][4] = wv_[l][1];
            float32x4_t a0, a1, a2, a3, b0, b1, b2, b3;
            SFDN_R4_ROW(a0, WA, v0, v2, v4, v6, 0)
            SFDN_R4_ROW(a1, WA, v0, v2, v4, v6, 1)
            SFDN_R4_ROW(a2, WA, v0, v2, v4, v6, 2)
            SFDN_R4_ROW(a3, WA, v0, v2, v4, v6, 3)
            SFDN_R4_ROW(b0, WB, v1, v3, v5, v7, 0)
            SFDN_R4_ROW(b1, WB, v1, v3, v5, v7, 1)
            SFDN_R4_ROW(b2, WB, v1, v3, v5, v7, 2)
            SFDN_R4_ROW(b3, WB, v1, v3, v5, v7, 3)
            SFDN_R4_TRANSPOSE(v0, v1, v2, v3, a0, a1, a2, a3)
            SFDN_R4_TRANSPOSE(v4, v5, v6, v7, b0, b1, b2, b3)
        }
        vst1q_f32(y, v0);      vst1q_f32(y + 4, v1);
        vst1q_f32(y + 8, v2);  vst1q_f32(y + 12, v3);
        vst1q_f32(y + 16, v4); vst1q_f32(y + 20, v5);
        vst1q_f32(y + 24, v6); vst1q_f32(y + 28, v7);
    }

    //n=64, sixteen state registers. group g consumes exactly v[g],
    //v[g+4], v[g+8], v[g+12], so outputs retire into dead inputs
    inline void forwardReg64(const float* __restrict__ x,
                             float* __restrict__ y) const noexcept {
        float32x4_t v[16], w[16];
        for (int k = 0; k < 16; ++k) v[k] = vld1q_f32(x + 4 * k);
#pragma clang loop unroll(full)
        for (int l = 0; l < L; ++l) {
#pragma clang loop unroll(full)
            for (int g = 0; g < 4; ++g) {
                const float (*W)[4][4] = wv_[l][g];
                float32x4_t a0, a1, a2, a3;
                SFDN_R4_ROW(a0, W, v[g], v[g+4], v[g+8], v[g+12], 0)
                SFDN_R4_ROW(a1, W, v[g], v[g+4], v[g+8], v[g+12], 1)
                SFDN_R4_ROW(a2, W, v[g], v[g+4], v[g+8], v[g+12], 2)
                SFDN_R4_ROW(a3, W, v[g], v[g+4], v[g+8], v[g+12], 3)
                SFDN_R4_TRANSPOSE(w[4*g], w[4*g+1], w[4*g+2], w[4*g+3],
                                  a0, a1, a2, a3)
            }
            for (int k = 0; k < 16; ++k) v[k] = w[k];
        }
        for (int k = 0; k < 16; ++k) vst1q_f32(y + 4 * k, v[k]);
    }

    //n >= 128, the state no longer fits the register file, ping-pong
    //through stack buffers, tree-reduced fma, vst4 interleave
    inline void forwardGeneric(const float* __restrict__ x,
                               float* __restrict__ y) const noexcept {
        constexpr int Q = N / 4;
        alignas(16) float t0[N], t1[N];
        const float* src = x; float* dst = t0;
        for (int l = 0; l < L; ++l) {
            for (int g = 0; g < kGroups; ++g) {
                const int i = g * 4;
                float32x4_t xj0 = vld1q_f32(src + i);
                float32x4_t xj1 = vld1q_f32(src + i + Q);
                float32x4_t xj2 = vld1q_f32(src + i + 2 * Q);
                float32x4_t xj3 = vld1q_f32(src + i + 3 * Q);
                const float (*W)[4][4] = wv_[l][g];
                float32x4x4_t o;
                SFDN_R4_ROW(o.val[0], W, xj0, xj1, xj2, xj3, 0)
                SFDN_R4_ROW(o.val[1], W, xj0, xj1, xj2, xj3, 1)
                SFDN_R4_ROW(o.val[2], W, xj0, xj1, xj2, xj3, 2)
                SFDN_R4_ROW(o.val[3], W, xj0, xj1, xj2, xj3, 3)
                vst4q_f32(dst + i * 4, o);
            }
            const float* s = src; src = dst; dst = (dst == t0) ? t1 : t0;
            (void)s;
        }
        for (int i = 0; i < N; i += 4) vst1q_f32(y + i, vld1q_f32(src + i));
    }

#else

    inline void forwardScalar(const float* __restrict__ x,
                              float* __restrict__ y) const noexcept {
        constexpr int Q = N / 4;
        float t0[N], t1[N];
        const float* src = x; float* dst = t0;
        for (int l = 0; l < L; ++l) {
            for (int g = 0; g < kGroups; ++g)
                for (int b = 0; b < 4; ++b)
                    for (int r = 0; r < 4; ++r) {
                        float acc = 0.f;
                        for (int j = 0; j < 4; ++j)
                            acc += wv_[l][g][j][r][b] * src[(g * 4 + b) + Q * j];
                        dst[4 * (g * 4 + b) + r] = acc;
                    }
            const float* s = src; src = dst; dst = (dst == t0) ? t1 : t0;
            (void)s;
        }
        std::memcpy(y, src, N * sizeof(float));
    }

#endif
};

#if R4FDN_HAS_NEON
#undef SFDN_R4_TRANSPOSE
#undef SFDN_R4_ROW
#endif

//============================================================================
//Radix4FDN<N, L>
//============================================================================
template <int N, int L>
class Radix4FDN {
    static constexpr int kMaxDelay = 8192;

public:
    Radix4FDN() noexcept {
        std::memset(this, 0, sizeof(*this));
    }

    void setDelays(const int* d)        noexcept { for (int i=0;i<N;++i) delay_[i] = d[i]<kMaxDelay ? d[i] : kMaxDelay-1; }
    void setInputGains(const float* b)  noexcept { std::memcpy(b_, b, N*sizeof(float)); }
    void setOutputGains(const float* c) noexcept { std::memcpy(c_, c, N*sizeof(float)); }
    void setDecayGains(const float* gm) noexcept { std::memcpy(gamma_m_, gm, N*sizeof(float)); }
    void setLowpassAlpha(float a)       noexcept { alpha_ = a; }

    Radix4Ortho<N,L>&       feedback()       noexcept { return fb_; }
    const Radix4Ortho<N,L>& feedback() const noexcept { return fb_; }

    void reset() noexcept {
        std::memset(buf_, 0, sizeof(buf_));
        std::memset(ptr_, 0, sizeof(ptr_));
        std::memset(filt_, 0, sizeof(filt_));
    }

    inline float processSample(float input) noexcept {
        alignas(16) float outs[N], atten[N], fb_out[N];
        for (int i=0;i<N;++i) outs[i] = buf_[i][ptr_[i]];
        for (int i=0;i<N;++i) filt_[i] = (1.f-alpha_)*outs[i] + alpha_*filt_[i];
        float output = 0.f;
        for (int i=0;i<N;++i) output += c_[i]*filt_[i];
        for (int i=0;i<N;++i) atten[i] = filt_[i]*gamma_m_[i];
        fb_.forward(atten, fb_out);
        for (int i=0;i<N;++i) {
            buf_[i][ptr_[i]] = fb_out[i] + b_[i]*input;
            ptr_[i] = (ptr_[i]+1 < delay_[i]) ? ptr_[i]+1 : 0;
        }
        return output;
    }

private:
    Radix4Ortho<N,L> fb_;
    float buf_[N][kMaxDelay];
    int ptr_[N], delay_[N];
    float filt_[N], alpha_;
    float b_[N], c_[N], gamma_m_[N];
};

}  //namespace sfdn
#endif
