//staged orthogonal FDN feedback matrix
//
/*
constant-geometry O(N·L) feedback for real-time reverberation,
it is orthogonal by construction, with no allocation in the audio path.
all memory accesses are contiguous, with zero scalar gathers.

author: facundo franchino, 2026
license: MIT

*/

#ifndef STAGED_FDN_H_INCLUDED
#define STAGED_FDN_H_INCLUDED

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define SFDN_HAS_NEON 1
#elif defined(__SSE2__)
    #include <immintrin.h>
    #define SFDN_HAS_SSE 1
#endif

namespace sfdn {

/*
stagedOrtho<N, L>

constant-geometry staged orthogonal matrix.
same perfect-shuffle permutation at every layer.

per-layer hot path (zero gathers, all contiguous):
   1, linear read from first half:  vld1q(src + i)
   2, linear read from second half: vld1q(src + half + i)
   3, rotate:                       vmls / vmla
   4, interleave-write:             vst2q (shuffle is free in hardware)

*/


template <int N, int L>
class StagedOrtho {
    static_assert(N >= 2 && N % 2 == 0, "N must be even and >= 2");
    static_assert(N <= 65535, "N must fit in permutation indices");
    static_assert(L >= 1, "need at least one layer");

    static constexpr int kBlocks = N / 2;
    static constexpr int kHalf   = N / 2;
    static constexpr int kTotalAngles = L * kBlocks;

public:
    using perm_idx_t = std::conditional_t<(N <= 255), uint8_t, uint16_t>;

    StagedOrtho() noexcept {
        std::memset(cos_, 0, sizeof(cos_));
        std::memset(sin_, 0, sizeof(sin_));
    }

    void setAngles(const float* angles) noexcept {
        for (int l = 0; l < L; ++l)
            for (int k = 0; k < kBlocks; ++k) {
                cos_[l][k] = std::cos(angles[l * kBlocks + k]);
                sin_[l][k] = std::sin(angles[l * kBlocks + k]);
            }
    }

    void setCosSin(const float* cos_tab, const float* sin_tab) noexcept {
        std::memcpy(cos_, cos_tab, sizeof(cos_));
        std::memcpy(sin_, sin_tab, sizeof(sin_));
    }

    //constant geometry, the permutation is hardcoded as perfect-shuffle.
    //API kept for compatibility; argument ignored.
    void setPermutation(int, const perm_idx_t*) noexcept {}

    //real-time audio path
    //
    //constant-geometry perfect-shuffle:
    //input  layout: [a0, a1, ..., a_{H-1}, b0, b1, ..., b_{H-1}]
    //                   (first half / second half)
    //output layout: [a0, b0, a1, b1, ..., a_{H-1}, b_{H-1}]
    //                   (interleaved pairs)
    //
    //the rotation operates on interleaved pairs (block k = elements 2k, 2k+1).
    //vst2q writes the rotated result in interleaved form.
    //the next layer reads it as first-half / second-half again.
    //
    //so the data flow per layer is:
    //split-layout → read lo/hi → rotate → vst2q interleave → split-layout
    //
    //but, after vst2q the data is interleaved [a0,b0,a1,b1,...],
    //which is NOT the split layout the next layer expects.
    //the split layout is [a0,a1,...,a_{H-1}, b0,b1,...,b_{H-1}].
    //
    //these are the SAME data, just interpreted differently.
    //after interleave-write, element at position 2k is the "first" of pair k,
    //and element at position 2k+1 is the "second" of pair k.
    //the next layer's "first half" is positions 0..H-1 and "second half" is H..N-1.
    //position 2k maps to: if 2k < H then first-half[2k], else second-half[2k-H].
    //
    //this is actually the perfect shuffle, the interleaved output, when read as two halves,
    //gives exactly the shuffled pairing for the next layer.
    //
    //therefore, just alternate between interleaved and split layout each layer.
    //layer 0: read split → write interleaved
    //layer 1: read interleaved-as-split → write interleaved
    // ... and so on. the "shuffle" is implicit in the layout reinterpretation.

    inline void forward(const float* __restrict__ x,
                        float* __restrict__ y) const noexcept {
        alignas(16) float a[N];
        alignas(16) float b[N];
        std::memcpy(a, x, N * sizeof(float));

        const float* __restrict__ src = a;
        float* __restrict__ dst = b;

#pragma clang loop unroll_count(16)
        for (int l = 0; l < L; ++l) {
            //prefetch next layer's cos/sin while computing current
            if (l + 1 < L) {
                __builtin_prefetch(cos_[l + 1], 0, 3);
                __builtin_prefetch(sin_[l + 1], 0, 3);
            }
            shuffleRotate(dst, src, cos_[l], sin_[l]);
            const float* t = src; src = dst; dst = const_cast<float*>(t);
        }

        std::memcpy(y, src, N * sizeof(float));
    }

    static constexpr int size()         noexcept { return N; }
    static constexpr int layers()       noexcept { return L; }
    static constexpr int blocks()       noexcept { return kBlocks; }
    static constexpr int paramCount()   noexcept { return kTotalAngles; }
    static constexpr int storedFloats() noexcept { return 2 * kTotalAngles; }
    static constexpr int flops()        noexcept { return 6 * kBlocks * L; }

private:
    alignas(16) float cos_[L][kBlocks];
    alignas(16) float sin_[L][kBlocks];

    //the core operation; read two halves linearly, rotate, interleave-write.
    //all memory accesses are contiguous. The shuffle is free inside vst2q.

#if SFDN_HAS_NEON

    static inline void shuffleRotate(float* __restrict__ dst,
                                     const float* __restrict__ src,
                                     const float* __restrict__ c,
                                     const float* __restrict__ s) noexcept {
        constexpr int kSimd = kHalf / 4;
        const float* __restrict__ lo = src;
        const float* __restrict__ hi = src + kHalf;
#pragma clang loop unroll_count(8)
        for (int v = 0; v < kSimd; ++v) {
            //prefetch next iteration's data
            if constexpr (kSimd > 1) {
                __builtin_prefetch(lo + (v+1) * 4, 0, 3);
                __builtin_prefetch(hi + (v+1) * 4, 0, 3);
            }
            float32x4_t a = vld1q_f32(lo + v * 4);
            float32x4_t b = vld1q_f32(hi + v * 4);
            float32x4_t cv = vld1q_f32(c + v * 4);
            float32x4_t sv = vld1q_f32(s + v * 4);
            float32x4x2_t out;
            out.val[0] = vmlsq_f32(vmulq_f32(cv, a), sv, b);
            out.val[1] = vmlaq_f32(vmulq_f32(sv, a), cv, b);
            vst2q_f32(dst + v * 8, out);
        }
        for (int k = kSimd * 4; k < kHalf; ++k) {
            float va = lo[k], vb = hi[k];
            dst[2*k]   = c[k]*va - s[k]*vb;
            dst[2*k+1] = s[k]*va + c[k]*vb;
        }
    }

#elif SFDN_HAS_SSE

    static inline void shuffleRotate(float* __restrict__ dst,
                                     const float* __restrict__ src,
                                     const float* __restrict__ c,
                                     const float* __restrict__ s) noexcept {
        constexpr int kSimd = kHalf / 4;
        const float* lo = src;
        const float* hi = src + kHalf;
        for (int v = 0; v < kSimd; ++v) {
            __m128 a  = _mm_load_ps(lo + v * 4);
            __m128 b  = _mm_load_ps(hi + v * 4);
            __m128 cv = _mm_load_ps(c + v * 4);
            __m128 sv = _mm_load_ps(s + v * 4);
            __m128 ye = _mm_sub_ps(_mm_mul_ps(cv, a), _mm_mul_ps(sv, b));
            __m128 yo = _mm_add_ps(_mm_mul_ps(sv, a), _mm_mul_ps(cv, b));
            _mm_store_ps(dst + v * 8,     _mm_unpacklo_ps(ye, yo));
            _mm_store_ps(dst + v * 8 + 4, _mm_unpackhi_ps(ye, yo));
        }
        for (int k = kSimd * 4; k < kHalf; ++k) {
            float va = lo[k], vb = hi[k];
            dst[2*k]   = c[k]*va - s[k]*vb;
            dst[2*k+1] = s[k]*va + c[k]*vb;
        }
    }

#else

    static inline void shuffleRotate(float* __restrict__ dst,
                                     const float* __restrict__ src,
                                     const float* __restrict__ c,
                                     const float* __restrict__ s) noexcept {
        const float* lo = src;
        const float* hi = src + kHalf;
        for (int k = 0; k < kHalf; ++k) {
            float va = lo[k], vb = hi[k];
            dst[2*k]   = c[k]*va - s[k]*vb;
            dst[2*k+1] = s[k]*va + c[k]*vb;
        }
    }

#endif
};

//============================================================================
//StagedFDN<N, L>
//============================================================================
template <int N, int L>
class StagedFDN {
    static constexpr int kMaxDelay = 8192;

public:
    StagedFDN() noexcept {
        std::memset(this, 0, sizeof(*this));
    }

    void setDelays(const int* d)           noexcept { for (int i=0;i<N;++i) delay_[i] = d[i]<kMaxDelay ? d[i] : kMaxDelay-1; }
    void setInputGains(const float* b)     noexcept { std::memcpy(b_, b, N*sizeof(float)); }
    void setOutputGains(const float* c)    noexcept { std::memcpy(c_, c, N*sizeof(float)); }
    void setDecayGains(const float* gm)    noexcept { std::memcpy(gamma_m_, gm, N*sizeof(float)); }
    void setLowpassAlpha(float a)          noexcept { alpha_ = a; }

    StagedOrtho<N,L>&       feedback()       noexcept { return fb_; }
    const StagedOrtho<N,L>& feedback() const noexcept { return fb_; }

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
    StagedOrtho<N,L> fb_;
    float buf_[N][kMaxDelay];
    int ptr_[N], delay_[N];
    float filt_[N], alpha_;
    float b_[N], c_[N], gamma_m_[N];
};

}  //namespace sfdn
#endif
