//four-sample batched radix-4, weight loads amortised across four
//independent input vectors, legitimate because fdn delay lines
//decouple consecutive samples
#include "../radix4_fdn.h"
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

template<int N, int L>
struct R4Batch4 {
    static constexpr int G = N / 16;
    static constexpr int Q = N / 4;
    alignas(16) float Wv[L][G][4][4][4];

    inline void forward4(const float* __restrict__ x0, const float* __restrict__ x1,
                         const float* __restrict__ x2, const float* __restrict__ x3,
                         float* __restrict__ y0, float* __restrict__ y1,
                         float* __restrict__ y2, float* __restrict__ y3) const noexcept {
        alignas(16) float t[4][2][256];
        const float* src[4] = {x0, x1, x2, x3};
        float* dst[4] = {t[0][0], t[1][0], t[2][0], t[3][0]};
        for (int l = 0; l < L; ++l) {
            for (int g = 0; g < G; ++g) {
                const int i = g * 4;
                const float (*W)[4][4] = Wv[l][g];
                //weights for this group loaded once, used by all four samples
                float32x4_t w[4][4];
                for (int j = 0; j < 4; ++j)
                    for (int r = 0; r < 4; ++r) w[j][r] = vld1q_f32(W[j][r]);
                #pragma clang loop unroll(full)
                for (int s = 0; s < 4; ++s) {
                    float32x4_t xj0 = vld1q_f32(src[s] + i);
                    float32x4_t xj1 = vld1q_f32(src[s] + i + Q);
                    float32x4_t xj2 = vld1q_f32(src[s] + i + 2*Q);
                    float32x4_t xj3 = vld1q_f32(src[s] + i + 3*Q);
                    float32x4x4_t o;
                    for (int r = 0; r < 4; ++r)
                        o.val[r] = vfmaq_f32(vfmaq_f32(vfmaq_f32(
                            vmulq_f32(w[0][r], xj0), w[1][r], xj1), w[2][r], xj2), w[3][r], xj3);
                    vst4q_f32(dst[s] + i * 4, o);
                }
            }
            for (int s = 0; s < 4; ++s) {
                const float* tmp = src[s];
                src[s] = dst[s];
                dst[s] = (dst[s] == t[s][0]) ? t[s][1] : t[s][0];
                (void)tmp;
            }
        }
        float* outs[4] = {y0, y1, y2, y3};
        for (int s = 0; s < 4; ++s)
            for (int i = 0; i < N; i += 4) vst1q_f32(outs[s] + i, vld1q_f32(src[s] + i));
    }
};

static double now_ns(){ return std::chrono::duration<double,std::nano>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }

template<int N, int L>
void row() {
    static sfdn::Radix4Ortho<N, L> single;
    static R4Batch4<N, L> batch;
    static float blocks[L * (N/4) * 16];
    for (int i = 0; i < L*(N/4)*16; ++i) blocks[i] = (double)rand()/RAND_MAX - 0.5;
    single.setBlocks(blocks);
    //same layout, wv[l][g][j][r][lane]
    __builtin_memcpy(batch.Wv, &single, sizeof(batch.Wv));

    alignas(16) static float x[4][256], y[4][256], yr[4][256];
    for (int s = 0; s < 4; ++s)
        for (int i = 0; i < N; ++i) x[s][i] = (double)rand()/RAND_MAX - 0.5;

    //sanity, batch must equal four single calls
    for (int s = 0; s < 4; ++s) single.forward(x[s], yr[s]);
    batch.forward4(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]);
    float err = 0;
    for (int s = 0; s < 4; ++s)
        for (int i = 0; i < N; ++i) err = fmaxf(err, fabsf(y[s][i] - yr[s][i]));

    auto bench = [&](auto&& fn, int calls_per_iter) {
        for (int i = 0; i < 2000; ++i) fn();
        std::vector<double> t(200);
        for (int b = 0; b < 200; ++b) {
            double t0 = now_ns();
            for (int i = 0; i < 512; ++i) { fn(); asm volatile("" ::: "memory"); }
            t[b] = (now_ns() - t0) / 512 / calls_per_iter;
        }
        std::sort(t.begin(), t.end());
        return t[100];
    };
    double t1 = bench([&](){ single.forward(x[0], y[0]); single.forward(x[1], y[1]);
                             single.forward(x[2], y[2]); single.forward(x[3], y[3]); }, 4);
    double t4 = bench([&](){ batch.forward4(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]); }, 4);
    printf("N=%3d L=%d | single %6.1f ns/sample | batch4 %6.1f ns/sample | %.2fx  [diff %.1e]\n",
           N, L, t1, t4, t1 / t4, err);
}

int main() {
    srand(42);
    row< 64, 2>();
    row<128, 2>();
    row<256, 2>();
    row<256, 4>();
    return 0;
}
