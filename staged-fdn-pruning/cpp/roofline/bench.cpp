//staged b=2 l=8 against monarch b=16 l=2 on the same harness.
//the staged kernel is the production one from ../staged_fdn.h;
//the monarch kernel is written to the same standard but has not
//been integrated into the audio path. peak.cpp establishes the
//ceiling of the roofline.
#include "../staged_fdn.h"
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static constexpr int N = 256;
static constexpr int NB = 16;
static constexpr int BLOCKS = N / NB;

//monarch: y = b2 p b1 x, where b1 and b2 are block-diagonal with
//sixteen dense 16x16 orthogonal blocks and p is the block transpose.
//weights are stored column-major per block, so the matvec proceeds
//by columns, four fmas per column into four running accumulators.
//
//the transpose permutation is folded into the second layer's loads:
//layer one stores contiguously, layer two reads element j of its
//block from position j*16 + b, a strided scalar load feeding vfmaq_n
//directly. nothing is ever scattered.
struct Monarch {
    alignas(16) float W[2][BLOCKS][NB][NB];

    inline void forward(const float* __restrict__ x, float* __restrict__ y) const noexcept {
        alignas(16) float mid[N];
        for (int b = 0; b < BLOCKS; ++b) {
            const float* __restrict__ xb = x + b * NB;
            float32x4_t x0 = vld1q_f32(xb), x1 = vld1q_f32(xb+4),
                        x2 = vld1q_f32(xb+8), x3 = vld1q_f32(xb+12);
            float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
            const float (*Wb)[NB] = W[0][b];
            #define COL(j, xv, lane) { \
                const float* c_ = Wb[j]; \
                a0 = vfmaq_laneq_f32(a0, vld1q_f32(c_),    xv, lane); \
                a1 = vfmaq_laneq_f32(a1, vld1q_f32(c_+4),  xv, lane); \
                a2 = vfmaq_laneq_f32(a2, vld1q_f32(c_+8),  xv, lane); \
                a3 = vfmaq_laneq_f32(a3, vld1q_f32(c_+12), xv, lane); }
            COL(0,x0,0) COL(1,x0,1) COL(2,x0,2) COL(3,x0,3)
            COL(4,x1,0) COL(5,x1,1) COL(6,x1,2) COL(7,x1,3)
            COL(8,x2,0) COL(9,x2,1) COL(10,x2,2) COL(11,x2,3)
            COL(12,x3,0) COL(13,x3,1) COL(14,x3,2) COL(15,x3,3)
            #undef COL
            float* mb = mid + b * NB;
            vst1q_f32(mb, a0); vst1q_f32(mb+4, a1);
            vst1q_f32(mb+8, a2); vst1q_f32(mb+12, a3);
        }
        for (int b = 0; b < BLOCKS; ++b) {
            float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
            const float (*Wb)[NB] = W[1][b];
            const float* __restrict__ mb = mid + b;
            #define COLN(j) { const float* c_ = Wb[j]; const float xj = mb[(j)*NB]; \
                a0 = vfmaq_n_f32(a0, vld1q_f32(c_),    xj); \
                a1 = vfmaq_n_f32(a1, vld1q_f32(c_+4),  xj); \
                a2 = vfmaq_n_f32(a2, vld1q_f32(c_+8),  xj); \
                a3 = vfmaq_n_f32(a3, vld1q_f32(c_+12), xj); }
            COLN(0) COLN(1) COLN(2) COLN(3) COLN(4) COLN(5) COLN(6) COLN(7)
            COLN(8) COLN(9) COLN(10) COLN(11) COLN(12) COLN(13) COLN(14) COLN(15)
            #undef COLN
            float* yb = y + b * NB;
            vst1q_f32(yb, a0); vst1q_f32(yb+4, a1);
            vst1q_f32(yb+8, a2); vst1q_f32(yb+12, a3);
        }
    }
};

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

//the output of each call feeds the next, a serial dependency chain,
//exactly as inside a running reverberator. best of seven runs.
template <typename F>
static double bench_ns(F&& f, float* x, float* y, int iters) {
    for (int i = 0; i < 20000; ++i) { f(x, y); float* t = x; x = y; y = t; }
    double best = 1e30;
    for (int rep = 0; rep < 7; ++rep) {
        double t0 = now_s();
        for (int i = 0; i < iters; ++i) { f(x, y); float* t = x; x = y; y = t; }
        double dt = (now_s() - t0) / iters * 1e9;
        if (dt < best) best = dt;
    }
    return best;
}

int main() {
    srand(42);
    alignas(16) static float x[N], y[N];
    for (int i = 0; i < N; ++i) x[i] = (float)((double)rand() / RAND_MAX) - 0.5f;

    //the production staged kernel, angles arbitrary since timing
    //does not depend on their values
    {
        static sfdn::StagedOrtho<N, 8> st;
        float angles[8 * N / 2];
        for (int i = 0; i < 8 * N / 2; ++i) angles[i] = (float)((double)rand() / RAND_MAX) * 6.28f;
        st.setAngles(angles);
        double ns = bench_ns([&](const float* a, float* b){ st.forward(a, b); }, x, y, 2000000);
        printf("staged  b=2  l=8: %7.1f ns  -> %5.1f gflop/s (%d flops)\n",
               ns, st.flops() / ns, st.flops());
    }

    //each monarch block starts as the identity and is stirred with
    //random givens rotations, orthogonal by construction
    {
        static Monarch mo;
        for (int l = 0; l < 2; ++l)
            for (int b = 0; b < BLOCKS; ++b) {
                float (*Wb)[NB] = mo.W[l][b];
                for (int i = 0; i < NB; ++i) for (int j = 0; j < NB; ++j) Wb[j][i] = (i == j);
                for (int g = 0; g < 200; ++g) {
                    int p = rand() % NB, q = rand() % NB; if (p == q) continue;
                    float th = (float)((double)rand() / RAND_MAX) * 6.28f, c = cosf(th), s = sinf(th);
                    for (int j = 0; j < NB; ++j) {
                        float vp = Wb[j][p], vq = Wb[j][q];
                        Wb[j][p] = c * vp - s * vq; Wb[j][q] = s * vp + c * vq;
                    }
                }
            }
        //orthogonality of one block, checked directly
        float err = 0;
        for (int i = 0; i < NB; ++i) for (int j = 0; j < NB; ++j) {
            float d = 0;
            for (int k = 0; k < NB; ++k) d += mo.W[0][3][k][i] * mo.W[0][3][k][j];
            err += fabsf(d - (i == j));
        }
        const int flops = 2 * BLOCKS * (2 * NB * NB);
        double ns = bench_ns([&](const float* a, float* b){ mo.forward(a, b); }, x, y, 2000000);
        printf("monarch b=16 l=2: %7.1f ns  -> %5.1f gflop/s (%d flops)  [block ortho err %.2e]\n",
               ns, flops / ns, flops, err);
    }

    //the sink defeats dead-code elimination
    printf("sink %f %f\n", x[0], y[0]);
    return 0;
}
