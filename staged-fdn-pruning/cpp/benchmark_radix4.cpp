//staged b=2 (thesis layer counts) against radix-4
//(listening-validated layer counts) and apple BLAS, on the exact same
//same harness as benchmark.cpp, per-call timing, median of 50000 trials
//
//facundo franchino, 2026

#include "staged_fdn.h"
#include "radix4_fdn.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
    #define HAS_BLAS 1
#endif

static inline double now_ns() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::nano>(
        clk::now().time_since_epoch()).count();
}

static constexpr int kWarmup = 2000;
static constexpr int kBlock  = 512;    //calls per timed block
static constexpr int kBlocks = 400;    //blocks, reporting the median

//the clock on apple silicon ticks at 24 MHz, a granularity of 41.7 ns,
//so timing single calls quantises to multiples of it and sub-40 ns
//kernels read as zero. the harness keeps the independent-call
//semantics of benchmark.cpp (calls may overlap in the pipeline, as
//successive fdn samples do once the delay lines decouple them) but
//times blocks of calls and divides, restoring resolution.
template <typename Func>
static double bench(Func&& fn) {
    float dummy = 0.0f;
    for (int i = 0; i < kWarmup; ++i)
        dummy += fn();
    volatile float sink = dummy;
    (void)sink;

    std::vector<double> times(kBlocks);
    for (int b = 0; b < kBlocks; ++b) {
        double t0 = now_ns();
        for (int i = 0; i < kBlock; ++i) {
            dummy += fn();
            //memory clobber so the call gets re-executed every
            //iteration, the cpu may still overlap successive calls
            asm volatile("" ::: "memory");
        }
        double t1 = now_ns();
        times[b] = (t1 - t0) / kBlock;
    }
    sink = dummy;

    std::sort(times.begin(), times.end());
    return times[kBlocks / 2];
}

//the n=4 radix-4 case. a single dense 4x4 orthogonal block is the
//whole matrix, exact by construction rather than a projection
struct Dense4 {
    alignas(16) float W[4][4];  //column-major
    inline void forward(const float* __restrict__ x,
                        float* __restrict__ y) const noexcept {
#if defined(__aarch64__)
        float32x4_t xv = vld1q_f32(x);
        float32x4_t acc = vmulq_laneq_f32(vld1q_f32(W[0]), xv, 0);
        acc = vfmaq_laneq_f32(acc, vld1q_f32(W[1]), xv, 1);
        acc = vfmaq_laneq_f32(acc, vld1q_f32(W[2]), xv, 2);
        acc = vfmaq_laneq_f32(acc, vld1q_f32(W[3]), xv, 3);
        vst1q_f32(y, acc);
#else
        for (int r = 0; r < 4; ++r) {
            float acc = 0.f;
            for (int j = 0; j < 4; ++j) acc += W[j][r] * x[j];
            y[r] = acc;
        }
#endif
    }
};

template <int N>
static double bench_blas(std::mt19937& rng, float* x, float* y) {
#if HAS_BLAS
    static float A_col[N * N];
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) A_col[i] = dist(rng);
    return bench([&]() -> float {
        cblas_sgemv(CblasColMajor, CblasNoTrans, N, N,
                    1.0f, A_col, N, x, 1, 0.0f, y, 1);
        return y[0];
    });
#else
    return 0.0;
#endif
}

template <int N, int LS, int LR>
static void bench_row() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 6.2831853f);
    std::normal_distribution<float> input_dist(0.0f, 1.0f);

    static sfdn::StagedOrtho<N, LS> staged;
    {
        float angles[LS * (N / 2)];
        for (int i = 0; i < LS * (N / 2); ++i) angles[i] = angle_dist(rng);
        staged.setAngles(angles);
    }

    static sfdn::Radix4Ortho<N, LR> r4;
    {
        static float blocks[LR * (N / 4) * 16];
        for (int i = 0; i < LR * (N / 4) * 16; ++i)
            blocks[i] = input_dist(rng);   //timing is value-independent
        r4.setBlocks(blocks);
    }

    alignas(16) float x[N], y[N];
    for (int i = 0; i < N; ++i) x[i] = input_dist(rng);

    double staged_ns = bench([&]() -> float { staged.forward(x, y); return y[0]; });
    double r4_ns     = bench([&]() -> float { r4.forward(x, y);     return y[0]; });
    double blas_ns   = bench_blas<N>(rng, x, y);

    std::printf("N=%3d | staged L=%2d: %6.1f ns (%5.1fx) | radix4 L=%d: %6.1f ns (%5.1fx) "
                "| BLAS: %7.1f ns | r4/staged %.2fx\n",
                N, LS, staged_ns, blas_ns / staged_ns,
                LR, r4_ns, blas_ns / r4_ns,
                blas_ns, staged_ns / r4_ns);
}

//n=4 and n=8 rows. radix-4 covers n=4 exactly with one dense block;
//n=8 is not representable at b=4 and keeps the b=2 structure
static void bench_row_4() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 6.2831853f);
    std::normal_distribution<float> input_dist(0.0f, 1.0f);

    static sfdn::StagedOrtho<4, 8> staged;
    float angles[8 * 2];
    for (int i = 0; i < 16; ++i) angles[i] = angle_dist(rng);
    staged.setAngles(angles);

    static Dense4 d4;
    for (int j = 0; j < 4; ++j)
        for (int r = 0; r < 4; ++r) d4.W[j][r] = input_dist(rng);

    alignas(16) float x[4], y[4];
    for (int i = 0; i < 4; ++i) x[i] = input_dist(rng);

    double staged_ns = bench([&]() -> float { staged.forward(x, y); return y[0]; });
    double d4_ns     = bench([&]() -> float { d4.forward(x, y);     return y[0]; });
    double blas_ns   = bench_blas<4>(rng, x, y);

    std::printf("N=  4 | staged L= 8: %6.1f ns (%5.1fx) | dense 4x4 : %6.1f ns (%5.1fx) "
                "| BLAS: %7.1f ns | exact, lsd 0\n",
                staged_ns, blas_ns / staged_ns, d4_ns, blas_ns / d4_ns, blas_ns);
}

static void bench_row_8() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 6.2831853f);
    std::normal_distribution<float> input_dist(0.0f, 1.0f);

    static sfdn::StagedOrtho<8, 6> staged;
    float angles[6 * 4];
    for (int i = 0; i < 24; ++i) angles[i] = angle_dist(rng);
    staged.setAngles(angles);

    alignas(16) float x[8], y[8];
    for (int i = 0; i < 8; ++i) x[i] = input_dist(rng);

    double staged_ns = bench([&]() -> float { staged.forward(x, y); return y[0]; });
    double blas_ns   = bench_blas<8>(rng, x, y);

    std::printf("N=  8 | staged L= 6: %6.1f ns (%5.1fx) | radix4    : not representable "
                "| BLAS: %7.1f ns |\n",
                staged_ns, blas_ns / staged_ns, blas_ns);
}

int main() {
    std::printf("Staged b=2 vs Radix-4 Benchmark (benchmark.cpp harness)\n");
    std::printf("=======================================================\n");
    std::printf("Warmup: %d, blocks of %d calls, reporting median block; speedups vs BLAS in parentheses\n\n",
                kWarmup, kBlock);

    bench_row_4();
    bench_row_8();
    bench_row< 16,  4, 2>();
    bench_row< 32, 10, 4>();
    bench_row< 64,  3, 2>();
    bench_row<128,  3, 2>();
    bench_row<256,  3, 2>();

    return 0;
}
