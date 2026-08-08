//benchmark.cpp: staged vs dense (naive + BLAS) matvec benchmark
//
//author: facundo franchino, 2026

#include "staged_fdn.h"
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

template <int N>
struct DenseMatvec {
    alignas(16) float A[N][N];

    void randomise(std::mt19937& rng) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                A[i][j] = dist(rng);
    }

    inline void forward(const float* __restrict__ x,
                        float* __restrict__ y) const noexcept {
        for (int i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j)
                sum += A[i][j] * x[j];
            y[i] = sum;
        }
    }
};

static constexpr int kWarmup = 2000;
static constexpr int kTrials = 50000;

template <typename Func>
static double bench(Func&& fn) {
    float dummy = 0.0f;
    for (int i = 0; i < kWarmup; ++i)
        dummy += fn();
    volatile float sink = dummy;
    (void)sink;

    std::vector<double> times(kTrials);
    for (int i = 0; i < kTrials; ++i) {
        double t0 = now_ns();
        dummy += fn();
        double t1 = now_ns();
        times[i] = t1 - t0;
    }
    sink = dummy;

    std::sort(times.begin(), times.end());
    return times[kTrials / 2];
}

template <int N, int L>
static void bench_one() {
    using pidx = typename sfdn::StagedOrtho<N, L>::perm_idx_t;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 6.2831853f);
    std::normal_distribution<float> input_dist(0.0f, 1.0f);

    //staged setup
    sfdn::StagedOrtho<N, L> staged;
    {
        float angles[L * (N / 2)];
        for (int i = 0; i < L * (N / 2); ++i)
            angles[i] = angle_dist(rng);
        staged.setAngles(angles);

        for (int l = 0; l < L; ++l) {
            pidx perm[N];
            const int half = N / 2;
            for (int i = 0; i < half; ++i) {
                perm[2 * i]     = static_cast<pidx>(i);
                perm[2 * i + 1] = static_cast<pidx>(half + i);
            }
            staged.setPermutation(l, perm);
        }
    }

    //dense setup
    DenseMatvec<N> dense;
    dense.randomise(rng);

    alignas(16) float x[N], y[N];
    for (int i = 0; i < N; ++i)
        x[i] = input_dist(rng);

    double staged_ns = bench([&]() -> float {
        staged.forward(x, y);
        return y[0];
    });

    double naive_ns = bench([&]() -> float {
        dense.forward(x, y);
        return y[0];
    });

#if HAS_BLAS
    alignas(16) float A_col[N * N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_col[j * N + i] = dense.A[i][j];

    double blas_ns = bench([&]() -> float {
        cblas_sgemv(CblasColMajor, CblasNoTrans, N, N,
                    1.0f, A_col, N, x, 1, 0.0f, y, 1);
        return y[0];
    });

    std::printf("N=%3d  L=%d  | staged: %6.1f ns  naive: %7.1f ns (%5.2fx)  "
                "BLAS: %7.1f ns (%5.2fx)  | FLOPs: %5d vs %6d (%5.1fx)\n",
                N, L, staged_ns,
                naive_ns, naive_ns / staged_ns,
                blas_ns,  blas_ns / staged_ns,
                staged.flops(), 2 * N * N - N,
                static_cast<double>(2 * N * N - N) / staged.flops());
#else
    std::printf("N=%3d  L=%d  | staged: %6.1f ns  naive: %7.1f ns (%5.2fx)  "
                "| FLOPs: %5d vs %6d (%5.1fx)\n",
                N, L, staged_ns,
                naive_ns, naive_ns / staged_ns,
                staged.flops(), 2 * N * N - N,
                static_cast<double>(2 * N * N - N) / staged.flops());
#endif
}

int main() {
    std::printf("Staged FDN Benchmark\n");
    std::printf("====================\n");
    std::printf("Warmup: %d, Trials: %d, reporting median\n\n", kWarmup, kTrials);

    bench_one<  8, 3>();
    bench_one< 16, 4>();
    bench_one< 32, 5>();
    bench_one< 32,10>();
    bench_one< 32,12>();
    bench_one< 64, 6>();
    bench_one< 64,12>();
    bench_one<128, 7>();
    bench_one<128,12>();
    bench_one<256, 8>();

#if HAS_BLAS
    std::printf("\nBaselines: naive = compiler auto-vectorised loop, BLAS = Apple Accelerate cblas_sgemv\n");
    std::printf("Speedup columns: staged_time / baseline_time (>1 = staged wins)\n");
#endif

    return 0;
}
