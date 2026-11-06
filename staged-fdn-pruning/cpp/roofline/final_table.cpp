//the whole table in one binary. p-core canary first, then staged,
//radix-4 and blas (latency + throughput bounds) interleaved per n
#include "../staged_fdn.h"
#include "../radix4_fdn.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

static double now_ns(){ return std::chrono::duration<double,std::nano>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }

//peak fma canary, ~111 gflop/s on an m2 p-core, ~a third on an e-core
static double canary_gflops() {
    float32x4_t a0=vdupq_n_f32(1.f),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0,
                a8=a0,a9=a0,a10=a0,a11=a0,a12=a0,a13=a0,a14=a0,a15=a0;
    float32x4_t m=vdupq_n_f32(0.9999999f), c=vdupq_n_f32(1e-9f);
    const long iters=20000000L;
    double t0=now_ns();
    for(long i=0;i<iters;++i){
        a0=vfmaq_f32(c,a0,m); a1=vfmaq_f32(c,a1,m); a2=vfmaq_f32(c,a2,m); a3=vfmaq_f32(c,a3,m);
        a4=vfmaq_f32(c,a4,m); a5=vfmaq_f32(c,a5,m); a6=vfmaq_f32(c,a6,m); a7=vfmaq_f32(c,a7,m);
        a8=vfmaq_f32(c,a8,m); a9=vfmaq_f32(c,a9,m); a10=vfmaq_f32(c,a10,m); a11=vfmaq_f32(c,a11,m);
        a12=vfmaq_f32(c,a12,m); a13=vfmaq_f32(c,a13,m); a14=vfmaq_f32(c,a14,m); a15=vfmaq_f32(c,a15,m);
            // Keep all sixteen vector chains live and distinct after optimisation.
            asm volatile("" : "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3),
                              "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7));
            asm volatile("" : "+w"(a8), "+w"(a9), "+w"(a10), "+w"(a11),
                              "+w"(a12), "+w"(a13), "+w"(a14), "+w"(a15));
    }
    double dt=(now_ns()-t0)*1e-9;
    volatile float s=vgetq_lane_f32(a0,0)+vgetq_lane_f32(a8,1); (void)s;
    return iters*16.0*4*2/dt/1e9;
}

template <typename Func>
static double bench_thr(Func&& fn) {
    for (int i = 0; i < 2000; ++i) fn();
    std::vector<double> t(200);
    for (int b = 0; b < 200; ++b) {
        double t0 = now_ns();
        for (int i = 0; i < 512; ++i) { fn(); asm volatile("" ::: "memory"); }
        t[b] = (now_ns() - t0) / 512;
    }
    std::sort(t.begin(), t.end());
    return t[100];
}

template <typename Func>
static double bench_lat(Func&& fn) {
    for (int i = 0; i < 2000; ++i) fn();
    std::vector<double> t(50000);
    for (int i = 0; i < 50000; ++i) {
        double t0 = now_ns(); fn(); double t1 = now_ns();
        t[i] = t1 - t0;
    }
    std::sort(t.begin(), t.end());
    return t[25000];
}

template <int N, int LS, int LR>
static void bench_row() {
    static sfdn::StagedOrtho<N, LS> staged;
    static sfdn::Radix4Ortho<N, LR> r4;
    alignas(16) static float A[N*N], x[N], y[N];
    {
        float angles[LS * (N/2)];
        for (int i = 0; i < LS*(N/2); ++i) angles[i] = (double)rand()/RAND_MAX*6.28;
        staged.setAngles(angles);
        static float blocks[LR * (N/4) * 16];
        for (int i = 0; i < LR*(N/4)*16; ++i) blocks[i] = (double)rand()/RAND_MAX-0.5;
        r4.setBlocks(blocks);
        for (int i = 0; i < N*N; ++i) A[i] = (double)rand()/RAND_MAX-0.5;
        for (int i = 0; i < N; ++i) x[i] = (double)rand()/RAND_MAX-0.5;
    }
    auto blas = [&](){ cblas_sgemv(CblasColMajor, CblasNoTrans, N, N, 1.0f, A, N, x, 1, 0.0f, y, 1); };
    double st  = bench_thr([&](){ staged.forward(x, y); });
    double r4t = bench_thr([&](){ r4.forward(x, y); });
    double bt  = bench_thr(blas);
    double bl  = bench_lat(blas);
    printf("N=%3d | staged L=%2d %6.1f ns (%4.1f-%4.1fx) | r4 L=%d %6.1f ns (%4.1f-%4.1fx) | "
           "blas thr %7.1f lat %7.1f\n",
           N, LS, st, bt/st, bl/st, LR, r4t, bt/r4t, bl/r4t, bt, bl);
}

int main() {
    srand(42);
    double g = 0;
    for (int att = 0; att < 20; ++att) {
        g = canary_gflops();
        if (g > 95.0) break;
        fprintf(stderr, "canary %.0f gflop/s, waiting for a high-throughput interval...\n", g);
        usleep(3000000);
    }
    printf("canary: %.1f gflop/s (probe threshold met: %s)\n\n", g, g > 95 ? "yes" : "NO - results suspect");
    bench_row< 16,  4, 2>();
    bench_row< 32, 10, 4>();
    bench_row< 64,  3, 2>();
    bench_row<128,  3, 2>();
    bench_row<256,  3, 2>();
    printf("\nspeedup ranges: vs blas throughput (lower) to blas per-call latency (upper)\n");
    return 0;
}
