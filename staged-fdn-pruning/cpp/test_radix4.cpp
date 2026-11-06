//test_radix4.cpp, validate Radix4Ortho against a scalar reference
//self-contained, blocks get generated in-test with seeded givens
//rotations, the reference implements the layer definition directly
//
//facundo franchino, 2026

#include "radix4_fdn.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

//reference forward straight from the definition:
//out[4k+r] = sum_j blocks[l][k][r][j] * in[k + (N/4) j]
static void reference_forward(const float* blocks, int N, int L,
                              const float* x, float* y) {
    std::vector<double> src(x, x + N), dst(N);
    const int Q = N / 4;
    for (int l = 0; l < L; ++l) {
        for (int k = 0; k < Q; ++k)
            for (int r = 0; r < 4; ++r) {
                double acc = 0.0;
                for (int j = 0; j < 4; ++j)
                    acc += blocks[((l * Q + k) * 4 + r) * 4 + j] * src[k + Q * j];
                dst[4 * k + r] = acc;
            }
        src = dst;
    }
    for (int i = 0; i < N; ++i) y[i] = (float)src[i];
}

static void make_blocks(float* blocks, int N, int L) {
    const int Q = N / 4;
    for (int l = 0; l < L; ++l)
        for (int k = 0; k < Q; ++k) {
            float W[4][4];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) W[i][j] = (i == j);
            //stir with random givens rotations, orthogonal by construction
            for (int g = 0; g < 40; ++g) {
                int p = rand() % 4, q = rand() % 4; if (p == q) continue;
                double th = (double)rand() / RAND_MAX * 6.28;
                float c = cosf((float)th), s = sinf((float)th);
                for (int j = 0; j < 4; ++j) {
                    float vp = W[p][j], vq = W[q][j];
                    W[p][j] = c * vp - s * vq; W[q][j] = s * vp + c * vq;
                }
            }
            for (int r = 0; r < 4; ++r)
                for (int j = 0; j < 4; ++j)
                    blocks[((l * Q + k) * 4 + r) * 4 + j] = W[r][j];
        }
}

template <int N, int L>
static int run_test() {
    const int Q = N / 4;
    std::vector<float> blocks(L * Q * 16);
    make_blocks(blocks.data(), N, L);

    sfdn::Radix4Ortho<N, L> r4;
    r4.setBlocks(blocks.data());

    int fail = 0;
    for (int t = 0; t < 32; ++t) {
        alignas(16) float x[N], y[N], yr[N];
        for (int i = 0; i < N; ++i) x[i] = (float)((double)rand() / RAND_MAX) - 0.5f;

        r4.forward(x, y);
        reference_forward(blocks.data(), N, L, x, yr);

        float max_err = 0.f;
        for (int i = 0; i < N; ++i) {
            float e = std::fabs(y[i] - yr[i]);
            if (e > max_err) max_err = e;
        }

        //orthogonal layers preserve energy
        double e_in = 0, e_out = 0;
        for (int i = 0; i < N; ++i) { e_in += x[i] * (double)x[i]; e_out += y[i] * (double)y[i]; }
        float e_err = (float)std::fabs(e_in - e_out) / (float)e_in;

        if (max_err > 1e-4f || e_err > 1e-4f) {
            fail++;
            std::printf("  FAIL N=%d L=%d test %d: max_err=%.3e energy_err=%.3e\n",
                        N, L, t, max_err, e_err);
        }
    }
    std::printf("  N=%3d L=%d: %d/32 passed\n", N, L, 32 - fail);
    return fail;
}

int main() {
    srand(42);
    int fail = 0;
    fail += run_test<16, 2>();
    fail += run_test<16, 3>();
    fail += run_test<16, 4>();
    fail += run_test<32, 2>();
    fail += run_test<32, 3>();
    fail += run_test<32, 4>();
    fail += run_test<64, 2>();
    fail += run_test<64, 3>();
    fail += run_test<64, 4>();
    fail += run_test<128, 2>();
    fail += run_test<128, 3>();
    fail += run_test<128, 4>();
    fail += run_test<256, 2>();
    fail += run_test<256, 4>();
    std::printf("\n%s\n", fail == 0 ? "ALL PASSED" : "SOME TESTS FAILED");
    return fail ? 1 : 0;
}
