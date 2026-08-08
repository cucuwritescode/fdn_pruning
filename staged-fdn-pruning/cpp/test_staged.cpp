//test_staged.cpp, validate C++ StagedOrtho against Python test vectors
//constant-geometry format, single permutation for all layers
//
//author: facundo franchino, 2026

#include "staged_fdn.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <string>

//format: {N, L, n_test, cos[L][N/2], sin[L][N/2], perm[N], inputs[n_test][N], outputs[n_test][N]}

struct TestData {
    uint32_t N, L, n_test;
    float*   cos_tab;
    float*   sin_tab;
    uint8_t* perm;  //single permutation (constant geometry)
    float*   inputs;
    float*   outputs;
};

static bool load_test(const char* path, TestData& td) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;

    std::fread(&td.N, 4, 1, f);
    std::fread(&td.L, 4, 1, f);
    std::fread(&td.n_test, 4, 1, f);

    int blocks = td.N / 2;
    int cs = td.L * blocks;

    td.cos_tab = new float[cs];
    td.sin_tab = new float[cs];
    td.perm    = new uint8_t[td.N];
    td.inputs  = new float[td.n_test * td.N];
    td.outputs = new float[td.n_test * td.N];

    std::fread(td.cos_tab, 4, cs, f);
    std::fread(td.sin_tab, 4, cs, f);
    std::fread(td.perm, 1, td.N, f);
    std::fread(td.inputs, 4, td.n_test * td.N, f);
    std::fread(td.outputs, 4, td.n_test * td.N, f);

    std::fclose(f);
    return true;
}

static void free_test(TestData& td) {
    delete[] td.cos_tab; delete[] td.sin_tab;
    delete[] td.perm; delete[] td.inputs; delete[] td.outputs;
}

template <int N, int L>
static int run_tests(const TestData& td) {
    sfdn::StagedOrtho<N, L> stage;
    stage.setCosSin(td.cos_tab, td.sin_tab);
    //constant geometry, permutation is baked into the structure

    int pass = 0, fail = 0;
    for (uint32_t t = 0; t < td.n_test; ++t) {
        const float* xr = td.inputs + t * N;
        const float* yr = td.outputs + t * N;
        float y[N];
        stage.forward(xr, y);

        float max_err = 0.f;
        for (int i = 0; i < N; ++i) {
            float e = std::fabs(y[i] - yr[i]);
            if (e > max_err) max_err = e;
        }

        if (max_err < 1e-4f) {
            pass++;
        } else {
            fail++;
            std::printf("  FAIL test %u: max_err = %.6e\n", t, max_err);
            for (int i = 0; i < N; ++i)
                if (std::fabs(y[i] - yr[i]) > 1e-4f) {
                    std::printf("    [%d] got %.8f expected %.8f\n", i, y[i], yr[i]);
                    break;
                }
        }
    }
    std::printf("  %d/%u passed\n", pass, td.n_test);
    return fail;
}

int main(int argc, char** argv) {
    const char* dir = "../../experiments/cpp_test_vectors";
    if (argc > 1) dir = argv[1];

    struct Cfg { const char* file; int N; int L; };
    Cfg cfgs[] = {
        {"test_N8_splithalf.bin",   8,  3},
        {"test_N16_splithalf.bin",  16, 4},
        {"test_N32_splithalf.bin",  32, 5},
        {"test_N64_splithalf.bin",  64, 6},
        {"test_N128_splithalf.bin", 128, 7},
    };

    int total_fail = 0;
    for (auto& cfg : cfgs) {
        std::string path = std::string(dir) + "/" + cfg.file;
        TestData td;
        if (!load_test(path.c_str(), td)) {
            std::printf("[SKIP] %s\n", cfg.file);
            continue;
        }
        if ((int)td.N != cfg.N || (int)td.L != cfg.L) {
            std::printf("[SKIP] %s: header mismatch\n", cfg.file);
            free_test(td); continue;
        }

        std::printf("[TEST] %s (N=%d, L=%d, %u cases)\n", cfg.file, cfg.N, cfg.L, td.n_test);
        int fail = 0;
        switch (cfg.N * 100 + cfg.L) {
            case   803: fail = run_tests<  8,3>(td); break;
            case  1604: fail = run_tests< 16,4>(td); break;
            case  3205: fail = run_tests< 32,5>(td); break;
            case  6406: fail = run_tests< 64,6>(td); break;
            case 12807: fail = run_tests<128,7>(td); break;
            case 25608: fail = run_tests<256,8>(td); break;
        }
        total_fail += fail;
        free_test(td);
    }

    std::printf("\n%s\n", total_fail == 0 ? "ALL PASSED" : "SOME TESTS FAILED");
    return total_fail ? 1 : 0;
}
