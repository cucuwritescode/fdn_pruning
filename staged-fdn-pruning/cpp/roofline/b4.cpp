//radix-4 constant-geometry butterfly, b=4 l=4, n=256.
//four layers of 64 dense 4x4 orthogonal blocks reach global mixing,
//4^4 = 256, exactly as eight layers of pairs do at radix 2.
//
//pease dataflow: every layer reads lanes i, i+64, i+128, i+192 and
//writes contiguously through vst4q, so the permutation costs nothing.
//the same observation gives staged_fdn.h its free perfect shuffle,
//here one radix higher.
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
static constexpr int N=256, GROUPS=16; //16 groups of 4 blocks per layer
struct Radix4 {
    //wv[layer][group][j][r] holds w[block][r][j] vectorised across
    //the four blocks of a group, one lane per block
    alignas(16) float Wv[4][GROUPS][4][4][4];
    inline void forward(const float* __restrict__ x, float* __restrict__ y) const noexcept {
        alignas(16) float t0[N], t1[N];
        const float* src = x; float* dst = t0;
        for (int l = 0; l < 4; ++l) {
            for (int g = 0; g < GROUPS; ++g) {
                const int i = g * 4;
                float32x4_t xj0 = vld1q_f32(src + i);
                float32x4_t xj1 = vld1q_f32(src + i + 64);
                float32x4_t xj2 = vld1q_f32(src + i + 128);
                float32x4_t xj3 = vld1q_f32(src + i + 192);
                const float (*W)[4][4] = Wv[l][g];
                float32x4x4_t o;
                #define ROW(r) o.val[r] = vfmaq_f32(vfmaq_f32(vfmaq_f32(vmulq_f32(vld1q_f32(W[0][r]), xj0), \
                    vld1q_f32(W[1][r]), xj1), vld1q_f32(W[2][r]), xj2), vld1q_f32(W[3][r]), xj3);
                ROW(0) ROW(1) ROW(2) ROW(3)
                #undef ROW
                //vst4 interleaves the four row vectors, so block b
                //row r lands at position 4*(i+b)+r, the write is the
                //permutation
                vst4q_f32(dst + i * 4, o);
            }
            const float* s = src; src = dst; dst = (dst == t0) ? t1 : t0;
            (void)s;
        }
        for (int i = 0; i < N; ++i) y[i] = src[i];
    }
};
static double now_s(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(){
    srand(42);
    static Radix4 rx;
    //each block starts as the identity and is stirred with random
    //givens rotations, orthogonal by construction
    for(int l=0;l<4;++l) for(int g=0;g<GROUPS;++g) for(int b=0;b<4;++b){
        float W[4][4];
        for(int i=0;i<4;++i) for(int j=0;j<4;++j) W[i][j]=(i==j);
        for(int gv=0; gv<40; ++gv){
            int p=rand()%4,q=rand()%4; if(p==q) continue;
            double th=(double)rand()/RAND_MAX*6.28; float c=cosf(th),s=sinf(th);
            for(int j=0;j<4;++j){ float vp=W[p][j],vq=W[q][j]; W[p][j]=c*vp-s*vq; W[q][j]=s*vp+c*vq; }
        }
        for(int r=0;r<4;++r) for(int j=0;j<4;++j) rx.Wv[l][g][j][r][b]=W[r][j];
    }
    alignas(16) static float x[N],y[N];
    for(int i=0;i<N;++i) x[i]=(double)rand()/RAND_MAX-0.5;
    //energy in equals energy out iff every layer is orthogonal
    double e_in=0,e_out=0; for(int i=0;i<N;++i) e_in+=x[i]*x[i];
    rx.forward(x,y); for(int i=0;i<N;++i) e_out+=y[i]*y[i];
    printf("energy in/out: %.6f / %.6f (orthogonality check)\n", e_in, e_out);
    //serial dependency chain, output feeds the next input, best of seven
    float *px=x,*py=y;
    for(int i=0;i<20000;++i){ rx.forward(px,py); float*t=px;px=py;py=t; }
    double best=1e30;
    for(int rep=0;rep<7;++rep){
        double t0=now_s();
        for(int i=0;i<2000000;++i){ rx.forward(px,py); float*t=px;px=py;py=t; }
        double dt=(now_s()-t0)/2000000*1e9; if(dt<best) best=dt;
    }
    printf("butterfly b=4 l=4: %.1f ns -> %.1f gflop/s (8192 flops)\n", best, 8192/best);
    printf("sink %f\n", px[0]);
    return 0;
}
