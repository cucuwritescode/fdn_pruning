//peak single-core fp32 throughput, measured rather than quoted.
//sixteen independent fma chains cover the pipeline latency of the
//four 128-bit vector units, so the loop issues at machine rate and
//the figure is the ceiling of the roofline. eight chains are not
//enough and under-read the peak by a factor of three.
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
static double now_s(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(){
    float32x4_t a0=vdupq_n_f32(1.0f),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0,
                a8=a0,a9=a0,a10=a0,a11=a0,a12=a0,a13=a0,a14=a0,a15=a0;
    float32x4_t m=vdupq_n_f32(0.9999999f), c=vdupq_n_f32(1e-9f);
    const long iters=50000000L;
    double best=0;
    for(int rep=0;rep<5;++rep){
        double t0=now_s();
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
        double g = iters*16.0*4*2/(now_s()-t0)/1e9;
        if(g>best) best=g;
    }
    //the sink defeats dead-code elimination
    float s=vgetq_lane_f32(a0,0)+vgetq_lane_f32(a8,1);
    printf("peak fp32 (1 core, 16 chains): %.1f gflop/s (sink %f)\n", best, s);
    return 0;
}
