#block size against utilisation: a roofline study

the staged factorisation fixes the block size at b=2 and pays for it in
depth: eight layers of paired rotations to reach global mixing at n=256.
one naturally asks whether this was the right exchange. the structured
matrix literature (dao et al., monarch, 2022) argues the opposite corner,
two layers of dense 16x16 blocks, on the grounds that hardware rewards
dense arithmetic far more than it rewards parsimony in flops. these
programs put the question to the machine directly, which seemed
preferable to speculating about it.

##method

all kernels run on the same harness: the output of each matvec feeds the
next input, so successive calls form a serial dependency chain, as they
would inside a running reverberator. times are the best of seven runs of
two million iterations each, after a warm-up. the staged kernel is the
production one, included from ../staged_fdn.h; the other two are written
to the same standard (neon intrinsics, no allocation, weights resident
in l1) but have not been integrated into the audio path.

three structures, each achieving full mixing at n=256:

| file | structure | layers x blocks | mixing argument |
|---|---|---|---|
| bench.cpp | staged, b=2 | 8 x 128 rotations | 2^8 = 256 |
| b4.cpp | butterfly, b=4 | 4 x 64 blocks | 4^4 = 256 |
| bench.cpp | monarch, b=16 | 2 x 16 blocks | 16^2 = 256 |

the radix-4 kernel uses the pease constant-geometry dataflow: every
layer reads lanes i, i+64, i+128, i+192 and writes contiguously through
vst4q, so the permutation costs nothing at all. this is the observation
that gives the staged kernel its free perfect shuffle, one radix higher.
the monarch kernel folds its block transpose into the second layer's
loads for the same reason. peak.cpp establishes the ceiling: sixteen
independent fma chains, enough to cover the pipeline latency of the four
vector units.

##results

apple m2, single performance core, fp32. measured peak: 111.5 gflop/s
(theory: 4 fma pipes x 4 lanes x 2 flops x 3.49 ghz = 111.7, so the
microbenchmark is honest).

| structure | dof | flops | ns/matvec | gflop/s | of peak | rel. trace |
|---|---|---|---|---|---|---|
| staged, b=2, l=8 | 1024 | 6144 | 194.7 | 31.6 | 28% | 0.219 |
| butterfly, b=4, l=4 | 1536 | 8192 | 148.1 | 55.3 | 50% | 0.270 |
| monarch, b=16, l=2 | 3840 | 16384 | 252.3 | 64.9 | 58% | 0.392 |
| dense (apple blas) | 32640 | 131072 | 1989 | 65.9 | 59% | 1.0 |

rel. trace is the best attainable tr(a's)/n against a random orthogonal
target at n=256, each structure fitted by adam (3000 steps) with
cayley-parametrised blocks. a random target is the least favourable
case, and the frobenius trace is a proxy for the perceptual measure; the
figures order the structures but should not be read as more than that.
the same optimiser applied to the staged structure recovers 0.2189
against the projection pipeline's 0.2192, so the two-phase projection
is, within the structure, already extracting essentially all there is.

##the argument

every working set here fits comfortably in l1, so memory bandwidth never
enters into it; the roofline is flat and the whole question reduces to
one equation:

    time = flops / (peak x utilisation)

dense blas attains the best utilisation of the four (59%) and loses by a
factor of ten regardless, because it performs twenty-one times the work.
the staged kernel wins by parsimony but runs at 28% of peak: 2x2
rotations simply do not present enough independent arithmetic to keep
four fma pipes occupied. as the block size grows, utilisation improves
(28, 50, 58%) whilst the flop count grows faster still, and by b=16 the
exchange has turned against us: monarch does 2.7 times the work of the
staged kernel at twice its utilisation, and arrives later for its
trouble.

the product bottoms out in the middle. at b=4 the kernel does 1.33 times
the work at 1.75 times the utilisation, which nets a 1.31x win over the
production kernel, and with half again as many rotation parameters it
approximates orthogonal targets rather better into the bargain. one
might have expected either endpoint of the axis to win; in the event,
neither does.

##reproducing

    clang++ -O3 -std=c++17 -mcpu=native peak.cpp  -o peak  && ./peak
    clang++ -O3 -std=c++17 -mcpu=native bench.cpp -o bench && ./bench
    clang++ -O3 -std=c++17 -mcpu=native b4.cpp   -o b4    && ./b4

each kernel checks its own orthogonality (energy in equals energy out);
bench additionally verifies block orthogonality directly. figures will
vary with the machine, though the ordering ought not to. the staged
measurement here (194.7 ns) reproduces the 190.4 ns reported in the
top-level readme to within run-to-run variation.

##caveats, plainly

the b=4 result is a measured argument, not a finished system. what
remains before it could displace the production kernel: fitting trained
feedback matrices rather than random ones, refining against log-spectral
distance rather than the frobenius trace, and integration into
StagedFDN::processSample. there is no obvious obstacle in any of these,
but none has been done.
