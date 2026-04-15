# Structured Pruning of Feedback Delay Networks




### Two-staged feedback matrix optimisation:

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/340c4969-eeaa-4c6a-81ad-c99da48d311e"
    alt="pipeline_diagram_white"
    width="88%"
  />
</p>




### Some results:
| Matrix Size (N) | Optimal Layers (L) | Staged FDN | Apple BLAS | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **4** | L=8 | 1.5 ns | 25 ns | **17.4×** |
| **8** | L=10 | 16.2 ns | 26 ns | **1.6×** |
| **16** | L=8 | 26.3 ns | 37 ns | **1.4×** |
| **32** | L=10 | 59.3 ns | 70 ns | **1.2×** |
| **64** | L=8 | 65.8 ns | 190 ns | **2.9×** |
| **128** | L=8 | 100.8 ns | 660 ns | **6.5×** |
| **256** | L=8 | 190.4 ns | 1989 ns | **10.4×** |
