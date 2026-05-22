// hello_backend.cu, diagnostic kernel that proves CUDA execution occurred.
// Thread (0,0,0) writes CUDA_TOKEN to out[0]; host verifies the value.
// CUDA is the only backend that produces this token.

#include <stdint.h>

#define CUDA_TOKEN 0xCCDACaFEu  // C CuDA CaFE --> unique per-backend, not exported to Rust

extern "C" __global__ void hello_backend_kernel(uint32_t* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = CUDA_TOKEN;
    }
}
