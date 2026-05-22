// em_reduce.cu, M-step reduction for the Fellegi-Sunter EM algorithm.
//
// Two-pass strategy:
//   Pass 1 (em_reduce_kernel):       each block reduces BLOCK_SIZE pairs into
//                                    cell-major partial sums.
//   Pass 2 (em_reduce_final_kernel): grid = (num_cells + 2), one block per cell,
//                                    combines all partial sums.
//
// Per-warp private smem avoids atomicAdd in pass 1. Cell-major partial layout
// makes pass-2 reads coalesced. Two extra blocks handle match/nonmatch totals.
//
// Data layout:
//   comparison_levels[field * n_pairs + pair]  , field-major (coalesced reads)
//   match_probs[pair]                          , pair-major  (coalesced reads)
//   m_partials[cell * num_blocks + block]      , cell-major  (coalesced pass-2 reads)

#include <stdint.h>

#define BLOCK_SIZE        256
#define MAX_FIELDS         32
#define NUM_LEVELS          4
#define SHARED_SZ        (MAX_FIELDS * NUM_LEVELS)       // 128 floats = 512 bytes
// +1 padding per warp row shifts each row's bank-0 element to a different bank,
// eliminating the 8-way bank conflict that arises because SHARED_SZ=128 is a
// multiple of the 32-bank count.
#define SHARED_SZ_PADDED (MAX_FIELDS * NUM_LEVELS + 1)  // 129 floats = 516 bytes
#define WARP_SIZE          32
#define NUM_WARPS        (BLOCK_SIZE / WARP_SIZE)        // 8

static __device__ __forceinline__ uint32_t lane_id() { return threadIdx.x & 31u; }
static __device__ __forceinline__ uint32_t warp_id() { return threadIdx.x >> 5u;  }

// Warp-wide float sum via shuffle-down; lane 0 holds the total.
static __device__ __forceinline__ float warp_reduce_sum(float v)
{
    v += __shfl_down_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_down_sync(0xFFFFFFFFu, v,  8);
    v += __shfl_down_sync(0xFFFFFFFFu, v,  4);
    v += __shfl_down_sync(0xFFFFFFFFu, v,  2);
    v += __shfl_down_sync(0xFFFFFFFFu, v,  1);
    return v;
}

// --- E-step: P(match | vector) for each pair ---
// One thread per pair; writes sigmoid(log_prior_odds + sum of weights[f][level_f]).

extern "C" __global__ void em_estep_kernel(
    const uint32_t* __restrict__ comparison_levels,
    const float*    __restrict__ weights,
    float*          __restrict__ match_probs,
    float                        log_prior_odds,
    uint32_t                     n_pairs,
    uint32_t                     n_fields
)
{
    const uint32_t p = (uint32_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (p >= n_pairs) return;

    float log_odds = log_prior_odds;
    for (uint32_t f = 0; f < n_fields; ++f) {
        const uint32_t level = comparison_levels[f * n_pairs + p];
        log_odds += weights[f * NUM_LEVELS + level];
    }

    // fast-math exp is fine for EM convergence
    match_probs[p] = 1.0f / (1.0f + __expf(-log_odds));
}

// --- Pass 1: per-block partial reduction ---
// Each warp accumulates into its own smem slice; lane 0 stores the warp total
// with a plain write (no atomicAdd, no MIO stall). After __syncthreads, the
// inter-warp merge writes to global in cell-major order so pass-2 reads are coalesced.

extern "C" __global__ void em_reduce_kernel(
    const float*    __restrict__ match_probs,
    const uint32_t* __restrict__ comparison_levels,
    float*          __restrict__ m_partials,
    float*          __restrict__ u_partials,
    float*          __restrict__ match_totals,
    float*          __restrict__ nonmatch_totals,
    uint32_t                     n_pairs,
    uint32_t                     num_fields,
    uint32_t                     num_blocks
)
{
    // s_m_warp[wid][cell], padded to SHARED_SZ_PADDED (129) to avoid bank conflicts.
    // Total smem: 2 * 8 * 129 * 4 = 8,256 bytes, well within sm_86 limits.
    __shared__ float s_m_warp[NUM_WARPS][SHARED_SZ_PADDED];
    __shared__ float s_u_warp[NUM_WARPS][SHARED_SZ_PADDED];
    __shared__ float s_match_warp[NUM_WARPS];
    __shared__ float s_nonmatch_warp[NUM_WARPS];

    const uint32_t tid     = threadIdx.x;
    const uint32_t pair_id = (uint32_t)blockIdx.x * BLOCK_SIZE + tid;
    const uint32_t lane    = lane_id();
    const uint32_t wid     = warp_id();

    // Each warp zeroes its own slice; no cross-warp sync needed yet.
    for (uint32_t i = lane; i < SHARED_SZ; i += WARP_SIZE) {
        s_m_warp[wid][i] = 0.0f;
        s_u_warp[wid][i] = 0.0f;
    }
    if (lane == 0) {
        s_match_warp[wid]    = 0.0f;
        s_nonmatch_warp[wid] = 0.0f;
    }
    __syncthreads();

    // Inactive threads contribute zero; avoids divergent branches below.
    const bool  active = (pair_id < n_pairs);
    const float p      = active ? match_probs[pair_id] : 0.0f;
    const float q      = active ? (1.0f - p)           : 0.0f;

    const float warp_p = warp_reduce_sum(p);
    const float warp_q = warp_reduce_sum(q);
    if (lane == 0) {
        s_match_warp[wid]    = warp_p;
        s_nonmatch_warp[wid] = warp_q;
    }

    // Ternary for inactive/wrong-level threads compiles to FSEL, no divergence.
    // Lane 0 stores the warp total with a plain write, no atomicAdd.
    for (uint32_t f = 0; f < num_fields; ++f) {
        const uint32_t my_level = active
            ? comparison_levels[f * n_pairs + pair_id]
            : NUM_LEVELS;   // sentinel: contributes 0 to every level

        for (uint32_t l = 0; l < NUM_LEVELS; ++l) {
            const float mp    = (my_level == l) ? p : 0.0f;
            const float mq    = (my_level == l) ? q : 0.0f;
            const float sum_p = warp_reduce_sum(mp);
            const float sum_q = warp_reduce_sum(mq);
            if (lane == 0) {
                s_m_warp[wid][f * NUM_LEVELS + l] = sum_p;
                s_u_warp[wid][f * NUM_LEVELS + l] = sum_q;
            }
        }
    }
    __syncthreads();

    // Inter-warp merge + global flush; cell-major layout makes pass-2 reads coalesced.
    for (uint32_t i = tid; i < num_fields * NUM_LEVELS; i += BLOCK_SIZE) {
        float sum_m = 0.0f, sum_u = 0.0f;
        for (uint32_t w = 0; w < NUM_WARPS; ++w) {
            sum_m += s_m_warp[w][i];
            sum_u += s_u_warp[w][i];
        }
        m_partials[i * num_blocks + blockIdx.x] = sum_m;
        u_partials[i * num_blocks + blockIdx.x] = sum_u;
    }

    if (tid == 0) {
        float tm = 0.0f, tnm = 0.0f;
        for (uint32_t w = 0; w < NUM_WARPS; ++w) {
            tm  += s_match_warp[w];
            tnm += s_nonmatch_warp[w];
        }
        match_totals[blockIdx.x]    = tm;
        nonmatch_totals[blockIdx.x] = tnm;
    }
}

// --- Pass 2: multi-block final reduction ---
// Grid = (num_cells + 2): one block per cell, two extra for match/nonmatch totals.
// Smem is reused between the m and u tree reductions; acc_u stays in registers.

extern "C" __global__ void em_reduce_final_kernel(
    const float* __restrict__ m_partials,
    const float* __restrict__ u_partials,
    const float* __restrict__ match_totals,
    const float* __restrict__ nonmatch_totals,
    float*       __restrict__ m_out,
    float*       __restrict__ u_out,
    float*       __restrict__ total_match_out,
    float*       __restrict__ total_nonmatch_out,
    uint32_t                  num_blocks,
    uint32_t                  num_cells
)
{
    extern __shared__ float smem[];

    const uint32_t cell = blockIdx.x;
    const uint32_t tid  = threadIdx.x;

    if (cell < num_cells) {
        float acc_m = 0.0f, acc_u = 0.0f;
        for (uint32_t b = tid; b < num_blocks; b += blockDim.x) {
            acc_m += __ldg(&m_partials[cell * num_blocks + b]);
            acc_u += __ldg(&u_partials[cell * num_blocks + b]);
        }

        smem[tid] = acc_m;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }
        if (tid == 0) m_out[cell] = smem[0];

        // Reuse smem for u; acc_u is still live in registers, no hazard.
        smem[tid] = acc_u;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }
        if (tid == 0) u_out[cell] = smem[0];

    } else {
        // cell == num_cells → match totals; cell == num_cells+1 → nonmatch totals.
        const float* src = (cell == num_cells) ? match_totals : nonmatch_totals;

        float acc = 0.0f;
        for (uint32_t b = tid; b < num_blocks; b += blockDim.x)
            acc += __ldg(&src[b]);

        smem[tid] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            if (cell == num_cells) *total_match_out    = smem[0];
            else                   *total_nonmatch_out = smem[0];
        }
    }
}
