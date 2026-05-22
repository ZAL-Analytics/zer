// em_reduce.slang, M-step reduction for the Fellegi-Sunter EM algorithm.
// Port of em_reduce.cu to Slang/SPIR-V. Three entry points compiled separately:
//   em_estep         , E-step: P(match | vector) for every pair
//   em_mstep_partial , M-step pass 1: per-block partial reduction
//   em_mstep_final   , M-step pass 2: combine partials into final m/u counts
//
// Data layout (identical to em_reduce.cu):
//   comparison_levels[field * n_pairs + pair]  , field-major (coalesced reads)
//   match_probs[pair]                          , pair-major
//   m_partials[cell * num_blocks + block]      , cell-major (coalesced pass-2 reads)

static const uint BLOCK_SIZE    = 256u;
static const uint MAX_FIELDS    = 16u;
static const uint NUM_LEVELS    = 4u;
static const uint SHARED_SZ     = MAX_FIELDS * NUM_LEVELS;     // 64 cells
static const uint SHARED_SZ_PAD = SHARED_SZ + 1u;             // 65, avoids bank conflicts
static const uint WARP_SIZE     = 32u;
static const uint NUM_WARPS     = BLOCK_SIZE / WARP_SIZE;      // 8

// --- E-step ---

[[vk::binding(0, 0)]] StructuredBuffer<uint>    g_estep_levels;    // comparison_levels
[[vk::binding(1, 0)]] StructuredBuffer<float>   g_estep_weights;   // n_fields * NUM_LEVELS floats
[[vk::binding(2, 0)]] RWStructuredBuffer<float> g_estep_probs;     // output match_probs

struct EstepPush { uint n_pairs; uint n_fields; float log_prior_odds; };
[[vk::push_constant]] EstepPush g_estep_push;

[numthreads(BLOCK_SIZE, 1, 1)]
void em_estep(uint3 globalID : SV_DispatchThreadID)
{
    uint p = globalID.x;
    if (p >= g_estep_push.n_pairs) return;

    float log_odds = g_estep_push.log_prior_odds;
    uint  n_pairs  = g_estep_push.n_pairs;
    uint  n_fields = g_estep_push.n_fields;

    for (uint f = 0u; f < n_fields; f++) {
        uint level = g_estep_levels[f * n_pairs + p];
        log_odds  += g_estep_weights[f * NUM_LEVELS + level];
    }

    g_estep_probs[p] = 1.0f / (1.0f + exp(-log_odds));
}

// --- M-step pass 1: per-block partial reduction ---

[[vk::binding(0, 0)]] StructuredBuffer<float>   g_partial_probs;      // match_probs
[[vk::binding(1, 0)]] StructuredBuffer<uint>    g_partial_levels;     // comparison_levels
[[vk::binding(2, 0)]] RWStructuredBuffer<float> g_m_partials;         // cell-major output
[[vk::binding(3, 0)]] RWStructuredBuffer<float> g_u_partials;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> g_match_totals;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> g_nonmatch_totals;

struct MstepPush { uint n_pairs; uint n_fields; uint num_blocks; };
[[vk::push_constant]] MstepPush g_mstep_push;

// Per-warp private accumulators; padded to SHARED_SZ_PAD to avoid bank conflicts.
groupshared float s_m_warp[NUM_WARPS][SHARED_SZ_PAD];
groupshared float s_u_warp[NUM_WARPS][SHARED_SZ_PAD];
groupshared float s_match_warp[NUM_WARPS];
groupshared float s_nonmatch_warp[NUM_WARPS];

// Warp-wide float sum via subgroup shuffle; lane 0 holds the total.
float warp_reduce_sum(float v) {
    v += WaveReadLaneAt(v, WaveGetLaneIndex() ^ 16u);
    v += WaveReadLaneAt(v, WaveGetLaneIndex() ^  8u);
    v += WaveReadLaneAt(v, WaveGetLaneIndex() ^  4u);
    v += WaveReadLaneAt(v, WaveGetLaneIndex() ^  2u);
    v += WaveReadLaneAt(v, WaveGetLaneIndex() ^  1u);
    return v;
}

[numthreads(BLOCK_SIZE, 1, 1)]
void em_mstep_partial(uint3 globalID : SV_DispatchThreadID,
                       uint3 localID  : SV_GroupThreadID,
                       uint3 groupID  : SV_GroupID)
{
    uint tid     = localID.x;
    uint pair_id = globalID.x;
    uint lane    = WaveGetLaneIndex();
    uint wid     = tid / WARP_SIZE;

    uint n_pairs    = g_mstep_push.n_pairs;
    uint n_fields   = g_mstep_push.n_fields;
    uint num_blocks = g_mstep_push.num_blocks;

    // Each warp zeroes its own slice; no cross-warp sync needed yet.
    for (uint i = lane; i < SHARED_SZ; i += WARP_SIZE) {
        s_m_warp[wid][i] = 0.0f;
        s_u_warp[wid][i] = 0.0f;
    }
    if (lane == 0u) { s_match_warp[wid] = 0.0f; s_nonmatch_warp[wid] = 0.0f; }
    GroupMemoryBarrierWithGroupSync();

    // Inactive threads contribute zero; avoids divergent branches below.
    bool  active = (pair_id < n_pairs);
    float p_val  = active ? g_partial_probs[pair_id] : 0.0f;
    float q_val  = active ? (1.0f - p_val) : 0.0f;

    float warp_p = warp_reduce_sum(p_val);
    float warp_q = warp_reduce_sum(q_val);
    if (lane == 0u) { s_match_warp[wid] = warp_p; s_nonmatch_warp[wid] = warp_q; }

    // Ternary for inactive/wrong-level lanes avoids divergence; lane 0 stores, no atomics.
    for (uint f = 0u; f < n_fields; f++) {
        uint my_level = active ? g_partial_levels[f * n_pairs + pair_id] : NUM_LEVELS;
        for (uint l = 0u; l < NUM_LEVELS; l++) {
            float mp    = (my_level == l) ? p_val : 0.0f;
            float mq    = (my_level == l) ? q_val : 0.0f;
            float sum_p = warp_reduce_sum(mp);
            float sum_q = warp_reduce_sum(mq);
            if (lane == 0u) {
                s_m_warp[wid][f * NUM_LEVELS + l] = sum_p;
                s_u_warp[wid][f * NUM_LEVELS + l] = sum_q;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Inter-warp merge + flush to global in cell-major layout for coalesced pass-2 reads.
    uint block_idx = groupID.x;
    for (uint i = tid; i < n_fields * NUM_LEVELS; i += BLOCK_SIZE) {
        float sm = 0.0f, su = 0.0f;
        for (uint w = 0u; w < NUM_WARPS; w++) { sm += s_m_warp[w][i]; su += s_u_warp[w][i]; }
        g_m_partials[i * num_blocks + block_idx] = sm;
        g_u_partials[i * num_blocks + block_idx] = su;
    }

    if (tid == 0u) {
        float tm = 0.0f, tnm = 0.0f;
        for (uint w = 0u; w < NUM_WARPS; w++) { tm += s_match_warp[w]; tnm += s_nonmatch_warp[w]; }
        g_match_totals[block_idx]    = tm;
        g_nonmatch_totals[block_idx] = tnm;
    }
}

// --- M-step pass 2: combine block partials into final counts ---

[[vk::binding(0, 0)]] StructuredBuffer<float>   g_final_m_part;
[[vk::binding(1, 0)]] StructuredBuffer<float>   g_final_u_part;
[[vk::binding(2, 0)]] StructuredBuffer<float>   g_final_match_tot;
[[vk::binding(3, 0)]] StructuredBuffer<float>   g_final_nonmatch_tot;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> g_m_out;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> g_u_out;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> g_total_match;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> g_total_nonmatch;

struct FinalPush { uint num_blocks; uint num_cells; };
[[vk::push_constant]] FinalPush g_final_push;

groupshared float s_smem[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void em_mstep_final(uint3 globalID : SV_DispatchThreadID,
                     uint3 localID  : SV_GroupThreadID,
                     uint3 groupID  : SV_GroupID)
{
    uint cell       = groupID.x;
    uint tid        = localID.x;
    uint num_blocks = g_final_push.num_blocks;
    uint num_cells  = g_final_push.num_cells;

    if (cell < num_cells) {
        float acc_m = 0.0f, acc_u = 0.0f;
        for (uint b = tid; b < num_blocks; b += BLOCK_SIZE) {
            acc_m += g_final_m_part[cell * num_blocks + b];
            acc_u += g_final_u_part[cell * num_blocks + b];
        }

        s_smem[tid] = acc_m;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = BLOCK_SIZE >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) s_smem[tid] += s_smem[tid + stride];
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid == 0u) g_m_out[cell] = s_smem[0];

        // Reuse smem for u; acc_u is still in registers, no hazard.
        s_smem[tid] = acc_u;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = BLOCK_SIZE >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) s_smem[tid] += s_smem[tid + stride];
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid == 0u) g_u_out[cell] = s_smem[0];

    } else {
        // cell == num_cells → match totals; cell == num_cells+1 → nonmatch totals.
        bool is_match = (cell == num_cells);

        float acc = 0.0f;
        for (uint b = tid; b < num_blocks; b += BLOCK_SIZE) {
            acc += is_match ? g_final_match_tot[b] : g_final_nonmatch_tot[b];
        }

        s_smem[tid] = acc;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = BLOCK_SIZE >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) s_smem[tid] += s_smem[tid + stride];
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid == 0u) {
            if (is_match) g_total_match[0] = s_smem[0];
            else          g_total_nonmatch[0] = s_smem[0];
        }
    }
}
