// Main task: optimize the MoE forward pass.
#include "moe.h"
#include "moe_opt_common.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <immintrin.h>
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>

using moe_common::fast_exp_ps;
using moe_common::pack_amx_expert_matrices_u8bias;
using moe_common::pack_matrix_amx_kx16_vnni_u8bias;
using moe_common::pack_router_4x16;
using moe_common::router_exp_ps;
using moe_common::compute_gate_sum;
using moe_common::quantize_token;
using moe_common::ffn_requantize_with_amax;
using moe_common::ffn_requantize;

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#ifndef XFEATURE_XTILECFG
#define XFEATURE_XTILECFG 17
#endif

#if defined(__GNUC__) || defined(__clang__)
#define MOE_NOINLINE __attribute__((noinline))
#else
#define MOE_NOINLINE
#endif

namespace {

constexpr int kRowBlock = 4;
constexpr int kRouterColBlock = 16;
constexpr int kAmxTokenBlock = 16;
constexpr int kAmxRowBlock = 16;
constexpr int kAmxKBlock = 64;

int g_pack_d_model = 0;
int g_pack_d_ff = 0;
int g_pack_num_experts = 0;
bool g_pack_valid = false;
alignas(64) float g_batch_s[(size_t)MAX_NUM_TOKENS * MAX_NUM_EXPERTS];
alignas(64) int8_t g_batch_xq[(size_t)MAX_NUM_TOKENS * MAX_D_MODEL];
alignas(64) int g_batch_topk[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
alignas(64) float g_batch_gate_sum[MAX_NUM_TOKENS];
alignas(64) float g_batch_s_x[MAX_NUM_TOKENS];
alignas(64) int32_t g_batch_x_correction[MAX_NUM_TOKENS];
alignas(64) float g_shared_h_batch[(size_t)MAX_NUM_TOKENS * MAX_D_FF];
alignas(64) int8_t g_shared_hq_batch[(size_t)MAX_NUM_TOKENS * MAX_D_FF];
alignas(64) float g_shared_down_scale[MAX_NUM_TOKENS];
alignas(64) int32_t g_shared_h_correction[MAX_NUM_TOKENS];
alignas(64) int g_routed_bucket_count[MAX_NUM_EXPERTS];
alignas(64) int g_routed_bucket_token[(size_t)MAX_NUM_EXPERTS *
                                      MAX_NUM_TOKENS * MAX_TOP_K];
alignas(64) int g_routed_bucket_slot[(size_t)MAX_NUM_EXPERTS *
                                     MAX_NUM_TOKENS * MAX_TOP_K];
alignas(64) float g_routed_out[(size_t)MAX_NUM_TOKENS * MAX_TOP_K *
                               MAX_D_MODEL];
alignas(64) float g_routed_h[(size_t)MAX_NUM_TOKENS * MAX_TOP_K * MAX_D_FF];
alignas(64) int8_t g_routed_hq[(size_t)MAX_NUM_TOKENS * MAX_TOP_K * MAX_D_FF];
alignas(64) float g_routed_down_scale[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
alignas(64) int32_t g_routed_h_correction[(size_t)MAX_NUM_TOKENS * MAX_TOP_K];
alignas(64) int g_routed_task_expert[MAX_NUM_EXPERTS * MAX_NUM_TOKENS];
alignas(64) int g_routed_task_start[MAX_NUM_EXPERTS * MAX_NUM_TOKENS];
alignas(64) int g_routed_task_count[MAX_NUM_EXPERTS * MAX_NUM_TOKENS];

float* g_w_router_pack = nullptr;
int8_t* g_w_gate_amx_pack = nullptr;
int8_t* g_w_up_amx_pack = nullptr;
int8_t* g_w_down_amx_pack = nullptr;
int8_t* g_sh_gate_amx_pack = nullptr;
int8_t* g_sh_up_amx_pack = nullptr;
int8_t* g_sh_down_amx_pack = nullptr;

struct TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

static bool init_amx_for_thread() {
    static thread_local bool attempted = false;
    static thread_local bool ok = false;
    if (!attempted) {
        long cfg_ret =
            syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILECFG);
        long data_ret =
            syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
        ok = data_ret == 0;
        if (std::getenv("MOE_AMX_DEBUG") != nullptr) {
            std::fprintf(stderr,
                         "[moe_amx] arch_prctl cfg=%ld data=%ld ok=%d\n",
                         cfg_ret, data_ret, ok ? 1 : 0);
        }
        attempted = true;
    }
    return ok;
}

static void load_amx_config(int token_cols) {
    alignas(64) TileConfig cfg = {};
    cfg.palette_id = 1;
    cfg.colsb[0] = (uint16_t)(token_cols * (int)sizeof(int32_t));
    cfg.rows[0] = kAmxRowBlock;
    cfg.colsb[1] = (uint16_t)(token_cols * (int)sizeof(int32_t));
    cfg.rows[1] = kAmxRowBlock;
    cfg.colsb[2] = kAmxKBlock;
    cfg.rows[2] = kAmxRowBlock;
    cfg.colsb[3] = (uint16_t)(kAmxRowBlock * 4);
    cfg.rows[3] = kAmxKBlock / 4;
    cfg.colsb[4] = (uint16_t)(kAmxRowBlock * 4);
    cfg.rows[4] = kAmxKBlock / 4;
    _tile_loadconfig(&cfg);
}

static void clear_packed_weights() {
    delete[] g_w_router_pack;
    delete[] g_w_gate_amx_pack;
    delete[] g_w_up_amx_pack;
    delete[] g_w_down_amx_pack;
    delete[] g_sh_gate_amx_pack;
    delete[] g_sh_up_amx_pack;
    delete[] g_sh_down_amx_pack;
    g_w_router_pack = nullptr;
    g_w_gate_amx_pack = nullptr;
    g_w_up_amx_pack = nullptr;
    g_w_down_amx_pack = nullptr;
    g_sh_gate_amx_pack = nullptr;
    g_sh_up_amx_pack = nullptr;
    g_sh_down_amx_pack = nullptr;
    g_pack_valid = false;
}

static inline bool packed_matches(const MoEWeights& w) {
    return g_pack_valid && g_pack_d_model == w.d_model &&
           g_pack_d_ff == w.d_ff && g_pack_num_experts == w.num_experts &&
           w.num_experts % kRowBlock == 0;
}

}  // namespace

void preprocess(MoEWeights& w) {
    clear_packed_weights();

    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int num_experts = w.num_experts;
    if (d_model % kRouterColBlock != 0 || d_model % kAmxKBlock != 0 ||
        d_ff % kAmxKBlock != 0 || d_ff % kAmxRowBlock != 0 ||
        d_model % kAmxRowBlock != 0 ||
        num_experts % kRowBlock != 0) {
        return;
    }

    const size_t router_size = (size_t)num_experts * d_model;
    const size_t gate_size = (size_t)num_experts * d_ff * d_model;
    const size_t down_size = (size_t)num_experts * d_model * d_ff;
    const size_t shared_gate_size = (size_t)d_ff * d_model;
    const size_t shared_down_size = (size_t)d_model * d_ff;

    g_w_router_pack = new float[router_size];
    g_w_gate_amx_pack = new int8_t[gate_size];
    g_w_up_amx_pack = new int8_t[gate_size];
    g_w_down_amx_pack = new int8_t[down_size];
    g_sh_gate_amx_pack = new int8_t[shared_gate_size];
    g_sh_up_amx_pack = new int8_t[shared_gate_size];
    g_sh_down_amx_pack = new int8_t[shared_down_size];

    pack_router_4x16(w.w_router, g_w_router_pack, num_experts, d_model);
    pack_amx_expert_matrices_u8bias(w.w_gate, g_w_gate_amx_pack, num_experts, d_ff,
                             d_model);
    pack_amx_expert_matrices_u8bias(w.w_up, g_w_up_amx_pack, num_experts, d_ff,
                             d_model);
    pack_amx_expert_matrices_u8bias(w.w_down, g_w_down_amx_pack, num_experts,
                             d_model, d_ff);
    pack_matrix_amx_kx16_vnni_u8bias(w.sh_gate, g_sh_gate_amx_pack, d_ff,
                                     d_model);
    pack_matrix_amx_kx16_vnni_u8bias(w.sh_up, g_sh_up_amx_pack, d_ff,
                                     d_model);
    pack_matrix_amx_kx16_vnni_u8bias(w.sh_down, g_sh_down_amx_pack, d_model,
                                     d_ff);

    g_pack_d_model = d_model;
    g_pack_d_ff = d_ff;
    g_pack_num_experts = num_experts;
    g_pack_valid = true;
}

// Fixed 4 experts x 4 tokens register-tiled router kernel. This is the best
// tile shape for the S4 workload and matches the 4-row packed router layout.
static MOE_NOINLINE void router_tile_kernel_4x4(
    const float* pack, const float* x, float* s_out, int e, int ne, int nt,
    int d_model, int col_chunks) {
    constexpr int T = 4;
    const size_t group_stride = (size_t)col_chunks * kRowBlock * kRouterColBlock;
    const float* group = pack + (size_t)(e / kRowBlock) * group_stride;

    const float* row0 = group;
    const float* row1 = group + kRouterColBlock;
    const float* row2 = group + 2 * kRouterColBlock;
    const float* row3 = group + 3 * kRouterColBlock;

    for (int t = 0; t < nt; t += T) {
        const float* x0 = x + (size_t)t * d_model;
        const float* x1 = x0 + d_model;
        const float* x2 = x1 + d_model;
        const float* x3 = x2 + d_model;

        __m512 acc00 = _mm512_setzero_ps();
        __m512 acc01 = _mm512_setzero_ps();
        __m512 acc02 = _mm512_setzero_ps();
        __m512 acc03 = _mm512_setzero_ps();
        __m512 acc10 = _mm512_setzero_ps();
        __m512 acc11 = _mm512_setzero_ps();
        __m512 acc12 = _mm512_setzero_ps();
        __m512 acc13 = _mm512_setzero_ps();
        __m512 acc20 = _mm512_setzero_ps();
        __m512 acc21 = _mm512_setzero_ps();
        __m512 acc22 = _mm512_setzero_ps();
        __m512 acc23 = _mm512_setzero_ps();
        __m512 acc30 = _mm512_setzero_ps();
        __m512 acc31 = _mm512_setzero_ps();
        __m512 acc32 = _mm512_setzero_ps();
        __m512 acc33 = _mm512_setzero_ps();

        for (int cb = 0; cb < col_chunks; cb++) {
            const size_t coff = (size_t)cb * kRowBlock * kRouterColBlock;
            const int d = cb * kRouterColBlock;
            __m512 w0 = _mm512_loadu_ps(row0 + coff);
            __m512 w1 = _mm512_loadu_ps(row1 + coff);
            __m512 w2 = _mm512_loadu_ps(row2 + coff);
            __m512 w3 = _mm512_loadu_ps(row3 + coff);

            __m512 xv0 = _mm512_loadu_ps(x0 + d);
            acc00 = _mm512_fmadd_ps(w0, xv0, acc00);
            acc01 = _mm512_fmadd_ps(w1, xv0, acc01);
            acc02 = _mm512_fmadd_ps(w2, xv0, acc02);
            acc03 = _mm512_fmadd_ps(w3, xv0, acc03);

            __m512 xv1 = _mm512_loadu_ps(x1 + d);
            acc10 = _mm512_fmadd_ps(w0, xv1, acc10);
            acc11 = _mm512_fmadd_ps(w1, xv1, acc11);
            acc12 = _mm512_fmadd_ps(w2, xv1, acc12);
            acc13 = _mm512_fmadd_ps(w3, xv1, acc13);

            __m512 xv2 = _mm512_loadu_ps(x2 + d);
            acc20 = _mm512_fmadd_ps(w0, xv2, acc20);
            acc21 = _mm512_fmadd_ps(w1, xv2, acc21);
            acc22 = _mm512_fmadd_ps(w2, xv2, acc22);
            acc23 = _mm512_fmadd_ps(w3, xv2, acc23);

            __m512 xv3 = _mm512_loadu_ps(x3 + d);
            acc30 = _mm512_fmadd_ps(w0, xv3, acc30);
            acc31 = _mm512_fmadd_ps(w1, xv3, acc31);
            acc32 = _mm512_fmadd_ps(w2, xv3, acc32);
            acc33 = _mm512_fmadd_ps(w3, xv3, acc33);
        }
        alignas(64) float dots[16] = {
            _mm512_reduce_add_ps(acc00), _mm512_reduce_add_ps(acc01),
            _mm512_reduce_add_ps(acc02), _mm512_reduce_add_ps(acc03),
            _mm512_reduce_add_ps(acc10), _mm512_reduce_add_ps(acc11),
            _mm512_reduce_add_ps(acc12), _mm512_reduce_add_ps(acc13),
            _mm512_reduce_add_ps(acc20), _mm512_reduce_add_ps(acc21),
            _mm512_reduce_add_ps(acc22), _mm512_reduce_add_ps(acc23),
            _mm512_reduce_add_ps(acc30), _mm512_reduce_add_ps(acc31),
            _mm512_reduce_add_ps(acc32), _mm512_reduce_add_ps(acc33),
        };
        __m512 v = _mm512_load_ps(dots);
        __m512 ev = router_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), v));
        __m512 s = _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(_mm512_set1_ps(1.0f), ev));
        _mm512_store_ps(dots, s);
        float* st0 = s_out + (size_t)t * ne + e;
        float* st1 = st0 + ne;
        float* st2 = st1 + ne;
        float* st3 = st2 + ne;
        _mm_storeu_ps(st0, _mm_load_ps(dots));
        _mm_storeu_ps(st1, _mm_load_ps(dots + 4));
        _mm_storeu_ps(st2, _mm_load_ps(dots + 8));
        _mm_storeu_ps(st3, _mm_load_ps(dots + 12));
    }
}

static MOE_NOINLINE void compute_router_scores_batch(
    const float* x, const MoEWeights& w, int nt, float* s_out) {
    const int d_model = w.d_model;
    const int ne = w.num_experts;
    const int col_chunks = d_model / kRouterColBlock;
#pragma omp parallel for schedule(static)
    for (int e = 0; e < ne; e += kRowBlock) {
        router_tile_kernel_4x4(g_w_router_pack, x, s_out, e, ne, nt, d_model,
                               col_chunks);
    }
}

// SIMD top-2 selection for K=2 (S4): sweep s+bias in 16-expert chunks keeping
// the running top-1/top-2 (value + index) per lane, then reduce the 32 lane
// candidates. Strict > matches the reference's smaller-index tie break.
static MOE_NOINLINE void select_topk_2_simd(const MoEWeights& w, const float* s, int* topk_idx) {
    const int ne = w.num_experts;
    const int nchunks = ne / 16;
    __m512 max1 = _mm512_set1_ps(-1e30f);
    __m512 max2 = _mm512_set1_ps(-1e30f);
    __m512i idx1 = _mm512_set1_epi32(-1);
    __m512i idx2 = _mm512_set1_epi32(-1);
    const __m512i lane = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
                                          5, 4, 3, 2, 1, 0);
    for (int c = 0; c < nchunks; c++) {
        const int e = c * 16;
        __m512 v = _mm512_add_ps(_mm512_loadu_ps(s + e), _mm512_loadu_ps(w.bias + e));
        __m512i idx_new = _mm512_add_epi32(_mm512_set1_epi32(e), lane);
        const __m512 old1 = max1;
        const __m512i oldidx1 = idx1;
        __mmask16 m1 = _mm512_cmp_ps_mask(v, old1, _CMP_GT_OQ);
        max1 = _mm512_max_ps(old1, v);
        idx1 = _mm512_mask_blend_epi32(m1, idx1, idx_new);
        const __m512 lo = _mm512_min_ps(old1, v);
        __m512i idx_lo = _mm512_mask_blend_epi32(m1, idx_new, oldidx1);
        __mmask16 m2 = _mm512_cmp_ps_mask(lo, max2, _CMP_GT_OQ);
        max2 = _mm512_mask_blend_ps(m2, max2, lo);
        idx2 = _mm512_mask_blend_epi32(m2, idx2, idx_lo);
    }
    alignas(64) float v1[16], v2[16];
    alignas(64) int i1[16], i2[16];
    _mm512_storeu_ps(v1, max1);
    _mm512_storeu_ps(v2, max2);
    _mm512_storeu_si512((__m512i*)i1, idx1);
    _mm512_storeu_si512((__m512i*)i2, idx2);
    float b1 = -1e30f, b2 = -1e30f;
    int k1 = -1, k2 = -1;
    for (int j = 0; j < 16; j++) {
        if (v1[j] > b1 || (v1[j] == b1 && i1[j] < k1)) {
            b2 = b1; k2 = k1; b1 = v1[j]; k1 = i1[j];
        } else if (v1[j] > b2 || (v1[j] == b2 && i1[j] < k2)) {
            b2 = v1[j]; k2 = i1[j];
        }
        if (v2[j] > b1 || (v2[j] == b1 && i2[j] < k1)) {
            b2 = b1; k2 = k1; b1 = v2[j]; k1 = i2[j];
        } else if (v2[j] > b2 || (v2[j] == b2 && i2[j] < k2)) {
            b2 = v2[j]; k2 = i2[j];
        }
    }
    topk_idx[0] = k1;
    topk_idx[1] = k2;
}

// Generic top-K selection (K != 2): K rounds of a vectorized 16-expert
// max scan with strict > (matches the reference's smaller-index tie break).
static MOE_NOINLINE void select_topk_generic(const MoEWeights& w,
                                             const float* s, int* topk_idx) {
    const int ne = w.num_experts;
    const int top_k = w.top_k;
    const int nchunks = ne / 16;
    const __m512i lane = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
                                          4, 3, 2, 1, 0);
    float biased[MAX_NUM_EXPERTS];
    for (int e = 0; e < ne; e++) biased[e] = s[e] + w.bias[e];

    for (int k = 0; k < top_k; k++) {
        __m512 maxv = _mm512_set1_ps(-1e30f);
        __m512i idxv = _mm512_set1_epi32(-1);
        for (int c = 0; c < nchunks; c++) {
            const int e = c * 16;
            __m512 v = _mm512_loadu_ps(biased + e);
            __m512i idx_new = _mm512_add_epi32(_mm512_set1_epi32(e), lane);
            __mmask16 m = _mm512_cmp_ps_mask(v, maxv, _CMP_GT_OQ);
            maxv = _mm512_max_ps(maxv, v);
            idxv = _mm512_mask_blend_epi32(m, idxv, idx_new);
        }
        alignas(64) float v[16];
        alignas(64) int i[16];
        _mm512_storeu_ps(v, maxv);
        _mm512_storeu_si512((__m512i*)i, idxv);
        float b = -1e30f;
        int bi = -1;
        for (int j = 0; j < 16; j++) {
            if (v[j] > b || (v[j] == b && i[j] < bi)) {
                b = v[j];
                bi = i[j];
            }
        }
        for (int e = nchunks * 16; e < ne; e++) {
            if (biased[e] > b || (biased[e] == b && e < bi)) {
                b = biased[e];
                bi = e;
            }
        }
        topk_idx[k] = bi;
        biased[bi] = -1e30f;
    }
}

static MOE_NOINLINE void select_topk_dispatch(const MoEWeights& w,
                                              const float* s, int* topk_idx) {
    if (w.top_k == 2) {
        select_topk_2_simd(w, s, topk_idx);
    } else {
        select_topk_generic(w, s, topk_idx);
    }
}

static inline bool shared_amx_supported(const MoEWeights& w, int num_tokens) {
    return std::getenv("MOE_DISABLE_AMX") == nullptr &&
           g_sh_gate_amx_pack != nullptr && g_sh_up_amx_pack != nullptr &&
           num_tokens > 0 && num_tokens % kAmxTokenBlock == 0 &&
           w.d_model % kAmxKBlock == 0 && w.d_ff % kAmxRowBlock == 0 &&
           w.d_ff <= MAX_D_FF && w.d_model <= MAX_D_MODEL;
}

static MOE_NOINLINE bool shared_expert_amx_fused(const float* x,
                                                 const MoEWeights& w,
                                                 float* y, int num_tokens) {
    if (std::getenv("MOE_AMX_INIT_ONLY") != nullptr) return false;

    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int gate_col_chunks = d_model / kAmxKBlock;
    const int gate_group_stride =
        gate_col_chunks * kAmxRowBlock * kAmxKBlock;
    const int down_col_chunks = d_ff / kAmxKBlock;
    const int down_group_stride =
        down_col_chunks * kAmxRowBlock * kAmxKBlock;
    const int token_blocks = num_tokens / kAmxTokenBlock;
    const int dim_blocks = d_model / kAmxRowBlock;
    bool ok = true;
    const bool dryrun = std::getenv("MOE_AMX_DRYRUN") != nullptr;

#pragma omp parallel
    {
        const bool thread_ok = init_amx_for_thread();
        if (!thread_ok) {
#pragma omp critical
            { ok = false; }
        }
        if (thread_ok) {
            load_amx_config(kAmxTokenBlock);
        }

        if (!dryrun && thread_ok) {
#pragma omp for schedule(static)
            for (int tb = 0; tb < num_tokens; tb += kAmxTokenBlock) {
                alignas(64) int32_t acc_g[kAmxRowBlock * kAmxTokenBlock];
                alignas(64) int32_t acc_u[kAmxRowBlock * kAmxTokenBlock];
                for (int f = 0; f < d_ff; f += kAmxRowBlock) {
                    _tile_zero(0);
                    _tile_zero(1);
                    const int8_t* wg_group =
                        g_sh_gate_amx_pack +
                        (size_t)(f / kAmxRowBlock) * gate_group_stride;
                    const int8_t* wu_group =
                        g_sh_up_amx_pack +
                        (size_t)(f / kAmxRowBlock) * gate_group_stride;

                    for (int cb = 0; cb < gate_col_chunks; cb++) {
                        const int d0 = cb * kAmxKBlock;
                        const size_t block =
                            (size_t)cb * kAmxRowBlock * kAmxKBlock;
                        _tile_loadd(2,
                                    g_batch_xq + (size_t)tb * d_model + d0,
                                    d_model);
                        _tile_loadd(3, wg_group + block, kAmxRowBlock * 4);
                        _tile_loadd(4, wu_group + block, kAmxRowBlock * 4);
                        _tile_dpbsud(0, 2, 3);
                        _tile_dpbsud(1, 2, 4);
                    }

                    _tile_stored(0, acc_g,
                                 kAmxTokenBlock * (int)sizeof(int32_t));
                    _tile_stored(1, acc_u,
                                 kAmxTokenBlock * (int)sizeof(int32_t));
                    for (int tt = 0; tt < kAmxTokenBlock; tt++) {
                        const int t = tb + tt;
                        const __m512 scale_g =
                            _mm512_set1_ps(g_batch_s_x[t] * w.sh_s_gate);
                        const __m512 scale_u =
                            _mm512_set1_ps(g_batch_s_x[t] * w.sh_s_up);
                        const __m512i corr =
                            _mm512_set1_epi32(g_batch_x_correction[t]);
                        __m512i ig = _mm512_sub_epi32(
                            _mm512_load_si512(
                                (const __m512i*)(acc_g +
                                                 tt * kAmxRowBlock)),
                            corr);
                        __m512i iu = _mm512_sub_epi32(
                            _mm512_load_si512(
                                (const __m512i*)(acc_u +
                                                 tt * kAmxRowBlock)),
                            corr);
                        __m512 vg =
                            _mm512_mul_ps(_mm512_cvtepi32_ps(ig), scale_g);
                        __m512 vu =
                            _mm512_mul_ps(_mm512_cvtepi32_ps(iu), scale_u);
                        __m512 exp_neg =
                            fast_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(),
                                                      vg));
                        __m512 silu = _mm512_div_ps(
                            vg, _mm512_add_ps(_mm512_set1_ps(1.0f),
                                              exp_neg));
                        _mm512_storeu_ps(
                            g_shared_h_batch + (size_t)t * d_ff + f,
                            _mm512_mul_ps(silu, vu));
                    }
                }
            }

#pragma omp for schedule(static)
            for (int t = 0; t < num_tokens; t++) {
                float* h = g_shared_h_batch + (size_t)t * d_ff;
                int8_t* hq = g_shared_hq_batch + (size_t)t * d_ff;
                int32_t sum_hq = 0;
                float h_amax = 0.0f;
                const __m512 sign_mask =
                    _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
                __m512 maxv = _mm512_setzero_ps();
                for (int f = 0; f < d_ff; f += 16) {
                    __m512 hv = _mm512_loadu_ps(h + f);
                    maxv = _mm512_max_ps(maxv, _mm512_and_ps(hv, sign_mask));
                }
                h_amax = _mm512_reduce_max_ps(maxv);
                const float s_h =
                    ffn_requantize_with_amax(h, hq, d_ff, h_amax, &sum_hq);
                g_shared_down_scale[t] = s_h * w.sh_s_down;
                g_shared_h_correction[t] = sum_hq * 128;
            }

#pragma omp for collapse(2) schedule(static)
            for (int tbi = 0; tbi < token_blocks; tbi++) {
                for (int dbi = 0; dbi < dim_blocks; dbi++) {
                    const int tb = tbi * kAmxTokenBlock;
                    const int db = dbi * kAmxRowBlock;
                    alignas(64) int32_t acc[kAmxTokenBlock * kAmxRowBlock];
                    _tile_zero(0);
                    const int8_t* wd_group =
                        g_sh_down_amx_pack + (size_t)dbi * down_group_stride;
                    for (int cb = 0; cb < down_col_chunks; cb++) {
                        const int f0 = cb * kAmxKBlock;
                        const size_t block =
                            (size_t)cb * kAmxRowBlock * kAmxKBlock;
                        _tile_loadd(2,
                                    g_shared_hq_batch + (size_t)tb * d_ff +
                                        f0,
                                    d_ff);
                        _tile_loadd(3, wd_group + block, kAmxRowBlock * 4);
                        _tile_dpbsud(0, 2, 3);
                    }
                    _tile_stored(0, acc,
                                 kAmxRowBlock * (int)sizeof(int32_t));
                    for (int tt = 0; tt < kAmxTokenBlock; tt++) {
                        const int t = tb + tt;
                        const __m512i corr =
                            _mm512_set1_epi32(g_shared_h_correction[t]);
                        __m512i iv = _mm512_sub_epi32(
                            _mm512_load_si512(
                                (const __m512i*)(acc + tt * kAmxRowBlock)),
                            corr);
                        __m512 outv =
                            _mm512_mul_ps(_mm512_cvtepi32_ps(iv),
                                          _mm512_set1_ps(
                                              g_shared_down_scale[t]));
                        __m512 xv =
                            _mm512_loadu_ps(x + (size_t)t * d_model + db);
                        _mm512_storeu_ps(y + (size_t)t * d_model + db,
                                         _mm512_add_ps(xv, outv));
                    }
                }
            }
        }

        if (thread_ok) {
            _tile_release();
        }
    }
    return ok && !dryrun;
}

static MOE_NOINLINE void routed_gate_up_amx_chunk(
    const MoEWeights& w, int e, const int* tokens, const int* slots, int start,
    int count) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int col_chunks = d_model / kAmxKBlock;
    const int amx_group_stride = col_chunks * kAmxRowBlock * kAmxKBlock;
    const size_t gate_size = (size_t)d_ff * d_model;
    alignas(64) int8_t x_tile[kAmxTokenBlock * kAmxKBlock];
    alignas(64) int32_t acc_g[kAmxTokenBlock * kAmxRowBlock];
    alignas(64) int32_t acc_u[kAmxTokenBlock * kAmxRowBlock];

    const int8_t* wg_base = g_w_gate_amx_pack + (size_t)e * gate_size;
    const int8_t* wu_base = g_w_up_amx_pack + (size_t)e * gate_size;
    for (int f = 0; f < d_ff; f += kAmxRowBlock) {
        _tile_zero(0);
        _tile_zero(1);
        const int8_t* wg_group =
            wg_base + (size_t)(f / kAmxRowBlock) * amx_group_stride;
        const int8_t* wu_group =
            wu_base + (size_t)(f / kAmxRowBlock) * amx_group_stride;

        for (int cb = 0; cb < col_chunks; cb++) {
            const int d0 = cb * kAmxKBlock;
            for (int tt = 0; tt < kAmxTokenBlock; tt++) {
                int8_t* dst = x_tile + tt * kAmxKBlock;
                if (tt < count) {
                    const int t = tokens[start + tt];
                    std::memcpy(dst, g_batch_xq + (size_t)t * d_model + d0,
                                kAmxKBlock);
                } else {
                    std::memset(dst, 0, kAmxKBlock);
                }
            }

            const size_t block = (size_t)cb * kAmxRowBlock * kAmxKBlock;
            _tile_loadd(2, x_tile, kAmxKBlock);
            _tile_loadd(3, wg_group + block, kAmxRowBlock * 4);
            _tile_loadd(4, wu_group + block, kAmxRowBlock * 4);
            _tile_dpbsud(0, 2, 3);
            _tile_dpbsud(1, 2, 4);
        }

        _tile_stored(0, acc_g, kAmxRowBlock * (int)sizeof(int32_t));
        _tile_stored(1, acc_u, kAmxRowBlock * (int)sizeof(int32_t));
        for (int tt = 0; tt < count; tt++) {
            const int t = tokens[start + tt];
            const int k = slots[start + tt];
            const float scale_g = g_batch_s_x[t] * w.s_gate[e];
            const float scale_u = g_batch_s_x[t] * w.s_up[e];
            const __m512i corr = _mm512_set1_epi32(g_batch_x_correction[t]);
            __m512i ig = _mm512_sub_epi32(
                _mm512_load_si512((const __m512i*)(acc_g +
                                                   tt * kAmxRowBlock)),
                corr);
            __m512i iu = _mm512_sub_epi32(
                _mm512_load_si512((const __m512i*)(acc_u +
                                                   tt * kAmxRowBlock)),
                corr);
            __m512 vg =
                _mm512_mul_ps(_mm512_cvtepi32_ps(ig), _mm512_set1_ps(scale_g));
            __m512 vu =
                _mm512_mul_ps(_mm512_cvtepi32_ps(iu), _mm512_set1_ps(scale_u));
            __m512 exp_neg =
                fast_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), vg));
            __m512 silu =
                _mm512_div_ps(vg, _mm512_add_ps(_mm512_set1_ps(1.0f),
                                                exp_neg));
            _mm512_storeu_ps(
                g_routed_h + ((size_t)t * MAX_TOP_K + k) * d_ff + f,
                _mm512_mul_ps(silu, vu));
        }
    }
}

static MOE_NOINLINE void routed_requantize_chunk(const MoEWeights& w, int e,
                                                 const int* tokens,
                                                 const int* slots, int start,
                                                 int count) {
    const int d_ff = w.d_ff;
    for (int i = 0; i < count; i++) {
        const int t = tokens[start + i];
        const int k = slots[start + i];
        const size_t base = (size_t)t * MAX_TOP_K + k;
        int32_t sum_hq = 0;
        const float s_h =
            ffn_requantize(g_routed_h + base * d_ff,
                           g_routed_hq + base * d_ff, d_ff, &sum_hq);
        g_routed_down_scale[base] = s_h * w.s_down[e];
        g_routed_h_correction[base] = sum_hq * 128;
    }
}

static MOE_NOINLINE void routed_down_amx_chunk(
    const MoEWeights& w, int e, const int* tokens, const int* slots, int start,
    int count) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int col_chunks = d_ff / kAmxKBlock;
    const int amx_group_stride = col_chunks * kAmxRowBlock * kAmxKBlock;
    const size_t down_size = (size_t)d_model * d_ff;
    const int8_t* wd_base = g_w_down_amx_pack + (size_t)e * down_size;
    alignas(64) int8_t hq_tile[kAmxTokenBlock * kAmxKBlock];
    alignas(64) int32_t acc[kAmxTokenBlock * kAmxRowBlock];

    for (int db = 0; db < d_model; db += kAmxRowBlock) {
        _tile_zero(0);
        const int8_t* wd_group =
            wd_base + (size_t)(db / kAmxRowBlock) * amx_group_stride;
        for (int cb = 0; cb < col_chunks; cb++) {
            const int f0 = cb * kAmxKBlock;
            for (int tt = 0; tt < kAmxTokenBlock; tt++) {
                int8_t* dst = hq_tile + tt * kAmxKBlock;
                if (tt < count) {
                    const int t = tokens[start + tt];
                    const int k = slots[start + tt];
                    const size_t base = (size_t)t * MAX_TOP_K + k;
                    std::memcpy(dst, g_routed_hq + base * d_ff + f0,
                                kAmxKBlock);
                } else {
                    std::memset(dst, 0, kAmxKBlock);
                }
            }

            const size_t block = (size_t)cb * kAmxRowBlock * kAmxKBlock;
            _tile_loadd(2, hq_tile, kAmxKBlock);
            _tile_loadd(3, wd_group + block, kAmxRowBlock * 4);
            _tile_dpbsud(0, 2, 3);
        }

        _tile_stored(0, acc, kAmxRowBlock * (int)sizeof(int32_t));
        for (int tt = 0; tt < count; tt++) {
            const int t = tokens[start + tt];
            const int k = slots[start + tt];
            const size_t base = (size_t)t * MAX_TOP_K + k;
            const float* st = g_batch_s + (size_t)t * w.num_experts;
            const float gate = st[e] / g_batch_gate_sum[t];
            const __m512i corr = _mm512_set1_epi32(g_routed_h_correction[base]);
            __m512i iv = _mm512_sub_epi32(
                _mm512_load_si512((const __m512i*)(acc +
                                                   tt * kAmxRowBlock)),
                corr);
            __m512 outv =
                _mm512_mul_ps(_mm512_cvtepi32_ps(iv),
                              _mm512_set1_ps(g_routed_down_scale[base] * gate));
            _mm512_storeu_ps(g_routed_out + base * d_model + db, outv);
        }
    }
}

static MOE_NOINLINE void process_routed_experts_bucketed(
    const MoEWeights& w, float* y, int num_tokens) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int ne = w.num_experts;
    const int top_k = w.top_k;
    const int max_assign = num_tokens * top_k;

    for (int e = 0; e < ne; e++) {
        g_routed_bucket_count[e] = 0;
    }
    for (int t = 0; t < num_tokens; t++) {
        const int* topk = g_batch_topk + (size_t)t * MAX_TOP_K;
        for (int k = 0; k < top_k; k++) {
            const int e = topk[k];
            const int idx = g_routed_bucket_count[e]++;
            g_routed_bucket_token[(size_t)e * max_assign + idx] = t;
            g_routed_bucket_slot[(size_t)e * max_assign + idx] = k;
        }
    }

    if (g_w_gate_amx_pack != nullptr && g_w_up_amx_pack != nullptr) {
        int num_tasks = 0;
        for (int e = 0; e < ne; e++) {
            const int count = g_routed_bucket_count[e];
            int done = 0;
            while (done + kAmxTokenBlock <= count) {
                g_routed_task_expert[num_tasks] = e;
                g_routed_task_start[num_tasks] = done;
                g_routed_task_count[num_tasks] = kAmxTokenBlock;
                num_tasks++;
                done += kAmxTokenBlock;
            }
            if (done < count) {
                g_routed_task_expert[num_tasks] = e;
                g_routed_task_start[num_tasks] = done;
                g_routed_task_count[num_tasks] = count - done;
                num_tasks++;
            }
        }

#pragma omp parallel
        {
            const bool thread_ok = init_amx_for_thread();
            if (thread_ok) {
                load_amx_config(kAmxTokenBlock);
            }

            if (thread_ok) {
#pragma omp for schedule(dynamic, 1)
                for (int task = 0; task < num_tasks; task++) {
                    const int e = g_routed_task_expert[task];
                    const int start = g_routed_task_start[task];
                    const int count = g_routed_task_count[task];
                    const int* tokens =
                        g_routed_bucket_token + (size_t)e * max_assign;
                    const int* slots =
                        g_routed_bucket_slot + (size_t)e * max_assign;
                    routed_gate_up_amx_chunk(w, e, tokens, slots, start,
                                             count);
                    routed_requantize_chunk(w, e, tokens, slots, start,
                                            count);
                    routed_down_amx_chunk(w, e, tokens, slots, start,
                                          count);
                }
            }

#pragma omp for schedule(static)
            for (int t = 0; t < num_tokens; t++) {
                float* yt = y + (size_t)t * d_model;
                const float* out0 =
                    g_routed_out + ((size_t)t * MAX_TOP_K) * d_model;
                if (top_k == 2) {
                    const float* out1 = out0 + d_model;
                    for (int d = 0; d < d_model; d += 16) {
                        __m512 acc = _mm512_loadu_ps(yt + d);
                        acc = _mm512_add_ps(acc, _mm512_loadu_ps(out0 + d));
                        acc = _mm512_add_ps(acc, _mm512_loadu_ps(out1 + d));
                        _mm512_storeu_ps(yt + d, acc);
                    }
                } else {
                    const float* outs[MAX_TOP_K];
                    for (int k = 0; k < top_k; k++) {
                        outs[k] = out0 + (size_t)k * d_model;
                    }
                    for (int d = 0; d < d_model; d += 16) {
                        __m512 acc = _mm512_loadu_ps(yt + d);
                        for (int k = 0; k < top_k; k++) {
                            acc = _mm512_add_ps(acc, _mm512_loadu_ps(outs[k] + d));
                        }
                        _mm512_storeu_ps(yt + d, acc);
                    }
                }
            }

            if (thread_ok) {
                _tile_release();
            }
        }
        return;
    }
}

static MOE_NOINLINE void process_tokens_shared_batch_first(
    const float* x, const MoEWeights& w, float* y, int num_tokens) {
    const int d_model = w.d_model;
    const int ne = w.num_experts;
    const size_t s_stride = (size_t)ne;
    const size_t xq_stride = (size_t)d_model;
    if (!shared_amx_supported(w, num_tokens)) {
        moe_forward_ref(x, w, y, num_tokens);
        return;
    }

    compute_router_scores_batch(x, w, num_tokens, g_batch_s);

#pragma omp parallel for schedule(static)
    for (int t = 0; t < num_tokens; t++) {
        const float* xt = x + (size_t)t * d_model;
        float* st = g_batch_s + (size_t)t * s_stride;
        int* topk = g_batch_topk + (size_t)t * MAX_TOP_K;
        int8_t* xq = g_batch_xq + (size_t)t * xq_stride;

        select_topk_dispatch(w, st, topk);
        g_batch_gate_sum[t] = compute_gate_sum(st, topk, w.top_k);
        int32_t sum_xq = 0;
        float s_x = quantize_token(xt, xq, d_model, &sum_xq);
        g_batch_s_x[t] = s_x;
        g_batch_x_correction[t] = sum_xq * 128;
    }

    if (!shared_expert_amx_fused(x, w, y, num_tokens)) {
        moe_forward_ref(x, w, y, num_tokens);
        return;
    }

    process_routed_experts_bucketed(w, y, num_tokens);
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    if (num_tokens > 1 && packed_matches(w)) {
        process_tokens_shared_batch_first(x, w, y, num_tokens);
        return;
    }

    moe_forward_ref(x, w, y, num_tokens);
}

#undef MOE_NOINLINE
