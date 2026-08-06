// Main task: optimize the MoE forward pass.
#include "moe.h"
#include "moe_opt_common.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

using moe_common::fast_exp_ps;
using moe_common::router_exp_ps;
using moe_common::reduce_add_epi32_512;
using moe_common::pack_router_4x16;
using moe_common::pack_matrix_u8bias;
using moe_common::pack_expert_matrices_u8bias;
using moe_common::compute_gate_sum;
using moe_common::quantize_token;
using moe_common::ffn_requantize_with_amax;

#if defined(__GNUC__) || defined(__clang__)
#define MOE_NOINLINE __attribute__((noinline, aligned(64)))
#else
#define MOE_NOINLINE
#endif

namespace {

constexpr int kRowBlock = 4;
constexpr int kGateUpRowBlock = 2;
constexpr int kDownRowBlock = 4;
constexpr int kGateUpColBlock = 64;
constexpr int kDownColBlock = 64;
constexpr int kRouterColBlock = 16;
constexpr int kExpertJobs = MAX_TOP_K + 1;
constexpr int kHybridThreadCount = 3 * kExpertJobs;

int g_pack_d_model = 0;
int g_pack_d_ff = 0;
int g_pack_num_experts = 0;
bool g_pack_valid = false;

std::thread g_expert_workers[MAX_TOP_K];
std::atomic<int> g_worker_epoch{0};
std::atomic<int> g_worker_ready{0};
std::atomic<bool> g_worker_stop{false};
bool g_worker_started = false;
bool g_worker_shutdown_registered = false;
const MoEWeights* g_worker_w = nullptr;
const int* g_worker_topk_idx = nullptr;
const int8_t* g_worker_xq = nullptr;
float g_worker_s_x = 1.0f;
int32_t g_worker_x_correction = 0;
float g_worker_gate[MAX_TOP_K] = {};
float (*g_worker_expert_out)[MAX_D_MODEL] = nullptr;
int g_worker_d_model = 0;
int g_worker_d_ff = 0;
int g_worker_top_k = 0;
struct alignas(64) WorkerStatus {
    std::atomic<int> done_epoch;
};
WorkerStatus g_worker_status[MAX_TOP_K];
int g_affinity_cpus[kHybridThreadCount] = {};
int g_affinity_count = 0;
bool g_main_affinity_set = false;

std::thread g_down_helpers[kExpertJobs];
std::atomic<int> g_down_epoch[kExpertJobs];
std::atomic<int> g_down_done_epoch[kExpertJobs];
std::atomic<int> g_down_ready{0};
std::atomic<bool> g_down_stop{false};
bool g_down_started = false;
bool g_down_shutdown_registered = false;
struct DownTask {
    const int8_t* w_down;
    const int8_t* hq;
    float s_down;
    int d_model;
    int d_ff;
    int32_t correction;
    int row_begin;
    int row_end;
    float* out;
};
DownTask g_down_task[kExpertJobs];

std::thread g_gate_up_helpers[kExpertJobs];
std::atomic<int> g_gate_up_epoch[kExpertJobs];
std::atomic<int> g_gate_up_done_epoch[kExpertJobs];
std::atomic<int> g_gate_up_ready{0};
std::atomic<bool> g_gate_up_stop{false};
bool g_gate_up_started = false;
bool g_gate_up_shutdown_registered = false;
struct GateUpTask {
    const int8_t* w_gate;
    const int8_t* w_up;
    const int8_t* xq;
    float s_gate;
    float s_up;
    float s_x;
    int32_t correction;
    float* h;
    int d_model;
    int d_ff;
    int f_begin;
    int f_end;
    float h_amax;
};
GateUpTask g_gate_up_task[kExpertJobs];

float* g_w_router_pack = nullptr;
int8_t* g_w_gate_pack = nullptr;
int8_t* g_w_up_pack = nullptr;
int8_t* g_w_down_pack = nullptr;
int8_t* g_sh_gate_pack = nullptr;
int8_t* g_sh_up_pack = nullptr;
int8_t* g_sh_down_pack = nullptr;

static void clear_packed_weights() {
    delete[] g_w_router_pack;
    delete[] g_w_gate_pack;
    delete[] g_w_up_pack;
    delete[] g_w_down_pack;
    delete[] g_sh_gate_pack;
    delete[] g_sh_up_pack;
    delete[] g_sh_down_pack;
    g_w_router_pack = nullptr;
    g_w_gate_pack = nullptr;
    g_w_up_pack = nullptr;
    g_w_down_pack = nullptr;
    g_sh_gate_pack = nullptr;
    g_sh_up_pack = nullptr;
    g_sh_down_pack = nullptr;
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

    const size_t router_size = (size_t)num_experts * d_model;
    const size_t gate_size = (size_t)num_experts * d_ff * d_model;
    const size_t down_size = (size_t)num_experts * d_model * d_ff;
    const size_t shared_gate_size = (size_t)d_ff * d_model;
    const size_t shared_down_size = (size_t)d_model * d_ff;

    g_w_router_pack = new float[router_size];
    g_w_gate_pack = new int8_t[gate_size];
    g_w_up_pack = new int8_t[gate_size];
    g_w_down_pack = new int8_t[down_size];
    g_sh_gate_pack = new int8_t[shared_gate_size];
    g_sh_up_pack = new int8_t[shared_gate_size];
    g_sh_down_pack = new int8_t[shared_down_size];

    pack_router_4x16(w.w_router, g_w_router_pack, num_experts, d_model);
    pack_expert_matrices_u8bias(w.w_gate, g_w_gate_pack, num_experts, d_ff,
                                 d_model, kGateUpRowBlock, kGateUpColBlock);
    pack_expert_matrices_u8bias(w.w_up, g_w_up_pack, num_experts, d_ff,
                                 d_model, kGateUpRowBlock, kGateUpColBlock);
    pack_expert_matrices_u8bias(w.w_down, g_w_down_pack, num_experts, d_model,
                              d_ff, kDownRowBlock, kDownColBlock);
    pack_matrix_u8bias(w.sh_gate, g_sh_gate_pack, d_ff, d_model, kGateUpRowBlock, kGateUpColBlock);
    pack_matrix_u8bias(w.sh_up, g_sh_up_pack, d_ff, d_model, kGateUpRowBlock, kGateUpColBlock);
    pack_matrix_u8bias(w.sh_down, g_sh_down_pack, d_model, d_ff, kDownRowBlock, kDownColBlock);

    g_pack_d_model = d_model;
    g_pack_d_ff = d_ff;
    g_pack_num_experts = num_experts;
    g_pack_valid = true;
}

static MOE_NOINLINE void compute_router_scores(const float* xt,
                                               const MoEWeights& w,
                                               float* s) {
    const int d_model = w.d_model;
    const int num_experts = w.num_experts;

    if (packed_matches(w)) {
        const int col_chunks = d_model / kRouterColBlock;
        const int group_stride = col_chunks * kRowBlock * kRouterColBlock;
        int e = 0;

        for (; e + kRowBlock <= num_experts; e += kRowBlock) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            const float* wr_group =
                g_w_router_pack + (size_t)(e / kRowBlock) * group_stride;

            for (int cb = 0; cb < col_chunks; cb++) {
                const int d = cb * kRouterColBlock;
                const size_t block =
                    (size_t)cb * kRowBlock * kRouterColBlock;
                __m512 xv = _mm512_loadu_ps(xt + d);
                acc0 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(wr_group + block), xv, acc0);
                acc1 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(wr_group + block + kRouterColBlock), xv,
                    acc1);
                acc2 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(wr_group + block + 2 * kRouterColBlock),
                    xv, acc2);
                acc3 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(wr_group + block + 3 * kRouterColBlock),
                    xv, acc3);
            }

            const __m512 one = _mm512_set1_ps(1.0f);
            float acc_arr[16] = {};
            acc_arr[0] = _mm512_reduce_add_ps(acc0);
            acc_arr[1] = _mm512_reduce_add_ps(acc1);
            acc_arr[2] = _mm512_reduce_add_ps(acc2);
            acc_arr[3] = _mm512_reduce_add_ps(acc3);
            __m512 accv = _mm512_loadu_ps(acc_arr);
            __m512 ev = router_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), accv));
            __m512 sv = _mm512_div_ps(one, _mm512_add_ps(one, ev));
            _mm512_mask_storeu_ps(s + e, 0x000F, sv);
        }

        return;
    }

    for (int e = 0; e < num_experts; e++) {
        float acc = 0.0f;
        for (int d = 0; d < d_model; d++) {
            acc += w.w_router[(size_t)e * d_model + d] * xt[d];
        }
        s[e] = 1.0f / (1.0f + expf(-acc));
    }
}

static MOE_NOINLINE void select_topk_experts(const MoEWeights& w,
                                             const float* s, int* topk_idx) {
    const int num_experts = w.num_experts;
    const int top_k = w.top_k;

    if (num_experts <= 16 && top_k <= 4) {
        __m512 bv = _mm512_add_ps(_mm512_loadu_ps(s), _mm512_loadu_ps(w.bias));
        const __mmask16 keep =
            (__mmask16)((num_experts == 16) ? 0xFFFF
                                            : (unsigned)((1u << num_experts) - 1));
        bv = _mm512_mask_blend_ps((__mmask16)(~keep & 0xFFFF), bv,
                                  _mm512_set1_ps(-INFINITY));
        for (int k = 0; k < top_k; k++) {
            __m512 maxv = _mm512_set1_ps(_mm512_reduce_max_ps(bv));
            __mmask16 eq = _mm512_cmp_ps_mask(bv, maxv, _CMP_EQ_OQ);
            int best = __builtin_ctz((unsigned)eq);
            topk_idx[k] = best;
            bv = _mm512_mask_blend_ps((__mmask16)(1u << best), bv,
                                      _mm512_set1_ps(-INFINITY));
        }
        return;
    }

    bool used[MAX_NUM_EXPERTS] = {};
    for (int k = 0; k < top_k; k++) {
        int best = -1;
        for (int e = 0; e < num_experts; e++) {
            if (used[e]) continue;
            if (best < 0 || s[e] + w.bias[e] > s[best] + w.bias[best]) {
                best = e;
            }
        }
        used[best] = true;
        topk_idx[k] = best;
    }
}

static MOE_NOINLINE float ffn_gate_up_packed_range(
    const int8_t* w_gate, const int8_t* w_up, float s_gate, float s_up,
    const int8_t* xq, float s_x, int32_t correction, float* h, int d_model,
    int f_begin, int f_end) {
    const float scale_g = s_x * s_gate;
    const float scale_u = s_x * s_up;
    const int col_chunks = d_model / kGateUpColBlock;
    const int group_stride = col_chunks * kGateUpRowBlock * kGateUpColBlock;

    float vg_buf[MAX_D_FF];
    float vu_buf[MAX_D_FF];

    for (int f = f_begin; f < f_end; f += kGateUpRowBlock) {
        const int8_t* wg_group = w_gate + (size_t)(f / kGateUpRowBlock) * group_stride;
        const int8_t* wu_group = w_up + (size_t)(f / kGateUpRowBlock) * group_stride;

        __m512i acc_g[kGateUpRowBlock];
        __m512i acc_u[kGateUpRowBlock];
        acc_g[0] = _mm512_setzero_si512();
        acc_u[0] = _mm512_setzero_si512();
        acc_g[1] = _mm512_setzero_si512();
        acc_u[1] = _mm512_setzero_si512();

        for (int cb = 0; cb < col_chunks; cb++) {
            const int d = cb * kGateUpColBlock;
            const size_t block = (size_t)cb * kGateUpRowBlock * kGateUpColBlock;
            __m512i xv8 = _mm512_loadu_si512((const __m512i*)(xq + d));

            // kGateUpRowBlock == 2: unrolled rows, load-then-compute interleaved.
            __m512i wg0 = _mm512_loadu_si512((const __m512i*)(wg_group + block));
            __m512i wu0 = _mm512_loadu_si512((const __m512i*)(wu_group + block));
            acc_g[0] = _mm512_dpbusd_epi32(acc_g[0], wg0, xv8);
            acc_u[0] = _mm512_dpbusd_epi32(acc_u[0], wu0, xv8);
            __m512i wg1 = _mm512_loadu_si512(
                (const __m512i*)(wg_group + block + kGateUpColBlock));
            __m512i wu1 = _mm512_loadu_si512(
                (const __m512i*)(wu_group + block + kGateUpColBlock));
            acc_g[1] = _mm512_dpbusd_epi32(acc_g[1], wg1, xv8);
            acc_u[1] = _mm512_dpbusd_epi32(acc_u[1], wu1, xv8);
        }
        
        vg_buf[f + 0] = (float)(reduce_add_epi32_512(acc_g[0]) - correction) * scale_g;
        vu_buf[f + 0] = (float)(reduce_add_epi32_512(acc_u[0]) - correction) * scale_u;
        vg_buf[f + 1] = (float)(reduce_add_epi32_512(acc_g[1]) - correction) * scale_g;
        vu_buf[f + 1] = (float)(reduce_add_epi32_512(acc_u[1]) - correction) * scale_u;
    }

    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 sign_mask =
        _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    __m512 maxv = _mm512_setzero_ps();
    for (int f = f_begin; f < f_end; f += 16) {
        __m512 vg = _mm512_loadu_ps(vg_buf + f);
        __m512 vu = _mm512_loadu_ps(vu_buf + f);
        __m512 exp_neg = fast_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), vg));
        __m512 silu = _mm512_div_ps(vg, _mm512_add_ps(vone, exp_neg));
        __m512 hv = _mm512_mul_ps(silu, vu);
        maxv = _mm512_max_ps(maxv, _mm512_and_ps(hv, sign_mask));
        _mm512_storeu_ps(h + f, hv);
    }
    return _mm512_reduce_max_ps(maxv);
}

static MOE_NOINLINE void ffn_down_packed_range(
    const int8_t* w_down, float s_down, const int8_t* hq, int d_ff,
    int32_t correction, float* out, int row_begin, int row_end) {
    const int col_chunks = d_ff / kDownColBlock;
    const int group_stride = col_chunks * kDownRowBlock * kDownColBlock;

    for (int d = row_begin; d < row_end; d += kDownRowBlock) {
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        const int8_t* wd_group =
            w_down + (size_t)(d / kDownRowBlock) * group_stride;

        for (int cb = 0; cb < col_chunks; cb++) {
            const int f = cb * kDownColBlock;
            const size_t block = (size_t)cb * kDownRowBlock * kDownColBlock;
            __m512i hv8 = _mm512_loadu_si512((const __m512i*)(hq + f));

            __m512i wd0 = _mm512_loadu_si512((const __m512i*)(wd_group + block));
            acc0 = _mm512_dpbusd_epi32(acc0, wd0, hv8);

            __m512i wd1 = _mm512_loadu_si512(
                (const __m512i*)(wd_group + block + kDownColBlock));
            acc1 = _mm512_dpbusd_epi32(acc1, wd1, hv8);

            __m512i wd2 = _mm512_loadu_si512(
                (const __m512i*)(wd_group + block + 2 * kDownColBlock));
            acc2 = _mm512_dpbusd_epi32(acc2, wd2, hv8);

            __m512i wd3 = _mm512_loadu_si512(
                (const __m512i*)(wd_group + block + 3 * kDownColBlock));
            acc3 = _mm512_dpbusd_epi32(acc3, wd3, hv8);
        }

        out[d] = (float)(reduce_add_epi32_512(acc0) - correction) * s_down;
        out[d + 1] = (float)(reduce_add_epi32_512(acc1) - correction) * s_down;
        out[d + 2] = (float)(reduce_add_epi32_512(acc2) - correction) * s_down;
        out[d + 3] = (float)(reduce_add_epi32_512(acc3) - correction) * s_down;
    }
}

static void stop_expert_workers() {
    if (!g_worker_started) return;
    g_worker_stop.store(true, std::memory_order_release);
    g_worker_epoch.fetch_add(1, std::memory_order_release);
    for (int i = 0; i < MAX_TOP_K; i++) {
        if (g_expert_workers[i].joinable()) {
            g_expert_workers[i].join();
        }
    }
    g_worker_started = false;
}

static void collect_affinity_cpus() {
    if (g_affinity_count > 0) return;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }
    for (int cpu = 0; cpu < CPU_SETSIZE &&
                      g_affinity_count < kHybridThreadCount;
         cpu++) {
        if (CPU_ISSET(cpu, &allowed)) {
            g_affinity_cpus[g_affinity_count++] = cpu;
        }
    }
}

static void bind_current_thread_to_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void bind_thread_to_cpu(std::thread& th, int cpu) {
    if (cpu < 0 || !th.joinable()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(th.native_handle(), sizeof(set), &set);
}

static void stop_down_helpers() {
    if (!g_down_started) return;
    g_down_stop.store(true, std::memory_order_release);
    for (int i = 0; i < kExpertJobs; i++) {
        g_down_epoch[i].fetch_add(1, std::memory_order_release);
    }
    for (int i = 0; i < kExpertJobs; i++) {
        if (g_down_helpers[i].joinable()) {
            g_down_helpers[i].join();
        }
    }
    g_down_started = false;
}

static void down_helper_loop(int job_id) {
    if (g_affinity_count > kExpertJobs + job_id) {
        bind_current_thread_to_cpu(g_affinity_cpus[kExpertJobs + job_id]);
    }
    int seen_epoch = g_down_epoch[job_id].load(std::memory_order_acquire);
    g_down_ready.fetch_add(1, std::memory_order_release);
    while (true) {
        int epoch = g_down_epoch[job_id].load(std::memory_order_acquire);
        while (epoch == seen_epoch &&
               !g_down_stop.load(std::memory_order_acquire)) {
            _mm_pause();
            epoch = g_down_epoch[job_id].load(std::memory_order_acquire);
        }
        if (g_down_stop.load(std::memory_order_acquire)) {
            return;
        }
        seen_epoch = epoch;

        const DownTask& task = g_down_task[job_id];
        ffn_down_packed_range(task.w_down, task.s_down, task.hq,
                              task.d_ff, task.correction,
                              task.out, task.row_begin, task.row_end);
        g_down_done_epoch[job_id].store(seen_epoch, std::memory_order_release);
    }
}

static void start_down_helpers() {
    if (g_down_started) return;
    collect_affinity_cpus();
    g_down_stop.store(false, std::memory_order_release);
    g_down_ready.store(0, std::memory_order_release);
    for (int i = 0; i < kExpertJobs; i++) {
        g_down_epoch[i].store(0, std::memory_order_release);
        g_down_done_epoch[i].store(0, std::memory_order_release);
    }
    for (int i = 0; i < kExpertJobs; i++) {
        g_down_helpers[i] = std::thread(down_helper_loop, i);
    }
    while (g_down_ready.load(std::memory_order_acquire) < kExpertJobs) {
        _mm_pause();
    }
    if (!g_down_shutdown_registered) {
        std::atexit(stop_down_helpers);
        g_down_shutdown_registered = true;
    }
    g_down_started = true;
}

static inline void ensure_down_helpers_started() {
    if (!g_down_started) {
        start_down_helpers();
    }
}

static void stop_gate_up_helpers() {
    if (!g_gate_up_started) return;
    g_gate_up_stop.store(true, std::memory_order_release);
    for (int i = 0; i < kExpertJobs; i++) {
        g_gate_up_epoch[i].fetch_add(1, std::memory_order_release);
    }
    for (int i = 0; i < kExpertJobs; i++) {
        if (g_gate_up_helpers[i].joinable()) {
            g_gate_up_helpers[i].join();
        }
    }
    g_gate_up_started = false;
}

static void gate_up_helper_loop(int job_id) {
    const int cpu_offset = 2 * kExpertJobs;
    if (g_affinity_count > cpu_offset + job_id) {
        bind_current_thread_to_cpu(g_affinity_cpus[cpu_offset + job_id]);
    }
    int seen_epoch = g_gate_up_epoch[job_id].load(std::memory_order_acquire);
    g_gate_up_ready.fetch_add(1, std::memory_order_release);
    while (true) {
        int epoch = g_gate_up_epoch[job_id].load(std::memory_order_acquire);
        while (epoch == seen_epoch &&
               !g_gate_up_stop.load(std::memory_order_acquire)) {
            _mm_pause();
            epoch = g_gate_up_epoch[job_id].load(std::memory_order_acquire);
        }
        if (g_gate_up_stop.load(std::memory_order_acquire)) {
            return;
        }
        seen_epoch = epoch;

        GateUpTask& task = g_gate_up_task[job_id];
        task.h_amax = ffn_gate_up_packed_range(
            task.w_gate, task.w_up, task.s_gate, task.s_up, task.xq,
            task.s_x, task.correction, task.h, task.d_model,
            task.f_begin, task.f_end);
        g_gate_up_done_epoch[job_id].store(seen_epoch,
                                          std::memory_order_release);
    }
}

static void start_gate_up_helpers() {
    if (g_gate_up_started) return;
    collect_affinity_cpus();
    g_gate_up_stop.store(false, std::memory_order_release);
    g_gate_up_ready.store(0, std::memory_order_release);
    for (int i = 0; i < kExpertJobs; i++) {
        g_gate_up_epoch[i].store(0, std::memory_order_release);
        g_gate_up_done_epoch[i].store(0, std::memory_order_release);
    }
    for (int i = 0; i < kExpertJobs; i++) {
        g_gate_up_helpers[i] = std::thread(gate_up_helper_loop, i);
    }
    while (g_gate_up_ready.load(std::memory_order_acquire) < kExpertJobs) {
        _mm_pause();
    }
    if (!g_gate_up_shutdown_registered) {
        std::atexit(stop_gate_up_helpers);
        g_gate_up_shutdown_registered = true;
    }
    g_gate_up_started = true;
}

static inline void ensure_gate_up_helpers_started() {
    if (!g_gate_up_started) {
        start_gate_up_helpers();
    }
}

static MOE_NOINLINE float ffn_gate_up_packed_hybrid(
    int job_id, const int8_t* w_gate, const int8_t* w_up, float s_gate,
    float s_up, const int8_t* xq, float s_x, int32_t correction, float* h,
    int d_model, int d_ff) {
    const int split = ((d_ff / 2) / 16) * 16;
    GateUpTask& task = g_gate_up_task[job_id];
    task.w_gate = w_gate;
    task.w_up = w_up;
    task.xq = xq;
    task.s_gate = s_gate;
    task.s_up = s_up;
    task.s_x = s_x;
    task.correction = correction;
    task.h = h;
    task.d_model = d_model;
    task.d_ff = d_ff;
    task.f_begin = split;
    task.f_end = d_ff;
    task.h_amax = 0.0f;

    const int epoch =
        g_gate_up_epoch[job_id].fetch_add(1, std::memory_order_release) + 1;

    float owner_amax = ffn_gate_up_packed_range(
        w_gate, w_up, s_gate, s_up, xq, s_x, correction, h, d_model,
        0, split);
    while (g_gate_up_done_epoch[job_id].load(std::memory_order_acquire) !=
           epoch) {
        _mm_pause();
    }
    return owner_amax > task.h_amax ? owner_amax : task.h_amax;
}

static MOE_NOINLINE void expert_ffn_packed_hybrid(
    int job_id, const int8_t* w_gate, const int8_t* w_up,
    const int8_t* w_down, float s_gate, float s_up, float s_down,
    const int8_t* xq, float s_x, int32_t x_correction, float out_scale,
    float* out, int d_model, int d_ff) {
    float h[MAX_D_FF];
    float h_amax = ffn_gate_up_packed_hybrid(
        job_id, w_gate, w_up, s_gate, s_up, xq, s_x, x_correction, h,
        d_model, d_ff);

    int8_t hq[MAX_D_FF];
    int32_t sum_hq = 0;
    float s_h = ffn_requantize_with_amax(h, hq, d_ff, h_amax, &sum_hq);

    const int split = ((d_model / 2) / kDownRowBlock) * kDownRowBlock;
    DownTask& task = g_down_task[job_id];
    task.w_down = w_down;
    task.hq = hq;
    task.s_down = s_h * s_down * out_scale;
    task.d_model = d_model;
    task.d_ff = d_ff;
    task.correction = sum_hq * 128;
    task.row_begin = split;
    task.row_end = d_model;
    task.out = out;
    const int epoch =
        g_down_epoch[job_id].fetch_add(1, std::memory_order_release) + 1;

    ffn_down_packed_range(w_down, task.s_down, hq, d_ff,
                          task.correction, out, 0, split);
    while (g_down_done_epoch[job_id].load(std::memory_order_acquire) !=
           epoch) {
        _mm_pause();
    }
}

static void expert_worker_loop(int worker_id) {
    int seen_epoch = g_worker_epoch.load(std::memory_order_acquire);
    g_worker_ready.fetch_add(1, std::memory_order_release);
    while (true) {
        int epoch = g_worker_epoch.load(std::memory_order_acquire);
        while (epoch == seen_epoch &&
               !g_worker_stop.load(std::memory_order_acquire)) {
            _mm_pause();
            epoch = g_worker_epoch.load(std::memory_order_acquire);
        }
        if (g_worker_stop.load(std::memory_order_acquire)) {
            return;
        }
        seen_epoch = epoch;

        if (worker_id < g_worker_top_k) {
            const MoEWeights& w = *g_worker_w;
            const int d_model = g_worker_d_model;
            const int d_ff = g_worker_d_ff;
            const size_t gate_size = (size_t)d_ff * d_model;
            const size_t down_size = (size_t)d_model * d_ff;
            const int e = g_worker_topk_idx[worker_id];
            expert_ffn_packed_hybrid(
                worker_id + 1, g_w_gate_pack + (size_t)e * gate_size,
                g_w_up_pack + (size_t)e * gate_size,
                g_w_down_pack + (size_t)e * down_size, w.s_gate[e],
                w.s_up[e], w.s_down[e], g_worker_xq, g_worker_s_x,
                g_worker_x_correction, g_worker_gate[worker_id],
                g_worker_expert_out[worker_id + 1], d_model, d_ff);
        }
        g_worker_status[worker_id].done_epoch.store(
            seen_epoch, std::memory_order_release);
    }
}

static void start_expert_workers() {
    if (g_worker_started) return;
    collect_affinity_cpus();
    g_worker_stop.store(false, std::memory_order_release);
    g_worker_ready.store(0, std::memory_order_release);
    g_worker_epoch.store(0, std::memory_order_release);
    for (int i = 0; i < MAX_TOP_K; i++) {
        g_worker_status[i].done_epoch.store(0, std::memory_order_release);
    }
    for (int i = 0; i < MAX_TOP_K; i++) {
        g_expert_workers[i] = std::thread(expert_worker_loop, i);
        if (g_affinity_count > i + 1) {
            bind_thread_to_cpu(g_expert_workers[i], g_affinity_cpus[i + 1]);
        }
    }
    while (g_worker_ready.load(std::memory_order_acquire) < MAX_TOP_K) {
        _mm_pause();
    }
    if (!g_worker_shutdown_registered) {
        std::atexit(stop_expert_workers);
        g_worker_shutdown_registered = true;
    }
    g_worker_started = true;
}

static inline void ensure_expert_workers_started() {
    if (!g_worker_started) {
        start_expert_workers();
    }
}

static MOE_NOINLINE void process_one_token_parallel_experts(const float* xt,
                                                            const MoEWeights& w,
                                                            float* yt) {
    const int d_model = w.d_model;
    const int d_ff = w.d_ff;
    const int top_k = w.top_k;

    float s[MAX_NUM_EXPERTS];
    int topk_idx[MAX_TOP_K];
    int8_t xq[MAX_D_MODEL];
    float expert_out[MAX_TOP_K + 1][MAX_D_MODEL];

    compute_router_scores(xt, w, s);
    select_topk_experts(w, s, topk_idx);
    float gate_sum = compute_gate_sum(s, topk_idx, top_k);
    int32_t sum_xq = 0;
    float s_x = quantize_token(xt, xq, d_model, &sum_xq);
    const int32_t x_correction = sum_xq * 128;

    ensure_down_helpers_started();
    ensure_gate_up_helpers_started();
    ensure_expert_workers_started();
    if (!g_main_affinity_set && g_affinity_count > 0) {
        bind_current_thread_to_cpu(g_affinity_cpus[0]);
        g_main_affinity_set = true;
    }
    g_worker_w = &w;
    g_worker_topk_idx = topk_idx;
    g_worker_xq = xq;
    g_worker_s_x = s_x;
    g_worker_x_correction = x_correction;
    g_worker_expert_out = expert_out;
    g_worker_d_model = d_model;
    g_worker_d_ff = d_ff;
    g_worker_top_k = top_k;
    for (int k = 0; k < top_k; k++) {
        g_worker_gate[k] = s[topk_idx[k]] / gate_sum;
    }
    const int epoch =
        g_worker_epoch.fetch_add(1, std::memory_order_release) + 1;

    expert_ffn_packed_hybrid(0, g_sh_gate_pack, g_sh_up_pack,
                                  g_sh_down_pack, w.sh_s_gate, w.sh_s_up,
                                  w.sh_s_down, xq, s_x, x_correction, 1.0f,
                                  expert_out[0], d_model, d_ff);

    for (int i = 0; i < top_k; i++) {
        while (g_worker_status[i].done_epoch.load(
                   std::memory_order_acquire) != epoch) {
            _mm_pause();
        }
    }
    const float* outs[MAX_TOP_K + 1];
    outs[0] = expert_out[0];
    for (int k = 0; k < top_k; k++) {
        outs[k + 1] = expert_out[k + 1];
    }
    int d = 0;
    for (; d + 16 <= d_model; d += 16) {
        __m512 acc = _mm512_loadu_ps(xt + d);
        for (int k = 0; k <= top_k; k++) {
            acc = _mm512_add_ps(acc, _mm512_loadu_ps(outs[k] + d));
        }
        _mm512_storeu_ps(yt + d, acc);
    }
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    if (num_tokens == 1 && w.top_k > 1 && packed_matches(w)) {
        process_one_token_parallel_experts(x, w, y);
        return;
    }
    moe_forward_ref(x, w, y, num_tokens);
}

#undef MOE_NOINLINE
