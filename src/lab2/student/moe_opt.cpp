// Main task: optimize the MoE forward pass.
#include "moe.h"
#include "moe_opt_common.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <immintrin.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace moe_case_s1 {
#define preprocess preprocess_s1
#define moe_forward_optimized moe_forward_optimized_s1
#include "moe_opt_S1.cpp"
#undef preprocess
#undef moe_forward_optimized
}  // namespace moe_case_s1

namespace moe_case_s2 {
#define preprocess preprocess_s2
#define moe_forward_optimized moe_forward_optimized_s2
#include "moe_opt_S2.cpp"
#undef preprocess
#undef moe_forward_optimized
}  // namespace moe_case_s2

namespace moe_case_s3 {
#define preprocess preprocess_s3
#define moe_forward_optimized moe_forward_optimized_s3
#include "moe_opt_S3.cpp"
#undef preprocess
#undef moe_forward_optimized
}  // namespace moe_case_s3

// S4 is the most layout-sensitive kernel; keep it at top level and only rename
// the public hooks so its hot static functions stay close to the standalone
// archive version.
#define preprocess preprocess_s4_impl
#define moe_forward_optimized moe_forward_optimized_s4_impl
#include "moe_opt_S4.cpp"
#undef preprocess
#undef moe_forward_optimized

void preprocess(MoEWeights& w) {
    // Forward picks a kernel at runtime from num_tokens, which preprocess
    // cannot see. Each specialized kernel keeps its own packed buffers sized
    // by the actual model, so run all of them; each self-checks divisibility
    // and packs nothing if its shapes are unsupported.
    preprocess_s4_impl(w);
    moe_case_s1::preprocess_s1(w);
    moe_case_s2::preprocess_s2(w);
    moe_case_s3::preprocess_s3(w);
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    switch (moe_common::classify_forward(w, num_tokens)) {
        case moe_common::KernelFamily::SmallSingleToken:
            moe_case_s1::moe_forward_optimized_s1(x, w, y, num_tokens);
            return;
        case moe_common::KernelFamily::LargeSingleToken:
            moe_case_s2::moe_forward_optimized_s2(x, w, y, num_tokens);
            return;
        case moe_common::KernelFamily::FewExpertBatch:
            moe_case_s3::moe_forward_optimized_s3(x, w, y, num_tokens);
            return;
        case moe_common::KernelFamily::ManyExpertBatch:
            moe_forward_optimized_s4_impl(x, w, y, num_tokens);
            return;
        case moe_common::KernelFamily::Ref:
            moe_forward_ref(x, w, y, num_tokens);
            return;
    }
}
