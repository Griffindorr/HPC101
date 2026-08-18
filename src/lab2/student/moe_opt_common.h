#pragma once

#include "moe.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace moe_common{

constexpr int kFFNGateUpRowBlock = 4;
constexpr int kFFNDownRowBlock = 8;
constexpr int kFFNGateUpColBlock = 64;
constexpr int kFFNDownColBlock = 32;

#if defined( __GNUC__ ) || defined( __clang__ )
#define MOE_COMMON_ALWAYS_INLINE inline __attribute__( ( always_inline ) )
#else
#define MOE_COMMON_ALWAYS_INLINE inline
#endif

enum class KernelFamily {
    Ref,
    SmallSingleToken,
    LargeSingleToken,
    FewExpertBatch,
    ManyExpertBatch,
};

static MOE_COMMON_ALWAYS_INLINE bool has_many_experts( const MoEWeights& w ){
    return w.num_experts >= 128;
}

static MOE_COMMON_ALWAYS_INLINE bool has_small_expert_matrix( const MoEWeights& w ){
    return w.d_model <= 512 && w.d_ff <= 256;
}

static MOE_COMMON_ALWAYS_INLINE KernelFamily classify_forward( const MoEWeights& w, int num_tokens ){
    if( num_tokens > 1 ){
        if( has_many_experts( w ) ) return KernelFamily::ManyExpertBatch;
        return KernelFamily::FewExpertBatch;
    }
    if( has_small_expert_matrix( w ) ) return KernelFamily::SmallSingleToken;
    return KernelFamily::LargeSingleToken;
}

static MOE_COMMON_ALWAYS_INLINE __m512 fast_exp_ps( __m512 x ){
    const __m512 log2e = _mm512_set1_ps( 1.44269504088896341f );
    const __m512 min_x = _mm512_set1_ps( -87.3365447505536f );
    const __m512 max_x = _mm512_set1_ps( 88.3762626647949f );

    x = _mm512_min_ps( _mm512_max_ps( x, min_x ), max_x );
    x = _mm512_mul_ps( x, log2e );

    __m512i n = _mm512_cvt_roundps_epi32( x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC );
    __m512 f = _mm512_sub_ps( x, _mm512_cvtepi32_ps( n ) );

    const __m512 c1 = _mm512_set1_ps( 0.69314718055994531f );
    const __m512 c2 = _mm512_set1_ps( 0.24022650695910040f );
    const __m512 c3 = _mm512_set1_ps( 0.05550410866482159f );

    __m512 p = _mm512_fmadd_ps( f, c3, c2 );
    p = _mm512_fmadd_ps( f, p, c1 );
    p = _mm512_fmadd_ps( f, p, _mm512_set1_ps( 1.0f ) );
    return _mm512_scalef_ps( p, _mm512_cvtepi32_ps( n ) );
}

static MOE_COMMON_ALWAYS_INLINE __m512 router_exp_ps( __m512 x ){
    const __m512 log2e = _mm512_set1_ps( 1.44269504088896341f );
    const __m512 min_x = _mm512_set1_ps( -87.3365447505536f );
    const __m512 max_x = _mm512_set1_ps( 88.3762626647949f );
    x = _mm512_min_ps( _mm512_max_ps( x, min_x ), max_x );
    x = _mm512_mul_ps( x, log2e );
    __m512i n = _mm512_cvt_roundps_epi32( x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC );
    __m512 f = _mm512_sub_ps( x, _mm512_cvtepi32_ps( n ) );
    const __m512 c5_5 = _mm512_set1_ps( 0.00189438436497822764f );
    const __m512 c5_4 = _mm512_set1_ps( 0.00894061236063881233f );
    const __m512 c5_3 = _mm512_set1_ps( 0.0558764837290737746f );
    __m512 p = _mm512_fmadd_ps( f, c5_5, c5_4 );
    p = _mm512_fmadd_ps( f, p, c5_3 );
    p = _mm512_fmadd_ps( f, p, _mm512_set1_ps( 0.240131743625231294f ) );
    p = _mm512_fmadd_ps( f, p, _mm512_set1_ps( 0.693156763387437861f ) );
    p = _mm512_fmadd_ps( f, p, _mm512_set1_ps( 0.999999770600321158f ) );
    return _mm512_scalef_ps( p, _mm512_cvtepi32_ps( n ) );
}

static MOE_COMMON_ALWAYS_INLINE int32_t reduce_add_epi32_256( __m256i v ){
    __m128i lo = _mm256_castsi256_si128( v );
    __m128i hi = _mm256_extracti128_si256( v, 1 );
    __m128i sum = _mm_add_epi32( lo, hi );
    __m128i t = _mm_shuffle_epi32( sum, _MM_SHUFFLE( 1, 0, 3, 2 ) );
    sum = _mm_add_epi32( sum, t );
    t = _mm_shuffle_epi32( sum, _MM_SHUFFLE( 2, 3, 0, 1 ) );
    sum = _mm_add_epi32( sum, t );
    return _mm_cvtsi128_si32( sum );
}

static MOE_COMMON_ALWAYS_INLINE int32_t reduce_add_epi32_512( __m512i v ){
    __m256i lo = _mm512_castsi512_si256( v );
    __m256i hi = _mm512_extracti64x4_epi64( v, 1 );
    return reduce_add_epi32_256( _mm256_add_epi32( lo, hi ) );      /* 复用 256bit */
}

static MOE_COMMON_ALWAYS_INLINE void pack_router_4x16( const float* src, float* dst, int rows, int cols ){
    size_t pos = 0;
    for( int rb = 0; rb < rows; rb += 4 ){
        for( int cb = 0; cb < cols; cb += 16 ){
            for (int r = 0; r < 4; r++ ){
                std::memcpy( dst + pos, src + ( size_t )( rb + r ) * cols + cb, 16 * sizeof( float ) );
                pos += 16;
            }
        }
    }
}

static MOE_COMMON_ALWAYS_INLINE void pack_matrix_u8bias( const int8_t* src, int8_t* dst, int rows, int cols, int row_block, int col_block ){
    size_t pos = 0;
    for( int rb = 0; rb < rows; rb += row_block ){
        for( int cb = 0; cb < cols; cb += col_block ){
            for( int r = 0; r < row_block; r++ ){
                const int8_t* row = src + ( size_t )( rb + r ) * cols + cb;
                for( int c = 0; c < col_block; c++ ){
                    dst[ pos++ ] = ( int8_t )( ( uint8_t )row[ c ] ^ 0x80u );
                }
            }
        }
    }
}

static MOE_COMMON_ALWAYS_INLINE void pack_expert_matrices_u8bias( const int8_t* src, int8_t* dst, int num_experts, int rows, int cols, int row_block, int col_block ){
    const size_t matrix_size = ( size_t )rows * cols;
    for( int e = 0; e < num_experts; e++ ){
        pack_matrix_u8bias( src + ( size_t )e * matrix_size, dst + ( size_t )e * matrix_size, rows, cols, row_block, col_block );
    }
}

static MOE_COMMON_ALWAYS_INLINE void pack_matrix_amx_kx16_vnni_u8bias(
    const int8_t* src, int8_t* dst, int rows, int cols ){
    constexpr int row_block = 16;
    constexpr int k_block = 64;
    size_t pos = 0;
    for( int rb = 0; rb < rows; rb += row_block ){
        for( int cb = 0; cb < cols; cb += k_block ){
            for( int kg = 0; kg < k_block / 4; kg++ ){
                for( int r = 0; r < row_block; r++ ){
                    const int8_t* row = src + ( size_t )rb * cols + cb + ( size_t )r * cols + kg * 4;
                    dst[ pos++ ] = ( int8_t )( ( uint8_t)row[ 0 ] ^ 0x80u );
                    dst[ pos++ ] = ( int8_t )( ( uint8_t)row[ 1 ] ^ 0x80u );
                    dst[ pos++ ] = ( int8_t )( ( uint8_t)row[ 2 ] ^ 0x80u );
                    dst[ pos++ ] = ( int8_t )( ( uint8_t)row[ 3 ] ^ 0x80u );
                }
            }
        }
    }
}

static MOE_COMMON_ALWAYS_INLINE void pack_amx_expert_matrices_u8bias(
    const int8_t* src, int8_t* dst, int num_experts, int rows, int cols) {
    const size_t matrix_size = (size_t)rows * cols;
    for (int e = 0; e < num_experts; e++) {
        pack_matrix_amx_kx16_vnni_u8bias(src + (size_t)e * matrix_size,
                                         dst + (size_t)e * matrix_size, rows,
                                         cols);
    }
}

static MOE_COMMON_ALWAYS_INLINE float compute_gate_sum(const float* s,
                                                       const int* topk_idx,
                                                       int top_k) {
    float gate_sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        gate_sum += s[topk_idx[k]];
    }
    return gate_sum;
}

static MOE_COMMON_ALWAYS_INLINE float quantize_token(const float* xt,
                                                     int8_t* xq, int d_model,
                                                     int32_t* sum_xq) {
    const __m512 sign_mask =
        _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    __m512 maxv = _mm512_setzero_ps();
    for (int d = 0; d < d_model; d += 16) {
        __m512 xv = _mm512_loadu_ps(xt + d);
        maxv = _mm512_max_ps(maxv, _mm512_and_ps(xv, sign_mask));
    }
    float x_amax = _mm512_reduce_max_ps(maxv);

    float s_x = (x_amax > 0.0f) ? x_amax / 127.0f : 1.0f;
    const __m512 inv_sx = _mm512_set1_ps(1.0f / s_x);
    __m512i sumv = _mm512_setzero_si512();
    for (int d = 0; d < d_model; d += 16) {
        __m512 scaled = _mm512_mul_ps(_mm512_loadu_ps(xt + d), inv_sx);
        __m512i q32 = _mm512_cvtps_epi32(scaled);
        sumv = _mm512_add_epi32(sumv, q32);
        __m128i q8 = _mm512_cvtsepi32_epi8(q32);
        _mm_storeu_si128((__m128i*)(xq + d), q8);
    }
    if (sum_xq) {
        *sum_xq = _mm512_reduce_add_epi32(sumv);
    }
    return s_x;
}

static MOE_COMMON_ALWAYS_INLINE float ffn_requantize_with_amax(
    const float* h, int8_t* hq, int d_ff, float h_amax, int32_t* sum_hq) {
    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    const __m512 inv_sh = _mm512_set1_ps(1.0f / s_h);
    __m512i sumv = _mm512_setzero_si512();
    for (int f = 0; f < d_ff; f += 16) {
        __m512 scaled = _mm512_mul_ps(_mm512_loadu_ps(h + f), inv_sh);
        __m512i q32 = _mm512_cvtps_epi32(scaled);
        sumv = _mm512_add_epi32(sumv, q32);
        __m128i q8 = _mm512_cvtsepi32_epi8(q32);
        _mm_storeu_si128((__m128i*)(hq + f), q8);
    }
    if (sum_hq) {
        *sum_hq = _mm512_reduce_add_epi32(sumv);
    }
    return s_h;
}

static MOE_COMMON_ALWAYS_INLINE float ffn_requantize(const float* h,
                                                     int8_t* hq, int d_ff,
                                                     int32_t* sum_hq) {
    const __m512 sign_mask =
        _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    __m512 maxv = _mm512_setzero_ps();
    for (int f = 0; f < d_ff; f += 16) {
        __m512 hv = _mm512_loadu_ps(h + f);
        maxv = _mm512_max_ps(maxv, _mm512_and_ps(hv, sign_mask));
    }
    float h_amax = _mm512_reduce_max_ps(maxv);

    float s_h = (h_amax > 0.0f) ? h_amax / 127.0f : 1.0f;
    const __m512 inv_sh = _mm512_set1_ps(1.0f / s_h);
    __m512i sumv = _mm512_setzero_si512();
    for (int f = 0; f < d_ff; f += 16) {
        __m512 scaled = _mm512_mul_ps(_mm512_loadu_ps(h + f), inv_sh);
        __m512i q32 = _mm512_cvtps_epi32(scaled);
        sumv = _mm512_add_epi32(sumv, q32);
        __m128i q8 = _mm512_cvtsepi32_epi8(q32);
        _mm_storeu_si128((__m128i*)(hq + f), q8);
    }
    if (sum_hq) {
        *sum_hq = _mm512_reduce_add_epi32(sumv);
    }
    return s_h;
}

static MOE_COMMON_ALWAYS_INLINE float ffn_gate_up_packed(
    const int8_t* w_gate, const int8_t* w_up, float s_gate, float s_up,
    const int8_t* xq, float s_x, int32_t correction, float* h, int d_model, int d_ff) {
    const float scale_g = s_x * s_gate;
    const float scale_u = s_x * s_up;
    const int col_chunks = d_model / kFFNGateUpColBlock;
    const int group_stride = col_chunks * kFFNGateUpRowBlock * kFFNGateUpColBlock;

    float vg_buf[MAX_D_FF];
    float vu_buf[MAX_D_FF];

    for (int f = 0; f < d_ff; f += kFFNGateUpRowBlock) {
        __m512i acc_g0 = _mm512_setzero_si512();
        __m512i acc_g1 = _mm512_setzero_si512();
        __m512i acc_g2 = _mm512_setzero_si512();
        __m512i acc_g3 = _mm512_setzero_si512();
        __m512i acc_u0 = _mm512_setzero_si512();
        __m512i acc_u1 = _mm512_setzero_si512();
        __m512i acc_u2 = _mm512_setzero_si512();
        __m512i acc_u3 = _mm512_setzero_si512();
        const int8_t* wg_group = w_gate + (size_t)(f / kFFNGateUpRowBlock) * group_stride;
        const int8_t* wu_group = w_up + (size_t)(f / kFFNGateUpRowBlock) * group_stride;

        for (int cb = 0; cb < col_chunks; cb++) {
            const int d = cb * kFFNGateUpColBlock;
            const size_t block =
                (size_t)cb * kFFNGateUpRowBlock * kFFNGateUpColBlock;
            __m512i xv8 = _mm512_loadu_si512((const __m512i*)(xq + d));

            // Interleave each load with its own dpbusd so a later weight load
            // overlaps the already-issued dot product, hiding load latency.
            __m512i wg0 = _mm512_loadu_si512((const __m512i*)(wg_group + block));
            acc_g0 = _mm512_dpbusd_epi32(acc_g0, wg0, xv8);
            __m512i wu0 = _mm512_loadu_si512((const __m512i*)(wu_group + block));
            acc_u0 = _mm512_dpbusd_epi32(acc_u0, wu0, xv8);

            __m512i wg1 = _mm512_loadu_si512(
                (const __m512i*)(wg_group + block + kFFNGateUpColBlock));
            acc_g1 = _mm512_dpbusd_epi32(acc_g1, wg1, xv8);
            __m512i wu1 = _mm512_loadu_si512(
                (const __m512i*)(wu_group + block + kFFNGateUpColBlock));
            acc_u1 = _mm512_dpbusd_epi32(acc_u1, wu1, xv8);

            __m512i wg2 = _mm512_loadu_si512(
                (const __m512i*)(wg_group + block + 2 * kFFNGateUpColBlock));
            acc_g2 = _mm512_dpbusd_epi32(acc_g2, wg2, xv8);
            __m512i wu2 = _mm512_loadu_si512(
                (const __m512i*)(wu_group + block + 2 * kFFNGateUpColBlock));
            acc_u2 = _mm512_dpbusd_epi32(acc_u2, wu2, xv8);

            __m512i wg3 = _mm512_loadu_si512(
                (const __m512i*)(wg_group + block + 3 * kFFNGateUpColBlock));
            acc_g3 = _mm512_dpbusd_epi32(acc_g3, wg3, xv8);
            __m512i wu3 = _mm512_loadu_si512(
                (const __m512i*)(wu_group + block + 3 * kFFNGateUpColBlock));
            acc_u3 = _mm512_dpbusd_epi32(acc_u3, wu3, xv8);
        }

        vg_buf[f] = (float)(reduce_add_epi32_512(acc_g0) - correction) *
                    scale_g;
        vg_buf[f + 1] =
            (float)(reduce_add_epi32_512(acc_g1) - correction) * scale_g;
        vg_buf[f + 2] =
            (float)(reduce_add_epi32_512(acc_g2) - correction) * scale_g;
        vg_buf[f + 3] =
            (float)(reduce_add_epi32_512(acc_g3) - correction) * scale_g;
        vu_buf[f] = (float)(reduce_add_epi32_512(acc_u0) - correction) *
                    scale_u;
        vu_buf[f + 1] =
            (float)(reduce_add_epi32_512(acc_u1) - correction) * scale_u;
        vu_buf[f + 2] =
            (float)(reduce_add_epi32_512(acc_u2) - correction) * scale_u;
        vu_buf[f + 3] =
            (float)(reduce_add_epi32_512(acc_u3) - correction) * scale_u;
    }

    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 sign_mask =
        _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    __m512 maxv = _mm512_setzero_ps();
    for (int f = 0; f < d_ff; f += 16) {
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

static MOE_COMMON_ALWAYS_INLINE void ffn_down_packed(const int8_t* w_down, float s_down,
                                         const int8_t* hq, int d_model,
                                         int d_ff, int32_t correction,
                                         float* out) {
    const int col_chunks = d_ff / kFFNDownColBlock;
    const int group_stride = col_chunks * kFFNDownRowBlock * kFFNDownColBlock;

    for (int d = 0; d < d_model; d += kFFNDownRowBlock) {
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        __m256i acc4 = _mm256_setzero_si256();
        __m256i acc5 = _mm256_setzero_si256();
        __m256i acc6 = _mm256_setzero_si256();
        __m256i acc7 = _mm256_setzero_si256();
        const int8_t* wd_group =
            w_down + (size_t)(d / kFFNDownRowBlock) * group_stride;

        for (int cb = 0; cb < col_chunks; cb++) {
            const int f = cb * kFFNDownColBlock;
            const size_t block = (size_t)cb * kFFNDownRowBlock * kFFNDownColBlock;
            __m256i hv8 = _mm256_loadu_si256((const __m256i*)(hq + f));

            __m256i wd0 = _mm256_loadu_si256((const __m256i*)(wd_group + block));
            acc0 = _mm256_dpbusd_epi32(acc0, wd0, hv8);

            __m256i wd1 = _mm256_loadu_si256(
                (const __m256i*)(wd_group + block + kFFNDownColBlock));
            acc1 = _mm256_dpbusd_epi32(acc1, wd1, hv8);

            __m256i wd2 = _mm256_loadu_si256(
                (const __m256i*)(wd_group + block + 2 * kFFNDownColBlock));
            acc2 = _mm256_dpbusd_epi32(acc2, wd2, hv8);

            __m256i wd3 = _mm256_loadu_si256(
                (const __m256i*)(wd_group + block + 3 * kFFNDownColBlock));
            acc3 = _mm256_dpbusd_epi32(acc3, wd3, hv8);

            __m256i wd4 = _mm256_loadu_si256(
                (const __m256i*)(wd_group + block + 4 * kFFNDownColBlock));
            acc4 = _mm256_dpbusd_epi32(acc4, wd4, hv8);

            __m256i wd5 = _mm256_loadu_si256(
                (const __m256i*)(wd_group + block + 5 * kFFNDownColBlock));
            acc5 = _mm256_dpbusd_epi32(acc5, wd5, hv8);

            __m256i wd6 = _mm256_loadu_si256(
                (const __m256i*)(wd_group + block + 6 * kFFNDownColBlock));
            acc6 = _mm256_dpbusd_epi32(acc6, wd6, hv8);

            __m256i wd7 = _mm256_loadu_si256(
                (const __m256i*)(wd_group + block + 7 * kFFNDownColBlock));
            acc7 = _mm256_dpbusd_epi32(acc7, wd7, hv8);
        }

        out[d] = (float)(reduce_add_epi32_256(acc0) - correction) * s_down;
        out[d + 1] = (float)(reduce_add_epi32_256(acc1) - correction) * s_down;
        out[d + 2] = (float)(reduce_add_epi32_256(acc2) - correction) * s_down;
        out[d + 3] = (float)(reduce_add_epi32_256(acc3) - correction) * s_down;
        out[d + 4] = (float)(reduce_add_epi32_256(acc4) - correction) * s_down;
        out[d + 5] = (float)(reduce_add_epi32_256(acc5) - correction) * s_down;
        out[d + 6] = (float)(reduce_add_epi32_256(acc6) - correction) * s_down;
        out[d + 7] = (float)(reduce_add_epi32_256(acc7) - correction) * s_down;
    }
}

static MOE_COMMON_ALWAYS_INLINE void expert_ffn_packed(
    const int8_t* w_gate, const int8_t* w_up, const int8_t* w_down,
    float s_gate, float s_up, float s_down, const int8_t* xq, float s_x,
    int32_t x_correction, float out_scale, float* out, int d_model, int d_ff) {
    float h[MAX_D_FF];
    float h_amax = ffn_gate_up_packed(w_gate, w_up, s_gate, s_up, xq, s_x,
                                      x_correction, h, d_model, d_ff);

    int8_t hq[MAX_D_FF];
    int32_t sum_hq = 0;
    float s_h = ffn_requantize_with_amax(h, hq, d_ff, h_amax, &sum_hq);

    ffn_down_packed(w_down, s_h * s_down * out_scale, hq, d_model, d_ff,
                    sum_hq * 128, out);
}


}  // namespace moe_common

#undef MOE_COMMON_ALWAYS_INLINE
