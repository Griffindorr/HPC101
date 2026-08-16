#include "prolongrestrict.h"

#include "fmisc.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// ==========================================
// 1. Constants & Helper Functions
// ==========================================

// Prolongation Coefficients (5th order)

__constant__ double C_PROLONG[6] = {
    77.0 / 8192.0,    // C1
    -693.0 / 8192.0,  // C2
    3465.0 / 4096.0,  // C3
    1155.0 / 4096.0,  // C4
    -495.0 / 8192.0,  // C5
    63.0 / 8192.0     // C6
};

// Restriction Coefficients
__constant__ double C_RESTRICT[3] = {
    3.0 / 256.0,      // C1
    -25.0 / 256.0,    // C2
    75.0 / 128.0      // C3
};

// Fortran IDINT equivalent
__device__ int d_idint(double a) {
    if (fabs(a) < 1.0) return 0;
    return (int)(a);
}

// Memory Access Helper: Column-Major (Fortran Layout)
// i, j, k are 0-based indices
__device__ __forceinline__ int get_col_major_idx(int i, int j, int k, int nx, int ny, int nz) {
    return k * (nx * ny) + j * nx + i;
}

// ==========================================
// 2. Prolongation Device Function
// ==========================================

// Calculate prolonged value for a single point (i, j, k) on the FINE grid.
// i, j, k are 0-based indices [0, extf-1]
__device__ void d_prolong3_device(
    int i, int j, int k, 
    const int* extc, const double* func,
    const int* extf, double* funf, 
    const int* lbf, const int* lbc,
    const int* imino, const int* imaxo,
    const double* SoA, int Symmetry
) {
    // Geometry & alignment (lbf, lbc, imino/imaxo) is per-launch constant and is
    // precomputed on the host by gpu_prolong3_launch.
    int jmino = imino[1];
    int jmaxo = imaxo[1];
    int kmino = imino[2];
    int kmaxo = imaxo[2];

    // Convert to 1-based for comparison
    int i_1b = i + 1;
    int j_1b = j + 1;
    int k_1b = k + 1;

    if (i_1b < imino[0] || i_1b > imaxo[0] || j_1b < jmino || j_1b > jmaxo || k_1b < kmino || k_1b > kmaxo) {
        return; 
    }

    // --- 2. Index Mapping ---
    
    // Global index (Still effectively 1-based logic for parity check)
    // Fortran: ii = i + lbf - 1. Since our i is 0-based, ii = (i+1) + lbf - 1 = i + lbf
    int ii = i + lbf[0];
    int jj = j + lbf[1];
    int kk = k + lbf[2];

    // Coarse Index Calculation
    // Fortran: cxI = i; cxI = (cxI + lbf - 1)/2; cxI = cxI - lbc + 1
    // CUDA: use i_1b for 'i'
    int cxI_i = (i_1b + lbf[0] - 1) / 2 - lbc[0] + 1;
    int cxI_j = (j_1b + lbf[1] - 1) / 2 - lbc[1] + 1;
    int cxI_k = (k_1b + lbf[2] - 1) / 2 - lbc[2] + 1;

    // Parity Checks (Even/Odd)
    bool k_even = ((kk / 2) * 2 == kk);
    bool j_even = ((jj / 2) * 2 == jj);
    bool i_even = ((ii / 2) * 2 == ii);

    // --- 3. Interpolation ---
    double tmp2[6][6];
    double tmp1[6];
    // Interior fast path: all 36 coarse reads stay strictly inside the coarse
    // grid (1-based indices in [1, extc]) so no symmetry reflection or bounds
    // check is needed. This condition is point-invariant, so the branch below
    // is uniform across the whole interpolation loop.
    const bool interior =
        (cxI_i - 2 >= 1 && cxI_i + 3 <= extc[0] &&
         cxI_j - 2 >= 1 && cxI_j + 3 <= extc[1] &&
         cxI_k - 2 >= 1 && cxI_k + 3 <= extc[2]);

    auto gv = [&](int ic, int jc, int kc) -> double {
        if (interior) {
            return func[get_col_major_idx(ic - 1, jc - 1, kc - 1,
                                          extc[0], extc[1], extc[2])];
        }
        return d_symmetry_bd_1b(3, extc, func, ic, jc, kc, SoA);
    };

    // Z-Direction Interpolation
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            int cur_ic = cxI_i - 2 + n;
            int cur_jc = cxI_j - 2 + m;
            
            double val = 0.0;
            // 1-based indices passed to d_get_sym_val
            if (k_even) {
                val += C_PROLONG[0] * gv(cur_ic, cur_jc, cxI_k - 2);
                val += C_PROLONG[1] * gv(cur_ic, cur_jc, cxI_k - 1);
                val += C_PROLONG[2] * gv(cur_ic, cur_jc, cxI_k    );
                val += C_PROLONG[3] * gv(cur_ic, cur_jc, cxI_k + 1);
                val += C_PROLONG[4] * gv(cur_ic, cur_jc, cxI_k + 2);
                val += C_PROLONG[5] * gv(cur_ic, cur_jc, cxI_k + 3);
            } else {
                val += C_PROLONG[5] * gv(cur_ic, cur_jc, cxI_k - 2);
                val += C_PROLONG[4] * gv(cur_ic, cur_jc, cxI_k - 1);
                val += C_PROLONG[3] * gv(cur_ic, cur_jc, cxI_k    );
                val += C_PROLONG[2] * gv(cur_ic, cur_jc, cxI_k + 1);
                val += C_PROLONG[1] * gv(cur_ic, cur_jc, cxI_k + 2);
                val += C_PROLONG[0] * gv(cur_ic, cur_jc, cxI_k + 3);
            }
            tmp2[m][n] = val;
        }
    }

    // Y-Direction Interpolation
    for (int n = 0; n < 6; n++) {
        double val = 0.0;
        if (j_even) {
            val += C_PROLONG[0] * tmp2[0][n] + C_PROLONG[1] * tmp2[1][n] + C_PROLONG[2] * tmp2[2][n] +
                   C_PROLONG[3] * tmp2[3][n] + C_PROLONG[4] * tmp2[4][n] + C_PROLONG[5] * tmp2[5][n];
        } else {
            val += C_PROLONG[5] * tmp2[0][n] + C_PROLONG[4] * tmp2[1][n] + C_PROLONG[3] * tmp2[2][n] +
                   C_PROLONG[2] * tmp2[3][n] + C_PROLONG[1] * tmp2[4][n] + C_PROLONG[0] * tmp2[5][n];
        }
        tmp1[n] = val;
    }

    // X-Direction Interpolation
    double final_val = 0.0;
    if (i_even) {
        final_val += C_PROLONG[0] * tmp1[0] + C_PROLONG[1] * tmp1[1] + C_PROLONG[2] * tmp1[2] +
                     C_PROLONG[3] * tmp1[3] + C_PROLONG[4] * tmp1[4] + C_PROLONG[5] * tmp1[5];
    } else {
        final_val += C_PROLONG[5] * tmp1[0] + C_PROLONG[4] * tmp1[1] + C_PROLONG[3] * tmp1[2] +
                     C_PROLONG[2] * tmp1[3] + C_PROLONG[1] * tmp1[4] + C_PROLONG[0] * tmp1[5];
    }

    // Write Output (0-based index)
    int out_idx = get_col_major_idx(i, j, k, extf[0], extf[1], extf[2]);
    funf[out_idx] = final_val;
}

// ==========================================
// 3. Restriction Device Function
// ==========================================

// Calculate restricted value for a single point (i, j, k) on the COARSE grid.
// i, j, k are 0-based indices [0, extc-1] (conceptually within the valid restricted range)
__device__ void d_restrict3_device(
    int i, int j, int k, 
    const int* extc, double* func, // func is output
    const int* extf, const double* funf, // funf is input
    const int* lbf, const int* lbc,
    const int* imino, const int* imaxo,
    const double* SoA, int Symmetry
) {
    // Geometry & alignment (lbf, lbc, imino/imaxo) is per-launch constant and is
    // precomputed on the host by gpu_restrict3_launch.
    int jmino = imino[1];
    int jmaxo = imaxo[1];
    int kmino = imino[2];
    int kmaxo = imaxo[2];

    int i_1b = i + 1;
    int j_1b = j + 1;
    int k_1b = k + 1;

    if (i_1b < imino[0] || i_1b > imaxo[0] || j_1b < jmino || j_1b > jmaxo || k_1b < kmino || k_1b > kmaxo) {
        return;
    }

    // --- 2. Index Mapping ---
    
    // Coarse to Fine mapping
    // Fortran: cxI = i; cxI = 2*(cxI+lbc-1) - 1; cxI = cxI - lbf + 1
    // CUDA: use i_1b for 'i'
    int if_fine = 2 * (i_1b + lbc[0] - 1) - 1 - lbf[0] + 1;
    int jf_fine = 2 * (j_1b + lbc[1] - 1) - 1 - lbf[1] + 1;
    int kf_fine = 2 * (k_1b + lbc[2] - 1) - 1 - lbf[2] + 1;

    // --- 3. Restriction ---
    double tmp2[6][6];
    double tmp1[6];
    // Interior fast path: all fine-grid reads (kf_fine-2 .. kf_fine+3, likewise
    // if_fine/jf_fine) stay strictly inside the fine grid (1-based [1, extf]) so
    // no symmetry reflection or bounds check is needed.
    const bool interior =
        (if_fine - 2 >= 1 && if_fine + 3 <= extf[0] &&
         jf_fine - 2 >= 1 && jf_fine + 3 <= extf[1] &&
         kf_fine - 2 >= 1 && kf_fine + 3 <= extf[2]);

    auto gv = [&](int ic, int jc, int kc) -> double {
        if (interior) {
            return funf[get_col_major_idx(ic - 1, jc - 1, kc - 1,
                                          extf[0], extf[1], extf[2])];
        }
        return d_symmetry_bd_1b(2, extf, funf, ic, jc, kc, SoA);
    };

    // Z-Direction Restriction
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            int cur_jf = jf_fine - 2 + m;
            int cur_if = if_fine - 2 + n;
            
            double val = 0.0;
            // Ord=2 passed to symmetry_bd as per Fortran restrict3
            // Indices: -2, -1, 0, 1, 2, 3 relative to fine center
            val += C_RESTRICT[0] * (
                gv(cur_if, cur_jf, kf_fine - 2) + 
                gv(cur_if, cur_jf, kf_fine + 3)
            );
            val += C_RESTRICT[1] * (
                gv(cur_if, cur_jf, kf_fine - 1) + 
                gv(cur_if, cur_jf, kf_fine + 2)
            );
            val += C_RESTRICT[2] * (
                gv(cur_if, cur_jf, kf_fine    ) + 
                gv(cur_if, cur_jf, kf_fine + 1)
            );
            
            tmp2[m][n] = val;
        }
    }

    // Y-Direction Restriction
    for (int n = 0; n < 6; n++) {
        double val = 0.0;
        val += C_RESTRICT[0] * (tmp2[0][n] + tmp2[5][n]);
        val += C_RESTRICT[1] * (tmp2[1][n] + tmp2[4][n]);
        val += C_RESTRICT[2] * (tmp2[2][n] + tmp2[3][n]);
        tmp1[n] = val;
    }

    // X-Direction Restriction
    double final_val = 0.0;
    final_val += C_RESTRICT[0] * (tmp1[0] + tmp1[5]);
    final_val += C_RESTRICT[1] * (tmp1[1] + tmp1[4]);
    final_val += C_RESTRICT[2] * (tmp1[2] + tmp1[3]);

    // Write Output (0-based index)
    int out_idx = get_col_major_idx(i, j, k, extc[0], extc[1], extc[2]);
    func[out_idx] = final_val;
}

// ++++++++++++++ Kernel Implementation ++++++++++++++
// ---------------------------------------------------------
// 1. 直接面向显存的单任务 Prolong Kernel
// ---------------------------------------------------------
__global__ void __launch_bounds__(256, 3) prolong3_kernel(
    int ni, int nj, int nk,
    int i_start, int j_start, int k_start,
    int lbc0, int lbc1, int lbc2,
    int lbf0, int lbf1, int lbf2,
    int extc0, int extc1, int extc2,
    int extf0, int extf1, int extf2,
    const double* __restrict__ d_src_c,
    double* __restrict__ d_dst_f,
    double SoA0, double SoA1, double SoA2,
    int Symmetry
) {
    // Shared Memory 10x10x8 = 800 doubles; [Z][Y][X] layout.
    // Symmetry reflection is applied during the cooperative load (d_symmetry_bd_0b),
    // so the compute phase is fully branch-free.
    __shared__ double smem[8][10][10];

    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    int block_i_start = i_start + blockIdx.x * blockDim.x;
    int block_j_start = j_start + blockIdx.y * blockDim.y;
    int block_k_start = k_start + blockIdx.z * blockDim.z;

    int base_c_i = (block_i_start + lbf0) / 2 - lbc0 - 2;
    int base_c_j = (block_j_start + lbf1) / 2 - lbc1 - 2;
    int base_c_k = (block_k_start + lbf2) / 2 - lbc2 - 2;

    for (int idx = tid; idx < 800; idx += 256) {
        int loc_k = idx / 100;
        int rem   = idx % 100;
        int loc_j = rem / 10;
        int loc_i = rem % 10;
        smem[loc_k][loc_j][loc_i] = d_symmetry_bd_0b(
            3, extc0, extc1, extc2, d_src_c,
            base_c_i + loc_i, base_c_j + loc_j, base_c_k + loc_k,
            SoA0, SoA1, SoA2
        );
    }
    __syncthreads();

    int i_local = blockIdx.x * blockDim.x + threadIdx.x;
    int j_local = blockIdx.y * blockDim.y + threadIdx.y;
    int k_local = blockIdx.z * blockDim.z + threadIdx.z;

    if (i_local >= ni || j_local >= nj || k_local >= nk) return;

    int i = i_start + i_local;
    int j = j_start + j_local;
    int k = k_start + k_local;

    int ii = i + lbf0;
    int jj = j + lbf1;
    int kk = k + lbf2;

    int cxI_i_0b = (i + lbf0) / 2 - lbc0;
    int cxI_j_0b = (j + lbf1) / 2 - lbc1;
    int cxI_k_0b = (k + lbf2) / 2 - lbc2;

    bool k_even = ((kk / 2) * 2 == kk);
    bool j_even = ((jj / 2) * 2 == jj);
    bool i_even = ((ii / 2) * 2 == ii);

    int smem_i_base = cxI_i_0b - 2 - base_c_i;
    int smem_j_base = cxI_j_0b - 2 - base_c_j;
    int smem_k_base = cxI_k_0b - 2 - base_c_k;

    double tmp2[6][6];
    double tmp1[6];

    // Z方向插值
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            int cur_i_smem = smem_i_base + n;
            int cur_j_smem = smem_j_base + m;
            double val = 0.0;
            #pragma unroll
            for (int step = 0; step < 6; step++) {
                int c_idx = k_even ? step : (5 - step);
                val += C_PROLONG[c_idx] * smem[smem_k_base + step][cur_j_smem][cur_i_smem];
            }
            tmp2[m][n] = val;
        }
    }

    // Y方向插值
    for (int n = 0; n < 6; n++) {
        double val = 0.0;
        #pragma unroll
        for (int step = 0; step < 6; step++) {
            int c_idx = j_even ? step : (5 - step);
            val += C_PROLONG[c_idx] * tmp2[step][n];
        }
        tmp1[n] = val;
    }

    // X方向插值
    double final_val = 0.0;
    #pragma unroll
    for (int step = 0; step < 6; step++) {
        int c_idx = i_even ? step : (5 - step);
        final_val += C_PROLONG[c_idx] * tmp1[step];
    }

    int out_idx = k * (extf0 * extf1) + j * extf0 + i;
    d_dst_f[out_idx] = final_val;
}


__global__ void __launch_bounds__(256, 3) restrict3_kernel(
    int ni, int nj, int nk,
    int i_start, int j_start, int k_start,
    int lbc0, int lbc1, int lbc2,
    int lbf0, int lbf1, int lbf2,
    int extc0, int extc1, int extc2,
    int extf0, int extf1, int extf2,
    const double* __restrict__ d_src_f,
    double* __restrict__ d_dst_c,
    double SoA0, double SoA1, double SoA2,
    int Symmetry
) {
    // Shared Memory 20x20x12 = 4800 doubles = 37.5 KB; [Z][Y][X] layout.
    // Symmetry reflection applied during the cooperative load (d_symmetry_bd_0b),
    // so the compute phase is fully branch-free.
    __shared__ double smem[12][20][20];

    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    int block_i_start = i_start + blockIdx.x * blockDim.x;
    int block_j_start = j_start + blockIdx.y * blockDim.y;
    int block_k_start = k_start + blockIdx.z * blockDim.z;

    int base_f_i = 2 * (block_i_start + lbc0) - lbf0 - 3;
    int base_f_j = 2 * (block_j_start + lbc1) - lbf1 - 3;
    int base_f_k = 2 * (block_k_start + lbc2) - lbf2 - 3;

    for (int idx = tid; idx < 4800; idx += 256) {
        int loc_k = idx / 400;
        int rem   = idx % 400;
        int loc_j = rem / 20;
        int loc_i = rem % 20;
        smem[loc_k][loc_j][loc_i] = d_symmetry_bd_0b(
            2, extf0, extf1, extf2, d_src_f,
            base_f_i + loc_i, base_f_j + loc_j, base_f_k + loc_k,
            SoA0, SoA1, SoA2
        );
    }
    __syncthreads();

    int i_local = blockIdx.x * blockDim.x + threadIdx.x;
    int j_local = blockIdx.y * blockDim.y + threadIdx.y;
    int k_local = blockIdx.z * blockDim.z + threadIdx.z;

    if (i_local >= ni || j_local >= nj || k_local >= nk) return;

    int i = i_start + i_local;
    int j = j_start + j_local;
    int k = k_start + k_local;

    int smem_i_start = 2 * threadIdx.x;
    int smem_j_start = 2 * threadIdx.y;
    int smem_k_start = 2 * threadIdx.z;

    double tmp2[6][6];
    double tmp1[6];

    // Z-Direction Restriction
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
            int cur_j_smem = smem_j_start + m;
            int cur_i_smem = smem_i_start + n;
            double val = 0.0;
            val += C_RESTRICT[0] * (
                smem[smem_k_start][cur_j_smem][cur_i_smem] +
                smem[smem_k_start + 5][cur_j_smem][cur_i_smem]
            );
            val += C_RESTRICT[1] * (
                smem[smem_k_start + 1][cur_j_smem][cur_i_smem] +
                smem[smem_k_start + 4][cur_j_smem][cur_i_smem]
            );
            val += C_RESTRICT[2] * (
                smem[smem_k_start + 2][cur_j_smem][cur_i_smem] +
                smem[smem_k_start + 3][cur_j_smem][cur_i_smem]
            );
            tmp2[m][n] = val;
        }
    }

    // Y-Direction Restriction
    #pragma unroll
    for (int n = 0; n < 6; n++) {
        double val = 0.0;
        val += C_RESTRICT[0] * (tmp2[0][n] + tmp2[5][n]);
        val += C_RESTRICT[1] * (tmp2[1][n] + tmp2[4][n]);
        val += C_RESTRICT[2] * (tmp2[2][n] + tmp2[3][n]);
        tmp1[n] = val;
    }

    // X-Direction Restriction
    double final_val = 0.0;
    final_val += C_RESTRICT[0] * (tmp1[0] + tmp1[5]);
    final_val += C_RESTRICT[1] * (tmp1[1] + tmp1[4]);
    final_val += C_RESTRICT[2] * (tmp1[2] + tmp1[3]);

    int out_idx = k * (extc0 * extc1) + j * extc0 + i;
    d_dst_c[out_idx] = final_val;
}



// ---------------------------------------------------------
// 3. Host 端启动接口
// ---------------------------------------------------------
void gpu_prolong3_launch(
    cudaStream_t stream,
    const double* d_src_c, double* d_dst_f,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
) {
    double CD[3], FD[3], base[3];
    for(int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            // 修正：使用 std::trunc 完美对齐 Fortran 的 idint (向零取整)
            int j_val = (int)std::trunc((llbc[d] - llbf[d]) / FD[d] + 0.4);
            if ((j_val / 2) * 2 == j_val) base[d] = llbf[d];
            else base[d] = llbf[d] - CD[d] / 2.0;
        }
    }

    int i_start, i_end, j_start, j_end, k_start, k_end;
    int lbf_p[3], lbc_p[3], lbp_p[3], ubp_p[3], imino[3], imaxo[3];
    for(int d = 0; d < 3; d++) {
        // 修正：使用 std::trunc（与 device 端 d_idint 对齐）
        lbp_p[d] = (int)std::trunc((llbt[d] - base[d]) / FD[d] + 0.4) + 1;
        ubp_p[d] = (int)std::trunc((uubt[d] - base[d]) / FD[d] + 0.4);
        lbf_p[d] = (int)std::trunc((llbf[d] - base[d]) / FD[d] + 0.4) + 1;
        lbc_p[d] = (int)std::trunc((llbc[d] - base[d]) / CD[d] + 0.4) + 1;
        
        if (d == 0) { i_start = lbp_p[0] - lbf_p[0]; i_end = ubp_p[0] - lbf_p[0]; }
        if (d == 1) { j_start = lbp_p[1] - lbf_p[1]; j_end = ubp_p[1] - lbf_p[1]; }
        if (d == 2) { k_start = lbp_p[2] - lbf_p[2]; k_end = ubp_p[2] - lbf_p[2]; }

        imino[d] = lbp_p[d] - lbf_p[d] + 1;
        imaxo[d] = ubp_p[d] - lbf_p[d] + 1;
    }

    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;

    if (ni <= 0 || nj <= 0 || nk <= 0) return; // 剔除空操作

    dim3 block3(8, 8, 4);
    dim3 grid3((ni + 7) / 8, (nj + 7) / 8, (nk + 3) / 4);

    prolong3_kernel<<<grid3, block3, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        lbc_p[0], lbc_p[1], lbc_p[2],
        lbf_p[0], lbf_p[1], lbf_p[2],
        extc[0], extc[1], extc[2],
        extf[0], extf[1], extf[2],
        d_src_c, d_dst_f,
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );
}

void gpu_restrict3_launch(
    cudaStream_t stream,
    const double* d_src_f, double* d_dst_c,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
) {
    double CD[3], FD[3], base[3];
    for(int d = 0; d < 3; d++) {
        CD[d] = (uubc[d] - llbc[d]) / (double)extc[d];
        FD[d] = (uubf[d] - llbf[d]) / (double)extf[d];
        if (llbc[d] <= llbf[d]) {
            base[d] = llbc[d];
        } else {
            int j_val = (int)std::trunc((llbc[d] - llbf[d]) / FD[d] + 0.4);
            if ((j_val / 2) * 2 == j_val) base[d] = llbf[d];
            else base[d] = llbf[d] - CD[d] / 2.0;
        }
    }

    int i_start, i_end, j_start, j_end, k_start, k_end;
    int lbf_p[3], lbc_p[3], lbr_p[3], ubr_p[3], imino[3], imaxo[3];
    for(int d = 0; d < 3; d++) {
        lbr_p[d] = (int)std::trunc((llbt[d] - base[d]) / CD[d] + 0.4) + 1;
        ubr_p[d] = (int)std::trunc((uubt[d] - base[d]) / CD[d] + 0.4);
        lbf_p[d] = (int)std::trunc((llbf[d] - base[d]) / FD[d] + 0.4) + 1;
        lbc_p[d] = (int)std::trunc((llbc[d] - base[d]) / CD[d] + 0.4) + 1;
        
        if (d == 0) { i_start = lbr_p[0] - lbc_p[0]; i_end = ubr_p[0] - lbc_p[0]; }
        if (d == 1) { j_start = lbr_p[1] - lbc_p[1]; j_end = ubr_p[1] - lbc_p[1]; }
        if (d == 2) { k_start = lbr_p[2] - lbc_p[2]; k_end = ubr_p[2] - lbc_p[2]; }

        imino[d] = lbr_p[d] - lbc_p[d] + 1;
        imaxo[d] = ubr_p[d] - lbc_p[d] + 1;
    }

    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;

    if (ni <= 0 || nj <= 0 || nk <= 0) return;

    dim3 block3(8, 8, 4);
    dim3 grid3((ni + 7) / 8, (nj + 7) / 8, (nk + 3) / 4);

    restrict3_kernel<<<grid3, block3, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        lbc_p[0], lbc_p[1], lbc_p[2],
        lbf_p[0], lbf_p[1], lbf_p[2],
        extc[0], extc[1], extc[2],
        extf[0], extf[1], extf[2],
        d_src_f, d_dst_c,
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );
}
