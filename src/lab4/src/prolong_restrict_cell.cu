#include "prolongrestrict.h"

#include "fmisc.h"
#include "fmisc_gpu.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

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

__global__ void prolong3_kernel(
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
    // ==========================================
    // 1. Shared Memory 声明 (10x10x8 = 800 个 double)
    // 对应当前 Block 需要的所有粗网格数据 + 幽灵区
    // ==========================================
    __shared__ double smem[8][10][10]; // 顺序: [Z][Y][X] 优化连续内存访问

    // 1D Block 内部线程 ID，用于协同加载
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    
    // 计算当前 Block 在细网格全局空间的起始索引
    int block_i_start = i_start + blockIdx.x * blockDim.x;
    int block_j_start = j_start + blockIdx.y * blockDim.y;
    int block_k_start = k_start + blockIdx.z * blockDim.z;

    // 计算当前 Block 需要加载的粗网格基准点（包含了 -2 的幽灵区偏移）
    int base_c_i = (block_i_start + lbf0) / 2 - lbc0 - 2;
    int base_c_j = (block_j_start + lbf1) / 2 - lbc1 - 2;
    int base_c_k = (block_k_start + lbf2) / 2 - lbc2 - 2;

    // ==========================================
    // 2. 协同加载阶段 (Cooperative Load)
    // 256 个线程去搬运 800 个数据，每个线程大概搬运 3~4 个
    // ==========================================
    for (int idx = tid; idx < 800; idx += 256) {
        // 1D 索引还原为 Tile 内部的 3D 偏移
        int loc_k = idx / 100;
        int rem   = idx % 100;
        int loc_j = rem / 10;
        int loc_i = rem % 10;

        // 调用对称取值函数，直接从 Global Memory 存入 Shared Memory
        smem[loc_k][loc_j][loc_i] = d_symmetry_bd_0b(
            3, extc0, extc1, extc2, d_src_c,
            base_c_i + loc_i, base_c_j + loc_j, base_c_k + loc_k,
            SoA0, SoA1, SoA2
        );
    }

    __syncthreads();

    // ==========================================
    // 3. 计算阶段 (纯 Shared Memory 读取)
    // ==========================================
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

    // 计算当前线程需要的粗网格点在 Shared Memory 中的起始索引
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

    // 写入细网格显存
    int out_idx = k * (extf0 * extf1) + j * extf0 + i;
    d_dst_f[out_idx] = final_val;
}

void gpu_prolong3_launch(
    cudaStream_t stream,
    const double* d_src_c, double* d_dst_f,
    const double* llbc, const double* uubc, const int* extc,
    const double* llbf, const double* uubf, const int* extf,
    const double* llbt, const double* uubt,
    const double* SoA, int Symmetry
) {
    double CD[3], FD[3], base[3];
    int lbc[3], lbf[3], lbp[3], ubp[3];

    // 1. 几何与对齐预计算 (仅在 Host 端执行一次)
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

        lbf[d] = (int)std::trunc((llbf[d] - base[d]) / FD[d] + 0.4) + 1;
        lbc[d] = (int)std::trunc((llbc[d] - base[d]) / CD[d] + 0.4) + 1;
        lbp[d] = (int)std::trunc((llbt[d] - base[d]) / FD[d] + 0.4) + 1;
        ubp[d] = (int)std::trunc((uubt[d] - base[d]) / FD[d] + 0.4);
    }

    // 2. 计算 0-based 循环起始与终止范围
    int i_start = lbp[0] - lbf[0];
    int i_end   = ubp[0] - lbf[0];
    int j_start = lbp[1] - lbf[1];
    int j_end   = ubp[1] - lbf[1];
    int k_start = lbp[2] - lbf[2];
    int k_end   = ubp[2] - lbf[2];

    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;

    if (ni <= 0 || nj <= 0 || nk <= 0) return; 

    dim3 block(8, 8, 4); // 256 threads, 适应常见的三维幽灵区形状
    dim3 grid((ni + block.x - 1) / block.x,
              (nj + block.y - 1) / block.y,
              (nk + block.z - 1) / block.z);

    prolong3_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, 
        i_start, j_start, k_start,
        lbc[0], lbc[1], lbc[2],
        lbf[0], lbf[1], lbf[2],
        extc[0], extc[1], extc[2],
        extf[0], extf[1], extf[2],
        d_src_c, d_dst_f,
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );
}

__global__ void restrict3_kernel(
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
    // ==========================================
    // 1. Shared Memory 声明 (20x20x12 = 4800 个 double = 37.5 KB)
    // 对应当前 Block 需要的所有细网格数据 + 幽灵区
    // ==========================================
    __shared__ double smem[12][20][20]; // [Z][Y][X] 布局

    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    // 计算当前 Block 对应的全局粗网格起始索引
    int block_i_start = i_start + blockIdx.x * blockDim.x;
    int block_j_start = j_start + blockIdx.y * blockDim.y;
    int block_k_start = k_start + blockIdx.z * blockDim.z;

    // 计算当前 Block 需要加载的细网格基准点 (已包含 -2 的左侧幽灵区偏移)
    int base_f_i = 2 * (block_i_start + lbc0) - lbf0 - 3;
    int base_f_j = 2 * (block_j_start + lbc1) - lbf1 - 3;
    int base_f_k = 2 * (block_k_start + lbc2) - lbf2 - 3;

    // ==========================================
    // 2. 协同加载阶段 (256个线程搬运 4800 个数据，每人约 18-19 个)
    // ==========================================
    for (int idx = tid; idx < 4800; idx += 256) {
        // 1D 索引还原为 Tile 内部的 3D 偏移 (12 * 20 * 20)
        int loc_k = idx / 400; // 20 * 20 = 400
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

    // ==========================================
    // 3. 计算阶段
    // ==========================================
    int i_local = blockIdx.x * blockDim.x + threadIdx.x;
    int j_local = blockIdx.y * blockDim.y + threadIdx.y;
    int k_local = blockIdx.z * blockDim.z + threadIdx.z;

    // 边界检查必须放在 syncthreads 之后
    if (i_local >= ni || j_local >= nj || k_local >= nk) return;

    int i = i_start + i_local;
    int j = j_start + j_local;
    int k = k_start + k_local;

    // 由于我们在加载 SMEM 时，base 已经减去了 3 (即包含了 -2 的幽灵区偏移)
    // 所以 SMEM 内部的局部起始坐标就是相对于细网格中心的相对跨度
    // 粗网格步长为 1 时，对应的细网格步长为 2
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
            // 完美对称的限制操作，彻底消除 Global Memory 访问
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

    // 写入目标粗网格显存
    int out_idx = k * (extc0 * extc1) + j * extc0 + i;
    d_dst_c[out_idx] = final_val;
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
    int lbc[3], lbf[3], lbr[3], ubr[3];

    // 1. 预计算所有几何对齐数据
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

        lbf[d] = (int)std::trunc((llbf[d] - base[d]) / FD[d] + 0.4) + 1;
        lbc[d] = (int)std::trunc((llbc[d] - base[d]) / CD[d] + 0.4) + 1;
        lbr[d] = (int)std::trunc((llbt[d] - base[d]) / CD[d] + 0.4) + 1;
        ubr[d] = (int)std::trunc((uubt[d] - base[d]) / CD[d] + 0.4);
    }

    // 2. 0-based 循环边界
    int i_start = lbr[0] - lbc[0];
    int i_end   = ubr[0] - lbc[0];
    int j_start = lbr[1] - lbc[1];
    int j_end   = ubr[1] - lbc[1];
    int k_start = lbr[2] - lbc[2];
    int k_end   = ubr[2] - lbc[2];

    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;

    if (ni <= 0 || nj <= 0 || nk <= 0) return;

    // 3. 配置 3D Grid 和 Block
    dim3 block(8, 8, 4);
    dim3 grid((ni + block.x - 1) / block.x,
              (nj + block.y - 1) / block.y,
              (nk + block.z - 1) / block.z);

    // 4. 传递标量
    restrict3_kernel<<<grid, block, 0, stream>>>(
        ni, nj, nk, i_start, j_start, k_start,
        lbc[0], lbc[1], lbc[2],
        lbf[0], lbf[1], lbf[2],
        extc[0], extc[1], extc[2],
        extf[0], extf[1], extf[2],
        d_src_f, d_dst_c,
        SoA[0], SoA[1], SoA[2],
        Symmetry
    );
}