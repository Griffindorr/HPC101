
#ifndef DERIVATIVES
#define DERIVATIVES

#ifdef fortran1
#define f_fderivs fderivs
#define f_fderivs_sh fderivs_sh
#define f_fderivs_shc fderivs_shc
#define f_fdderivs_shc fdderivs_shc
#define f_fdderivs fdderivs
#endif
#ifdef fortran2
#define f_fderivs FDERIVS
#define f_fderivs_sh FDERIVS_SH
#define f_fderivs_shc FDERIVS_SHC
#define f_fdderivs_shc FDDERIVS_SHC
#define f_fdderivs FDDERIVS
#endif
#ifdef fortran3
#define f_fderivs fderivs_
#define f_fderivs_sh fderivs_sh_
#define f_fderivs_shc fderivs_shc_
#define f_fdderivs_shc fdderivs_shc_
#define f_fdderivs fdderivs_
#endif

extern "C"
{
	void f_fderivs(int *, double *,
				   double *, double *, double *,
				   double *, double *, double *,
				   double &, double &, double &, int &, int &);
}

extern "C"
{
	void f_fderivs_sh(int *, double *,
					  double *, double *, double *,
					  double *, double *, double *,
					  double &, double &, double &, int &, int &, int &);
}

extern "C"
{
	void f_fderivs_shc(int *, double *,
					   double *, double *, double *,
					   double *, double *, double *,
					   double &, double &, double &, int &, int &, int &,
					   double *, double *, double *,
					   double *, double *, double *,
					   double *, double *, double *);
}

extern "C"
{
	void f_fdderivs_shc(int *, double *,
						double *, double *, double *, double *, double *, double *,
						double *, double *, double *,
						double &, double &, double &, int &, int &, int &,
						double *, double *, double *,
						double *, double *, double *,
						double *, double *, double *,
						double *, double *, double *, double *, double *, double *,
						double *, double *, double *, double *, double *, double *,
						double *, double *, double *, double *, double *, double *);
}

extern "C"
{
	void f_fdderivs(int *, double *,
					double *, double *, double *, double *, double *, double *,
					double *, double *, double *,
					double &, double &, double &, int &, int &);
}

#ifdef USE_GPU
#include <cuda_runtime.h>
#include "fmisc.h"

// Fortran Column-Major Layout: x varies fastest
#ifndef IDX3D
#define IDX3D(i, j, k, nx, ny, nz) ((i) + (nx) * ((j) + (ny) * (k)))
#endif

// Defined here (instead of a separate .cu) so they are force-inlined into every
// call site, letting the compiler CSE redundant neighbor loads across the many
// call sites and allocate registers for the whole call graph.
__device__ __forceinline__ void d_fderivs_point(
    const int ex[3], const double* f,
    double* fx, double* fy, double* fz,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, int onoff,
    int i, int j, int k
) {
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double EIT = 8.0;
    const double F12 = 12.0;
    const double ZEO = 0.0;
    const int NO_SYMM = 0, EQ_SYMM = 1;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const int imax = ex[0] - 1;
    const int jmax = ex[1] - 1;
    const int kmax = ex[2] - 1;

    *fx = ZEO;
    *fy = ZEO;
    *fz = ZEO;

    // Fortran 循环范围是 1 到 ex-1，对应 CUDA 0 到 ex-2。
    // 如果 i >= imax (即 i >= ex-1)，直接返回，保持输出为 0。
    if (i >= imax || j >= jmax || k >= kmax) return;

    // --- 修复开始 ---
    // Fortran 中 imin = -1 (1-based index)。
    // 在 CUDA (0-based) 中，为了使边界点 i=0 满足 (i-2 >= imin)，
    // 即 (0-2 >= imin) -> (-2 >= imin)，imin 必须设为 -2。
    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -2; // 原代码为 -1，修正为 -2
    if (symmetry > EQ_SYMM && fabs(X[0]) < dX) imin = -2; // 原代码为 -1，修正为 -2
    if (symmetry > EQ_SYMM && fabs(Y[0]) < dY) jmin = -2; // 原代码为 -1，修正为 -2

    double SoA[3] = {SYM1, SYM2, SYM3};

    const double d12dx = ONE / F12 / dX;
    const double d12dy = ONE / F12 / dY;
    const double d12dz = ONE / F12 / dZ;

    const double d2dx = ONE / TWO / dX;
    const double d2dy = ONE / TWO / dY;
    const double d2dz = ONE / TWO / dZ;

    // Helper lambda for symmetry boundary access
    const auto fh = [&](int ii, int jj, int kk) -> double {
        return d_symmetry_bd_1b(2, ex, f, ii + 1, jj + 1, kk + 1, SoA);
    };
    
    // Interior fast path: branch-free direct-indexing stencil.
    if (i >= 2 && i <= imax - 2 &&
        j >= 2 && j <= jmax - 2 &&
        k >= 2 && k <= kmax - 2) {
        const int nx = ex[0], ny = ex[1], nz = ex[2];
#define D(fi, fj, fk) (f[IDX3D((i)+(fi), (j)+(fj), (k)+(fk), nx, ny, nz)])
        *fx = d12dx * (D(-2,0,0) - EIT*D(-1,0,0) + EIT*D(1,0,0) - D(2,0,0));
        *fy = d12dy * (D(0,-2,0) - EIT*D(0,-1,0) + EIT*D(0,1,0) - D(0,2,0));
        *fz = d12dz * (D(0,0,-2) - EIT*D(0,0,-1) + EIT*D(0,0,1) - D(0,0,2));
#undef D
        return;
    }

    if (i + 2 <= imax && i - 2 >= imin &&
        j + 2 <= jmax && j - 2 >= jmin &&
        k + 2 <= kmax && k - 2 >= kmin) {

        *fx = d12dx * (fh(i-2,j,k) - EIT*fh(i-1,j,k) + EIT*fh(i+1,j,k) - fh(i+2,j,k));
        *fy = d12dy * (fh(i,j-2,k) - EIT*fh(i,j-1,k) + EIT*fh(i,j+1,k) - fh(i,j+2,k));
        *fz = d12dz * (fh(i,j,k-2) - EIT*fh(i,j,k-1) + EIT*fh(i,j,k+1) - fh(i,j,k+2));

    }
    else if (i + 1 <= imax && i - 1 >= imin &&
               j + 1 <= jmax && j - 1 >= jmin &&
               k + 1 <= kmax && k - 1 >= kmin) {

        *fx = d2dx * (-fh(i-1,j,k) + fh(i+1,j,k));
        *fy = d2dy * (-fh(i,j-1,k) + fh(i,j+1,k));
        *fz = d2dz * (-fh(i,j,k-1) + fh(i,j,k+1));
    }

    (void)onoff;
}

// ==========================================
// Device Function: 二阶导数 (4th Order)
// ==========================================
__device__ __forceinline__ void d_fdderivs_point(
    const int ex[3], const double* f,
    double* fxx, double* fxy, double* fxz,
    double* fyy, double* fyz, double* fzz,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, int onoff,
    int i, int j, int k
) {
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double F1o4 = 0.25;
    const double F1o12 = ONE / 12.0;
    const double F1o144 = ONE / 144.0;
    const double F8 = 8.0;
    const double F16 = 16.0;
    const double F30 = 30.0;
    const double ZEO = 0.0;
    const int NO_SYMM = 0, EQ_SYMM = 1;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const int imax = ex[0] - 1;
    const int jmax = ex[1] - 1;
    const int kmax = ex[2] - 1;

    *fxx = ZEO; *fyy = ZEO; *fzz = ZEO;
    *fxy = ZEO; *fxz = ZEO; *fyz = ZEO;

    if (i >= imax || j >= jmax || k >= kmax) return;

    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -2;
    if (symmetry > EQ_SYMM && fabs(X[0]) < dX) imin = -2;
    if (symmetry > EQ_SYMM && fabs(Y[0]) < dY) jmin = -2;

    double SoA[3] = {SYM1, SYM2, SYM3};

    const double Sdxdx = ONE / (dX * dX);
    const double Sdydy = ONE / (dY * dY);
    const double Sdzdz = ONE / (dZ * dZ);

    const double Fdxdx = F1o12 / (dX * dX);
    const double Fdydy = F1o12 / (dY * dY);
    const double Fdzdz = F1o12 / (dZ * dZ);

    const double Sdxdy = F1o4 / (dX * dY);
    const double Sdxdz = F1o4 / (dX * dZ);
    const double Sdydz = F1o4 / (dY * dZ);

    const double Fdxdy = F1o144 / (dX * dY);
    const double Fdxdz = F1o144 / (dX * dZ);
    const double Fdydz = F1o144 / (dY * dZ);

    const auto fh = [&](int ii, int jj, int kk) -> double {
        return d_symmetry_bd_1b(2, ex, f, ii + 1, jj + 1, kk + 1, SoA);
    };
    
    // Interior fast path: branch-free direct-indexing stencil.
    if (i >= 2 && i <= imax - 2 &&
        j >= 2 && j <= jmax - 2 &&
        k >= 2 && k <= kmax - 2) {
        const int nx = ex[0], ny = ex[1], nz = ex[2];
#define D(fi, fj, fk) (f[IDX3D((i)+(fi), (j)+(fj), (k)+(fk), nx, ny, nz)])
        *fxx = Fdxdx * (-D(-2,0,0) + F16*D(-1,0,0) - F30*D(0,0,0) + F16*D(1,0,0) - D(2,0,0));
        *fyy = Fdydy * (-D(0,-2,0) + F16*D(0,-1,0) - F30*D(0,0,0) + F16*D(0,1,0) - D(0,2,0));
        *fzz = Fdzdz * (-D(0,0,-2) + F16*D(0,0,-1) - F30*D(0,0,0) + F16*D(0,0,1) - D(0,0,2));

        *fxy = Fdxdy * (    (D(-2,-2,0) - F8*D(-1,-2,0) + F8*D(1,-2,0) - D(2,-2,0))
                        -F8*(D(-2,-1,0) - F8*D(-1,-1,0) + F8*D(1,-1,0) - D(2,-1,0))
                        +F8*(D(-2,1,0) - F8*D(-1,1,0) + F8*D(1,1,0) - D(2,1,0))
                        -   (D(-2,2,0) - F8*D(-1,2,0) + F8*D(1,2,0) - D(2,2,0)) );
        *fxz = Fdxdz * (    (D(-2,0,-2) - F8*D(-1,0,-2) + F8*D(1,0,-2) - D(2,0,-2))
                        -F8*(D(-2,0,-1) - F8*D(-1,0,-1) + F8*D(1,0,-1) - D(2,0,-1))
                        +F8*(D(-2,0,1) - F8*D(-1,0,1) + F8*D(1,0,1) - D(2,0,1))
                        -   (D(-2,0,2) - F8*D(-1,0,2) + F8*D(1,0,2) - D(2,0,2)) );
        *fyz = Fdydz * (    (D(0,-2,-2) - F8*D(0,-1,-2) + F8*D(0,1,-2) - D(0,2,-2))
                        -F8*(D(0,-2,-1) - F8*D(0,-1,-1) + F8*D(0,1,-1) - D(0,2,-1))
                        +F8*(D(0,-2,1) - F8*D(0,-1,1) + F8*D(0,1,1) - D(0,2,1))
                        -   (D(0,-2,2) - F8*D(0,-1,2) + F8*D(0,1,2) - D(0,2,2)) );
#undef D
        return;
    }

    // --- 4th Order Accuracy ---
    if (i + 2 <= imax && i - 2 >= imin &&
        j + 2 <= jmax && j - 2 >= jmin &&
        k + 2 <= kmax && k - 2 >= kmin) {

        *fxx = Fdxdx * (-fh(i-2,j,k) + F16*fh(i-1,j,k) - F30*fh(i,j,k)
                        -fh(i+2,j,k) + F16*fh(i+1,j,k));
        *fyy = Fdydy * (-fh(i,j-2,k) + F16*fh(i,j-1,k) - F30*fh(i,j,k)
                        -fh(i,j+2,k) + F16*fh(i,j+1,k));
        *fzz = Fdzdz * (-fh(i,j,k-2) + F16*fh(i,j,k-1) - F30*fh(i,j,k)
                        -fh(i,j,k+2) + F16*fh(i,j,k+1));

        *fxy = Fdxdy * (    (fh(i-2,j-2,k) - F8*fh(i-1,j-2,k) + F8*fh(i+1,j-2,k) - fh(i+2,j-2,k))
                        -F8*(fh(i-2,j-1,k) - F8*fh(i-1,j-1,k) + F8*fh(i+1,j-1,k) - fh(i+2,j-1,k))
                        +F8*(fh(i-2,j+1,k) - F8*fh(i-1,j+1,k) + F8*fh(i+1,j+1,k) - fh(i+2,j+1,k))
                        -   (fh(i-2,j+2,k) - F8*fh(i-1,j+2,k) + F8*fh(i+1,j+2,k) - fh(i+2,j+2,k)) );
        *fxz = Fdxdz * (    (fh(i-2,j,k-2) - F8*fh(i-1,j,k-2) + F8*fh(i+1,j,k-2) - fh(i+2,j,k-2))
                        -F8*(fh(i-2,j,k-1) - F8*fh(i-1,j,k-1) + F8*fh(i+1,j,k-1) - fh(i+2,j,k-1))
                        +F8*(fh(i-2,j,k+1) - F8*fh(i-1,j,k+1) + F8*fh(i+1,j,k+1) - fh(i+2,j,k+1))
                        -   (fh(i-2,j,k+2) - F8*fh(i-1,j,k+2) + F8*fh(i+1,j,k+2) - fh(i+2,j,k+2)) );
        *fyz = Fdydz * (    (fh(i,j-2,k-2) - F8*fh(i,j-1,k-2) + F8*fh(i,j+1,k-2) - fh(i,j+2,k-2))
                        -F8*(fh(i,j-2,k-1) - F8*fh(i,j-1,k-1) + F8*fh(i,j+1,k-1) - fh(i,j+2,k-1))
                        +F8*(fh(i,j-2,k+1) - F8*fh(i,j-1,k+1) + F8*fh(i,j+1,k+1) - fh(i,j+2,k+1))
                        -   (fh(i,j-2,k+2) - F8*fh(i,j-1,k+2) + F8*fh(i,j+1,k+2) - fh(i,j+2,k+2)) );

    } 
    // --- 2nd Order Accuracy ---
    else if (i + 1 <= imax && i - 1 >= imin &&
               j + 1 <= jmax && j - 1 >= jmin &&
               k + 1 <= kmax && k - 1 >= kmin) {

        *fxx = Sdxdx * (fh(i-1,j,k) - TWO*fh(i,j,k) + fh(i+1,j,k));
        *fyy = Sdydy * (fh(i,j-1,k) - TWO*fh(i,j,k) + fh(i,j+1,k));
        *fzz = Sdzdz * (fh(i,j,k-1) - TWO*fh(i,j,k) + fh(i,j,k+1));

        *fxy = Sdxdy * (fh(i-1,j-1,k) - fh(i+1,j-1,k) - fh(i-1,j+1,k) + fh(i+1,j+1,k));
        *fxz = Sdxdz * (fh(i-1,j,k-1) - fh(i+1,j,k-1) - fh(i-1,j,k+1) + fh(i+1,j,k+1));
        *fyz = Sdydz * (fh(i,j-1,k-1) - fh(i,j+1,k-1) - fh(i,j-1,k+1) + fh(i,j+1,k+1));
    }

    (void)onoff;
}
#endif

#endif /* DERIVATIVES */
