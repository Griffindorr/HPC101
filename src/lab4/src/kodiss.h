
#ifndef KODISS_H
#define KODISS_H

#ifdef USE_GPU
#include <cuda_runtime.h>
#include "fmisc.h"

// Fortran Column-Major Layout: x varies fastest
#ifndef IDX3D
#define IDX3D(i, j, k, nx, ny, nz) ((i) + (nx) * ((j) + (ny) * (k)))
#endif

// Defined here so it is force-inlined into rhs_dissipation_kernel, letting the
// compiler see all 24 stencil calls together and (with __launch_bounds__)
// allocate registers for the whole call graph.
__device__ __forceinline__ double d_kodis_point(
    const int ex[3], const double* f,
    const double* X, const double* Y, const double* Z,
    double SYM1, double SYM2, double SYM3,
    int symmetry, double eps,
    int i, int j, int k // 0-based
) {
    const double ONE = 1.0;
    const double SIX = 6.0;
    const double FIT = 15.0;
    const double TWT = 20.0;
    const double cof = 64.0;
    const int NO_SYMM = 0, OCTANT = 2;

    const double dX = X[1] - X[0];
    const double dY = Y[1] - Y[0];
    const double dZ = Z[1] - Z[0];

    const int imax = ex[0] - 1;
    const int jmax = ex[1] - 1;
    const int kmax = ex[2] - 1;

    int imin = 0, jmin = 0, kmin = 0;
    if (symmetry > NO_SYMM && fabs(Z[0]) < dZ) kmin = -3;
    if (symmetry == OCTANT && fabs(X[0]) < dX) imin = -3;
    if (symmetry == OCTANT && fabs(Y[0]) < dY) jmin = -3;

    // Interior fast path: point at least 3 cells away from every boundary and
    // symmetry plane -> branch-free 4th-order stencil with direct indexing.
    // Bit-identical to the guarded path below (factor=1, no flip).
    if (i >= 3 && i <= imax - 3 &&
        j >= 3 && j <= jmax - 3 &&
        k >= 3 && k <= kmax - 3) {
        const int nx = ex[0], ny = ex[1], nz = ex[2];
        const double f0 = f[IDX3D(i,j,k,nx,ny,nz)];
        double sum = 0.0;
        sum += (f[IDX3D(i-3,j,k,nx,ny,nz)] + f[IDX3D(i+3,j,k,nx,ny,nz)]
              - SIX*(f[IDX3D(i-2,j,k,nx,ny,nz)] + f[IDX3D(i+2,j,k,nx,ny,nz)])
              + FIT*(f[IDX3D(i-1,j,k,nx,ny,nz)] + f[IDX3D(i+1,j,k,nx,ny,nz)])
              - TWT*f0) / dX;
        sum += (f[IDX3D(i,j-3,k,nx,ny,nz)] + f[IDX3D(i,j+3,k,nx,ny,nz)]
              - SIX*(f[IDX3D(i,j-2,k,nx,ny,nz)] + f[IDX3D(i,j+2,k,nx,ny,nz)])
              + FIT*(f[IDX3D(i,j-1,k,nx,ny,nz)] + f[IDX3D(i,j+1,k,nx,ny,nz)])
              - TWT*f0) / dY;
        sum += (f[IDX3D(i,j,k-3,nx,ny,nz)] + f[IDX3D(i,j,k+3,nx,ny,nz)]
              - SIX*(f[IDX3D(i,j,k-2,nx,ny,nz)] + f[IDX3D(i,j,k+2,nx,ny,nz)])
              + FIT*(f[IDX3D(i,j,k-1,nx,ny,nz)] + f[IDX3D(i,j,k+1,nx,ny,nz)])
              - TWT*f0) / dZ;
        return eps / cof * sum;
    }

    double SoA[3] = {SYM1, SYM2, SYM3};

    const auto fh = [&](int ii, int jj, int kk) -> double {
        return d_symmetry_bd_1b(3, ex, f, ii + 1, jj + 1, kk + 1, SoA);
    };

    double rhs_add = 0.0;

    if (i - 3 >= imin && i + 3 <= imax &&
        j - 3 >= jmin && j + 3 <= jmax &&
        k - 3 >= kmin && k + 3 <= kmax) {

        rhs_add = eps / cof * (
            ((fh(i-3,j,k) + fh(i+3,j,k)) - SIX*(fh(i-2,j,k) + fh(i+2,j,k)) +
             FIT*(fh(i-1,j,k) + fh(i+1,j,k)) - TWT*fh(i,j,k)) / dX +
            ((fh(i,j-3,k) + fh(i,j+3,k)) - SIX*(fh(i,j-2,k) + fh(i,j+2,k)) +
             FIT*(fh(i,j-1,k) + fh(i,j+1,k)) - TWT*fh(i,j,k)) / dY +
            ((fh(i,j,k-3) + fh(i,j,k+3)) - SIX*(fh(i,j,k-2) + fh(i,j,k+2)) +
             FIT*(fh(i,j,k-1) + fh(i,j,k+1)) - TWT*fh(i,j,k)) / dZ
        );
    }

    return rhs_add;
}
#endif

#endif /* KODISS_H */
