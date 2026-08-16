// d_kodis_point now lives in kodiss.h as __device__ __forceinline__ so it is
// inlined into rhs_dissipation_kernel (24 call sites per point).
#include "kodiss.h"

#include "fmisc.h"

#include "macrodef.fh"
#include <cmath>
