// d_fderivs_point / d_fdderivs_point now live in derivatives.h as
// __device__ __forceinline__ (inlined at every call site so the compiler can
// CSE neighbor loads and allocate registers for the whole call graph).
#include "derivatives.h"

#include "fmisc.h"

#include "macrodef.fh"
#include <cmath>
#include <iostream>
