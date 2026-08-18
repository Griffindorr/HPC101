
#ifdef newc
#include <assert.h>
#include <ctype.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <strstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <complex>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
#else
#include <assert.h>
#include <ctype.h>
#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#endif

#include "TwoPunctures.h"

TwoPunctures::TwoPunctures(double mp, double mm, double b,
                           double P_plusx, double P_plusy, double P_plusz,
                           double S_plusx, double S_plusy, double S_plusz,
                           double P_minusx, double P_minusy, double P_minusz,
                           double S_minusx, double S_minusy, double S_minusz,
                           int nA, int nB, int nphi,
                           double Mp, double Mm, double admtol, double Newtontol,
                           int Newtonmaxit) : par_m_plus(mp), par_m_minus(mm), par_b(b), npoints_A(nA),
                                              npoints_B(nB), npoints_phi(nphi), target_M_plus(Mp), target_M_minus(Mm),
                                              adm_tol(admtol), Newton_tol(Newtontol), Newton_maxit(Newtonmaxit)
{
  assert(nA <= MAX_LINE_SIZE && nB <= MAX_LINE_SIZE);

  // TwoPunctureABE-specific OpenMP thread count (optional). run.sh may export
  // AMSS_TWOP_OMP_THREADS; when unset (e.g. evaluation environment) the
  // default OpenMP behavior (OMP_NUM_THREADS) is kept.
  const char *twop_omp = getenv("AMSS_TWOP_OMP_THREADS");
  if (twop_omp != NULL && *twop_omp != '\0')
    omp_set_num_threads(atoi(twop_omp));

  par_P_plus[0] = P_plusx;
  par_P_plus[1] = P_plusy;
  par_P_plus[2] = P_plusz;
  par_P_minus[0] = P_minusx;
  par_P_minus[1] = P_minusy;
  par_P_minus[2] = P_minusz;
  par_S_plus[0] = S_plusx;
  par_S_plus[1] = S_plusy;
  par_S_plus[2] = S_plusz;
  par_S_minus[0] = S_minusx;
  par_S_minus[1] = S_minusy;
  par_S_minus[2] = S_minusz;

  int const nvar = 1, n1 = npoints_A, n2 = npoints_B, n3 = npoints_phi;

  ntotal = n1 * n2 * n3 * nvar;

  F = dvector(0, ntotal - 1);
  allocate_derivs(&u, ntotal);
  allocate_derivs(&v, ntotal);

  lu_be_l = new double[ntotal];
  lu_be_u = new double[ntotal];
  lu_be_r = new double[ntotal];
  lu_al_l = new double[ntotal];
  lu_al_u = new double[ntotal];
  lu_al_r = new double[ntotal];

  relax_wall = 0.0;
  relax_calls = 0;
  t_F_of_v = t_SetJFD = t_BuildLF = t_J_times_dv = t_spec = 0.0;
}

TwoPunctures::~TwoPunctures()
{
  delete[] lu_be_l;
  delete[] lu_be_u;
  delete[] lu_be_r;
  delete[] lu_al_l;
  delete[] lu_al_u;
  delete[] lu_al_r;
  free_dvector(F, 0, ntotal - 1);
  free_derivs(&u, ntotal);
  free_derivs(&v, ntotal);
}

void TwoPunctures::Solve()
{

  double mp = par_m_plus;
  double mm = par_m_minus;

  enum GRID_SETUP_METHOD
  {
    GSM_Taylor_expansion,
    GSM_evaluation
  };
  enum GRID_SETUP_METHOD gsm;

  int antisymmetric_lapse, averaged_lapse, pmn_lapse, brownsville_lapse;

  int const nvar = 1, n1 = npoints_A, n2 = npoints_B, n3 = npoints_phi;

  int imin[3], imax[3];
  int const ntotal = n1 * n2 * n3 * nvar;

  // double admMass;

  /* initialise to 0 */
  for (int j = 0; j < ntotal; j++)
  {
    v.d0[j] = 0.0;
    v.d1[j] = 0.0;
    v.d2[j] = 0.0;
    v.d3[j] = 0.0;
    v.d11[j] = 0.0;
    v.d12[j] = 0.0;
    v.d13[j] = 0.0;
    v.d22[j] = 0.0;
    v.d23[j] = 0.0;
    v.d33[j] = 0.0;
  }

  double tmp, Mp_adm, Mm_adm, Mp_adm_err, Mm_adm_err, up, um;

  double M_p = target_M_plus;
  double M_m = target_M_minus;
  /* If bare masses are not given, iteratively solve for them given the
     target ADM masses target_M_plus and target_M_minus and with initial
     guesses given by par_m_plus and par_m_minus. */
  if (par_m_plus < 0 || par_m_minus < 0)
  {

    par_m_plus = target_M_plus;
    par_m_minus = target_M_minus;
    cout << "Attempting to find bare masses." << endl;
    cout << "Target ADM masses: M_p=" << M_p << " and M_m=" << M_m << endl;
    cout << "ADM mass tolerance: " << adm_tol << endl;

    /* Loop until both ADM masses are within adm_tol of their target */
    do
    {
      cout << "Bare masses: mp=" << mp << ", mm=" << mm << endl;
      Newton(nvar, n1, n2, n3, v, Newton_tol, 1);

      F_of_v(nvar, n1, n2, n3, v, F, u);

      up = PunctIntPolAtArbitPosition(0, nvar, n1, n2, n3, v, par_b, 0., 0.);
      um = PunctIntPolAtArbitPosition(0, nvar, n1, n2, n3, v, -par_b, 0., 0.);

      /* Calculate the ADM masses from the current bare mass guess PRD 70, 064011 (2004) Eq.(83)*/
      Mp_adm = (1 + up) * mp + mp * mm / (4. * par_b);
      Mm_adm = (1 + um) * mm + mp * mm / (4. * par_b);

      /* Check how far the current ADM masses are from the target */
      Mp_adm_err = fabs(M_p - Mp_adm);
      Mm_adm_err = fabs(M_m - Mm_adm);
      cout << "ADM mass error: M_p_err=" << Mp_adm_err << ", M_m_err=" << Mm_adm_err << endl;

      /* Invert the ADM mass equation and update the bare mass guess so that
         it gives the correct target ADM masses */
      tmp = -4 * par_b * (1 + um + up + um * up) +
            sqrt(16 * par_b * M_m * (1 + um) * (1 + up) +
                 pow(-M_m + M_p + 4 * par_b * (1 + um) * (1 + up), 2));
      par_m_plus = mp = (tmp + M_p - M_m) / (2. * (1 + up));
      par_m_minus = mm = (tmp - M_p + M_m) / (2. * (1 + um));

    } while ((Mp_adm_err > adm_tol) ||
             (Mm_adm_err > adm_tol));

    cout << "Found bare masses resulted Mp = " << Mp_adm << " and Mm = " << Mm_adm << endl;
  }

  Newton(nvar, n1, n2, n3, v, Newton_tol, Newton_maxit);

  F_of_v(nvar, n1, n2, n3, v, F, u);

  up = PunctIntPolAtArbitPosition(0, nvar, n1, n2, n3, v, par_b, 0., 0.);
  um = PunctIntPolAtArbitPosition(0, nvar, n1, n2, n3, v, -par_b, 0., 0.);

  /* Calculate the ADM masses from the current bare mass guess PRD 70, 064011 (2004) Eq.(83)*/
  Mp_adm = (1 + up) * mp + mp * mm / (4. * par_b);
  Mm_adm = (1 + um) * mm + mp * mm / (4. * par_b);

  cout << "The two puncture masses are mp = " << mp << " and mm = " << mm << endl;
  cout << "                   resulted Mp = " << Mp_adm << " and Mm = " << Mm_adm << endl;

  /* print out ADM mass, eq.: \Delta M_ADM=2*r*u=4*b*V for A=1,B=0,phi=0 PRD 70, 064011 (2004) Eq.(81)*/
  admMass = (mp + mm - 4 * par_b * PunctEvalAtArbitPosition(v.d0, 0, 1, 0, 0, nvar, n1, n2, n3));
  cout << "The total ADM mass is " << admMass << endl;

  target_M_plus = Mp_adm;
  target_M_minus = Mm_adm;
}
void TwoPunctures::Save(char *fname)
{
  ofstream outfile;
  outfile.open(fname, ios::trunc);

  time_t tnow;
  time(&tnow);
  struct tm *loc_time;
  loc_time = localtime(&tnow);
  outfile << "#File created on " << asctime(loc_time);
  outfile << "#Newton_tol = " << Newton_tol << endl;
  outfile << "#Mp         = " << target_M_plus << endl;
  outfile << "#Mm         = " << target_M_minus << endl;
  double D = 2 * par_b, x1, x2;
  x1 = D * target_M_minus / (target_M_plus + target_M_minus);
  x2 = -D * target_M_plus / (target_M_plus + target_M_minus);
  // in order to relate Brugmann's convention, rotate xy
  outfile << "bhmass1      = " << par_m_plus << endl;
  outfile << "bhx1         = " << 0 << endl;
  outfile << "bhy1         = " << x1 << endl;
  outfile << "bhz1         = " << 0 << endl;
  outfile << "bhpx1        = " << -par_P_plus[1] << endl;
  outfile << "bhpy1        = " << par_P_plus[0] << endl;
  outfile << "bhpz1        = " << par_P_plus[2] << endl;
  outfile << "bhsx1        = " << -par_S_plus[1] << endl;
  outfile << "bhsy1        = " << par_S_plus[0] << endl;
  outfile << "bhsz1        = " << par_S_plus[2] << endl;
  outfile << "bhmass2      = " << par_m_minus << endl;
  outfile << "bhx2         = " << 0 << endl;
  outfile << "bhy2         = " << x2 << endl;
  outfile << "bhz2         = " << 0 << endl;
  outfile << "bhpx2        = " << -par_P_minus[1] << endl;
  outfile << "bhpy2        = " << par_P_minus[0] << endl;
  outfile << "bhpz2        = " << par_P_minus[2] << endl;
  outfile << "bhsx2        = " << -par_S_minus[1] << endl;
  outfile << "bhsy2        = " << par_S_minus[0] << endl;
  outfile << "bhsz2        = " << par_S_minus[2] << endl;
  int const n1 = npoints_A, n2 = npoints_B, n3 = npoints_phi;
  outfile << "data " << n1 << " " << n2 << " " << n3 << endl;
  int ntotal = n1 * n2 * n3;

  outfile.setf(ios::scientific, ios::floatfield);
  outfile.precision(16);
  for (int i = 0; i < ntotal; i++)
    outfile << v.d0[i] << endl;

  outfile.close();

  // add output to facilitate python reading of puncture data, by Xiaoqu 2024/12/04
  ofstream outfile2;
  outfile2.open("puncture_parameters_new.txt", ios::trunc);

  // note that in this program the xy plane has been rotated
  outfile2 << setw(18) << setprecision(10) << par_m_plus
           << setw(18) << setprecision(10) << target_M_plus
           << setw(18) << setprecision(10) << admMass << " # bare mass 1   mass 1   ADM mass" << endl;
  outfile2 << setw(18) << setprecision(10) << 0.0
           << setw(18) << setprecision(10) << x1
           << setw(18) << setprecision(10) << 0.0 << " # position 1" << endl;
  outfile2 << setw(18) << setprecision(10) << -par_P_plus[1]
           << setw(18) << setprecision(10) << par_P_plus[0]
           << setw(18) << setprecision(10) << par_P_plus[2] << " # momentum 1" << endl;
  outfile2 << setw(18) << setprecision(10) << -par_S_plus[1]
           << setw(18) << setprecision(10) << par_S_plus[0]
           << setw(18) << setprecision(10) << par_S_plus[2] << " # angular mumentum 1" << endl;
  outfile2 << setw(18) << setprecision(10) << par_m_minus
           << setw(18) << setprecision(10) << target_M_minus
           << setw(18) << setprecision(10) << admMass << " # bare mass 2   mass 2   ADM mass" << endl;
  outfile2 << setw(18) << setprecision(10) << 0.0
           << setw(18) << setprecision(10) << x2
           << setw(18) << setprecision(10) << 0.0 << " # position 2" << endl;
  outfile2 << setw(18) << setprecision(10) << -par_P_minus[1]
           << setw(18) << setprecision(10) << par_P_minus[0]
           << setw(18) << setprecision(10) << par_P_minus[2] << " # momentum 2" << endl;
  outfile2 << setw(18) << setprecision(10) << -par_S_minus[1]
           << setw(18) << setprecision(10) << par_S_minus[0]
           << setw(18) << setprecision(10) << par_S_minus[2] << " # angular mumentum 2" << endl;

  outfile2.close();
}

void TwoPunctures::set_initial_guess(derivs v)
{

  int nvar = 1, n1 = npoints_A, n2 = npoints_B, n3 = npoints_phi;

  double *s_x, *s_y, *s_z; // Cartesian x,y,z
  double al, A, Am1, be, B, phi, R, r, X;
  int ivar, i, j, k, i3D, indx;
  derivs U;
  FILE *debug_file;

  s_x = (double *)calloc(n1 * n2 * n3, sizeof(double));
  s_y = (double *)calloc(n1 * n2 * n3, sizeof(double));
  s_z = (double *)calloc(n1 * n2 * n3, sizeof(double));
  allocate_derivs(&U, nvar);
  for (ivar = 0; ivar < nvar; ivar++)
    for (i = 0; i < n1; i++)
      for (j = 0; j < n2; j++)
        for (k = 0; k < n3; k++)
        {
          i3D = Index(ivar, i, j, k, 1, n1, n2, n3);

          al = Pih * (2 * i + 1) / n1;
          A = -cos(al);
          be = Pih * (2 * j + 1) / n2;
          B = -cos(be);
          phi = 2. * Pi * k / n3;

          /* Calculation of (X,R)*/
          AB_To_XR(nvar, A, B, &X, &R, U);
          /* Calculation of (x,r)*/
          C_To_c(nvar, X, R, &(s_x[i3D]), &r, U);
          /* Calculation of (y,z)*/
          rx3_To_xyz(nvar, s_x[i3D], r, phi, &(s_y[i3D]), &(s_z[i3D]), U);
        }
  //  Set_Initial_Guess_for_u(n1*n2*n3, v.d0, s_x, s_y, s_z);  //extern fortran code to set initial guess
  for (ivar = 0; ivar < nvar; ivar++)
    for (i = 0; i < n1; i++)
      for (j = 0; j < n2; j++)
        for (k = 0; k < n3; k++)
        {
          indx = Index(ivar, i, j, k, 1, n1, n2, n3);
          v.d0[indx] = 0;                                     // set initial guess 0
          v.d0[indx] /= (-cos(Pih * (2 * i + 1) / n1) - 1.0); // PRD 70, 064011 (2004) Eq.(5), from u to U
        }
  Derivatives_AB3(nvar, n1, n2, n3, v);
  if (0)
  {
    debug_file = fopen("initial.dat", "w");
    assert(debug_file);
    for (ivar = 0; ivar < nvar; ivar++)
      for (i = 0; i < n1; i++)
        for (j = 0; j < n2; j++)
        {
          al = Pih * (2 * i + 1) / n1;
          A = -cos(al);
          Am1 = A - 1.0;
          be = Pih * (2 * j + 1) / n2;
          B = -cos(be);
          phi = 0.0;
          indx = Index(ivar, i, j, 0, 1, n1, n2, n3);
          U.d0[0] = Am1 * v.d0[indx];                    /* U*/
          U.d1[0] = v.d0[indx] + Am1 * v.d1[indx];       /* U_A*/
          U.d2[0] = Am1 * v.d2[indx];                    /* U_B*/
          U.d3[0] = Am1 * v.d3[indx];                    /* U_3*/
          U.d11[0] = 2 * v.d1[indx] + Am1 * v.d11[indx]; /* U_AA*/
          U.d12[0] = v.d2[indx] + Am1 * v.d12[indx];     /* U_AB*/
          U.d13[0] = v.d3[indx] + Am1 * v.d13[indx];     /* U_AB*/
          U.d22[0] = Am1 * v.d22[indx];                  /* U_BB*/
          U.d23[0] = Am1 * v.d23[indx];                  /* U_B3*/
          U.d33[0] = Am1 * v.d33[indx];                  /* U_33*/
          /* Calculation of (X,R)*/
          AB_To_XR(nvar, A, B, &X, &R, U);
          /* Calculation of (x,r)*/
          C_To_c(nvar, X, R, &(s_x[indx]), &r, U);
          /* Calculation of (y,z)*/
          rx3_To_xyz(nvar, s_x[i3D], r, phi, &(s_y[indx]), &(s_z[indx]), U);
          fprintf(debug_file,
                  "%.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g "
                  "%.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g\n",
                  (double)s_x[indx], (double)s_y[indx],
                  (double)A, (double)B,
                  (double)U.d0[0],
                  (double)(-cos(Pih * (2 * i + 1) / n1) - 1.0),
                  (double)U.d1[0],
                  (double)U.d2[0],
                  (double)U.d3[0],
                  (double)U.d11[0],
                  (double)U.d22[0],
                  (double)U.d33[0],
                  (double)v.d0[indx],
                  (double)v.d1[indx],
                  (double)v.d2[indx],
                  (double)v.d3[indx],
                  (double)v.d11[indx],
                  (double)v.d22[indx],
                  (double)v.d33[indx]);
        }
    fprintf(debug_file, "\n\n");
    for (i = n2 - 10; i < n2; i++)
    {
      double d;
      indx = Index(0, 0, i, 0, 1, n1, n2, n3);
      d = PunctIntPolAtArbitPosition(0, nvar, n1, n2, n3, v,
                                     s_x[indx], 0.0, 0.0);
      fprintf(debug_file, "%.16g %.16g\n",
              (double)s_x[indx], (double)d);
    }
    fprintf(debug_file, "\n\n");
    for (i = n2 - 10; i < n2 - 1; i++)
    {
      double d;
      int ip = Index(0, 0, i + 1, 0, 1, n1, n2, n3);
      indx = Index(0, 0, i, 0, 1, n1, n2, n3);
      for (j = -10; j < 10; j++)
      {
        d = PunctIntPolAtArbitPosition(0, nvar, n1, n2, n3, v,
                                       s_x[indx] + (s_x[ip] - s_x[indx]) * j / 10,
                                       0.0, 0.0);
        fprintf(debug_file, "%.16g %.16g\n",
                (double)(s_x[indx] + (s_x[ip] - s_x[indx]) * j / 10), (double)d);
      }
    }
    fprintf(debug_file, "\n\n");
    for (i = 0; i < n1; i++)
      for (j = 0; j < n2; j++)
      {
        X = 2 * (2.0 * i / n1 - 1.0);
        R = 2 * (1.0 * j / n2);
        if (X * X + R * R > 1.0)
        {
          C_To_c(nvar, X, R, &(s_x[indx]), &r, U);
          rx3_To_xyz(nvar, s_x[i3D], r, phi, &(s_y[indx]), &(s_z[indx]), U);
          *U.d0 = s_x[indx] * s_x[indx];
          *U.d1 = 2 * s_x[indx];
          *U.d2 = 0.0;
          *U.d3 = 0.0;
          *U.d11 = 2.0;
          *U.d22 = 0.0;
          *U.d33 = *U.d12 = *U.d23 = *U.d13 = 0.0;
          C_To_c(nvar, X, R, &(s_x[indx]), &r, U);
          fprintf(debug_file,
                  "%.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g\n",
                  (double)s_x[indx], (double)r, (double)X, (double)R, (double)U.d0[0],
                  (double)U.d1[0],
                  (double)U.d2[0],
                  (double)U.d3[0],
                  (double)U.d11[0],
                  (double)U.d22[0],
                  (double)U.d33[0]);
        }
      }
    fclose(debug_file);
  }
  free(s_z);
  free(s_y);
  free(s_x);
  free_derivs(&U, nvar);
}

// some tools
/*---------------------------------------------------------------------------*/
int TwoPunctures::index(int i, int j, int k, int l, int a, int b, int c, int d)
{
  int rr = 0;
  rr = l + k * d + j * d * c + i * d * c * b;
  return rr;
}
/*---------------------------------------------------------------------------*/
int *TwoPunctures::ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
  int *retval;

  retval = (int *)malloc(sizeof(int) * (nh - nl + 1));
  if (retval == NULL)
    cout << "allocation failure in ivector()" << endl;

  return retval - nl;
}

/*---------------------------------------------------------------------------*/
double *TwoPunctures::dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
  double *retval;

  retval = (double *)malloc(sizeof(double) * (nh - nl + 1));
  if (retval == NULL)
    cout << "allocation failure in dvector()" << endl;

  return retval - nl;
}

/*---------------------------------------------------------------------------*/
int **TwoPunctures::imatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  int **retval;

  retval = (int **)malloc(sizeof(int *) * (nrh - nrl + 1));
  if (retval == NULL)
    cout << "allocation failure (1) in imatrix()" << endl;

  /* get all memory for the matrix in on chunk */
  retval[0] = (int *)malloc(sizeof(int) * (nrh - nrl + 1) * (nch - ncl + 1));
  if (retval[0] == NULL)
    cout << "allocation failure (2) in imatrix()" << endl;

  /* apply column and row offsets */
  retval[0] -= ncl;
  retval -= nrl;

  /* slice chunk into rows */
  long width = (nch - ncl + 1);
  for (long i = nrl + 1; i <= nrh; i++)
    retval[i] = retval[i - 1] + width;
  assert(retval[nrh] - retval[nrl] == (nrh - nrl) * width);

  return retval;
}

/*---------------------------------------------------------------------------*/
double **TwoPunctures::dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  double **retval;

  retval = (double **)malloc(sizeof(double *) * (nrh - nrl + 1));
  if (retval == NULL)
    cout << "allocation failure (1) in dmatrix()" << endl;

  /* get all memory for the matrix in on chunk */
  retval[0] = (double *)malloc(sizeof(double) * (nrh - nrl + 1) * (nch - ncl + 1));
  if (retval[0] == NULL)
    cout << "allocation failure (2) in dmatrix()" << endl;

  /* apply column and row offsets */
  retval[0] -= ncl;
  retval -= nrl;

  /* slice chunk into rows */
  long width = (nch - ncl + 1);
  for (long i = nrl + 1; i <= nrh; i++)
    retval[i] = retval[i - 1] + width;
  assert(retval[nrh] - retval[nrl] == (nrh - nrl) * width);

  return retval;
}

/*---------------------------------------------------------------------------*/
double ***TwoPunctures::d3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate a double 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
  double ***retval;

  /* get memory for index structures */
  retval = (double ***)malloc(sizeof(double **) * (nrh - nrl + 1));
  if (retval == NULL)
    cout << "allocation failure (1) in dmatrix()" << endl;

  retval[0] = (double **)malloc(sizeof(double *) * (nrh - nrl + 1) * (nch - ncl + 1));
  if (retval[0] == NULL)
    cout << "allocation failure (2) in dmatrix()" << endl;

  /* get all memory for the tensor in on chunk */
  retval[0][0] = (double *)malloc(sizeof(double) * (nrh - nrl + 1) * (nch - ncl + 1) * (nrh - nrl + 1));
  if (retval[0][0] == NULL)
    cout << "allocation failure (3) in dmatrix()" << endl;

  /* apply all offsets */
  retval[0][0] -= ndl;
  retval[0] -= ncl;
  retval -= nrl;

  /* slice chunk into rows and columns */
  long width = (nch - ncl + 1);
  long depth = (ndh - ndl + 1);
  for (long j = ncl + 1; j <= nch; j++)
  { /* first row of columns */
    retval[nrl][j] = retval[nrl][j - 1] + depth;
  }
  assert(retval[nrl][nch] - retval[nrl][ncl] == (nch - ncl) * depth);
  for (long i = nrl + 1; i <= nrh; i++)
  {
    retval[i] = retval[i - 1] + width;
    retval[i][ncl] = retval[i - 1][ncl] + width * depth; /* first cell in column */
    for (long j = ncl + 1; j <= nch; j++)
    {
      retval[i][j] = retval[i][j - 1] + depth;
    }
    assert(retval[i][nch] - retval[i][ncl] == (nch - ncl) * depth);
  }
  assert(retval[nrh] - retval[nrl] == (nrh - nrl) * width);
  assert(&retval[nrh][nch][ndh] - &retval[nrl][ncl][ndl] == (nrh - nrl + 1) * (nch - ncl + 1) * (ndh - ndl + 1) - 1);

  return retval;
}

/*--------------------------------------------------------------------------*/
void TwoPunctures::free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
  free(v + nl);
}

/*--------------------------------------------------------------------------*/
void TwoPunctures::free_dvector(double *v, long nl, long nh)
/* free an double vector allocated with dvector() */
{
  free(v + nl);
}

/*--------------------------------------------------------------------------*/
void TwoPunctures::free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
/* free an int matrix allocated by imatrix() */
{
  free(m[nrl] + ncl);
  free(m + nrl);
}

/*--------------------------------------------------------------------------*/
void TwoPunctures::free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
  free(m[nrl] + ncl);
  free(m + nrl);
}

/*--------------------------------------------------------------------------*/
void TwoPunctures::free_d3tensor(double ***t, long nrl, long nrh, long ncl, long nch,
                                 long ndl, long ndh)
/* free a double f3tensor allocated by f3tensor() */
{
  free(t[nrl][ncl] + ndl);
  free(t[nrl] + ncl);
  free(t + nrl);
}

/*--------------------------------------------------------------------------*/
int TwoPunctures::minimum2(int i, int j)
{
  int result = i;
  if (j < result)
    result = j;
  return result;
}

/*-------------------------------------------------------------------------*/
int TwoPunctures::minimum3(int i, int j, int k)
{
  int result = i;
  if (j < result)
    result = j;
  if (k < result)
    result = k;
  return result;
}

/*--------------------------------------------------------------------------*/
int TwoPunctures::maximum2(int i, int j)
{
  int result = i;
  if (j > result)
    result = j;
  return result;
}

/*--------------------------------------------------------------------------*/
int TwoPunctures::maximum3(int i, int j, int k)
{
  int result = i;
  if (j > result)
    result = j;
  if (k > result)
    result = k;
  return result;
}

/*--------------------------------------------------------------------------*/
int TwoPunctures::pow_int(int mantisse, int exponent)
{
  int i, result = 1;

  for (i = 1; i <= exponent; i++)
    result *= mantisse;

  return result;
}

/*--------------------------------------------------------------------------*/
void TwoPunctures::chebft_Zeros(double u[], int n, int inv)
/* eq. 5.8.7 and 5.8.8 at x = (5.8.4) of 2nd edition C++ NR
   The transform matrices depend only on n (not on the data), so the cosine
   factors are precomputed once per n and reused across all transform calls.
   This removes every cos() call from the hot O(n^2) loops. */
{
  double *c = dvector(0, n);
  chebft_Zeros_sc(u, n, inv, c);
  free_dvector(c, 0, n);
}

void TwoPunctures::chebft_Zeros_sc(double u[], int n, int inv, double *c)
/* Scratch variant of chebft_Zeros: c must be a caller-owned buffer of at
   least n+1 doubles.  This lets the hot loops in Derivatives_AB3 allocate
   every working array once per thread instead of per line. */
{
  int k, j;
  double sum, Pion;

  static int cached_n = -1;
  static double *fwd_mat = 0; /* fwd_mat[j*n+k] = fac * (-1)^j * cos(Pion*j*(k+0.5)) */
  static double *inv_mat = 0; /* inv_mat[j*n+k] = (-1)^k * cos(Pion*(j+0.5)*k) */

  if (n != cached_n)
  {
#pragma omp critical(chebft_Zeros_cache)
    {
      if (n != cached_n)
      {
        if (fwd_mat)
        {
          delete[] fwd_mat;
          delete[] inv_mat;
        }
        fwd_mat = new double[n * n];
        inv_mat = new double[n * n];
        Pion = Pi / n;
        {
          double fac = 2.0 / n;
          for (j = 0; j < n; j++)
          {
            double sgn = (j & 1) ? -fac : fac; /* fac * isignum with alternating sign */
            for (k = 0; k < n; k++)
              fwd_mat[j * n + k] = sgn * cos(Pion * j * (k + 0.5));
          }
        }
        for (j = 0; j < n; j++)
        {
          for (k = 0; k < n; k++)
          {
            double sgn = (k & 1) ? -1.0 : 1.0; /* isignum alternates per k */
            inv_mat[j * n + k] = sgn * cos(Pion * (j + 0.5) * k);
          }
        }
        cached_n = n;
      }
    }
  }

  if (inv == 0)
  {
    for (j = 0; j < n; j++)
    {
      const double *row = fwd_mat + j * n;
      sum = 0.0;
      for (k = 0; k < n; k++)
        sum += row[k] * u[k];
      c[j] = sum;
    }
  }
  else
  {
    for (j = 0; j < n; j++)
    {
      const double *row = inv_mat + j * n;
      sum = -0.5 * u[0];
      for (k = 0; k < n; k++)
        sum += row[k] * u[k];
      c[j] = sum;
    }
  }
  for (j = 0; j < n; j++)
    u[j] = c[j];
}

/* --------------------------------------------------------------------------*/
void TwoPunctures::chebft_Extremes(double u[], int n, int inv)
/* eq. 5.8.7 and 5.8.8 at x = (5.8.5) of 2nd edition C++ NR */
{
  int k, j, isignum, N = n - 1;
  double fac, sum, PioN, *c;

  c = dvector(0, N);
  PioN = Pi / N;
  if (inv == 0)
  {
    fac = 2.0 / N;
    isignum = 1;
    for (j = 0; j < n; j++)
    {
      sum = 0.5 * (u[0] + u[N] * isignum);
      for (k = 1; k < N; k++)
        sum += u[k] * cos(PioN * j * k);
      c[j] = fac * sum * isignum;
      isignum = -isignum;
    }
    c[N] = 0.5 * c[N];
  }
  else
  {
    for (j = 0; j < n; j++)
    {
      sum = -0.5 * u[0];
      isignum = 1;
      for (k = 0; k < n; k++)
      {
        sum += u[k] * cos(PioN * j * k) * isignum;
        isignum = -isignum;
      }
      c[j] = sum;
    }
  }
  for (j = 0; j < n; j++)
    u[j] = c[j];
  free_dvector(c, 0, N);
}

/* --------------------------------------------------------------------------*/

void TwoPunctures::chder(double *c, double *cder, int n)
{
  int j;

  cder[n] = 0.0;
  cder[n - 1] = 0.0;
  for (j = n - 2; j >= 0; j--)
    cder[j] = cder[j + 2] + 2 * (j + 1) * c[j + 1];
}

/* --------------------------------------------------------------------------*/
double TwoPunctures::chebev(double a, double b, double c[], int m, double x)
/* eq. 5.8.11 of C++ NR (2nd ed) */
{
  int j;
  double djp2, djp1, dj; /* d_{j+2}, d_{j+1} and d_j */
  double y;

  /* rescale input to lie within [-1,1] */
  y = 2 * (x - 0.5 * (b + a)) / (b - a);

  dj = djp1 = 0;
  for (j = m - 1; j >= 1; j--)
  {
    /* advance the coefficients */
    djp2 = djp1;
    djp1 = dj;
    dj = 2 * y * djp1 - djp2 + c[j];
  }

  return y * dj - djp1 + 0.5 * c[0];
}

/* --------------------------------------------------------------------------*/
void TwoPunctures::fourft(double *u, int N, int inv)
/* a (slow) Fourier transform, seems to be just eq. 12.1.6 and 12.1.9 of C++ NR (2nd ed)
   The cos/sin factors depend only on N, so the (M+1)xN tables are precomputed
   once per N and reused; no cos()/sin() call remains in the hot loops. */
{
  double *a = dvector(0, N / 2);
  double *b = dvector(0, N / 2);
  fourft_sc(u, N, inv, a, b);
  free_dvector(a, 0, N / 2);
  free_dvector(b, 0, N / 2);
}

void TwoPunctures::fourft_sc(double *u, int N, int inv, double *a, double *b)
/* Scratch variant of fourft: a and b must be caller-owned buffers of at least
   N/2+1 doubles each.  b is indexed as b[1..M] in the original; here it is
   shifted to b[0..M-1] for the caller-owned buffer.  This lets the hot loops
   in Derivatives_AB3 allocate every working array once per thread. */
{
  int l, k, iy, M;
  double fac, Pi_fac;

  static int cached_N = -1;
  static double *cos_mat = 0; /* cos_mat[l*N + k] = cos(Pi_fac*l*k), l in [0,M] */
  static double *sin_mat = 0; /* sin_mat[l*N + k] = sin(Pi_fac*l*k), l in [0,M] */

  M = N / 2;
  if (N != cached_N)
  {
#pragma omp critical(fourft_cache)
    {
      if (N != cached_N)
      {
        if (cos_mat)
        {
          delete[] cos_mat;
          delete[] sin_mat;
        }
        cos_mat = new double[(M + 1) * N];
        sin_mat = new double[(M + 1) * N];
        fac = 1. / M;
        Pi_fac = Pi * fac;
        for (l = 0; l <= M; l++)
        {
          double x1 = Pi_fac * l;
          for (k = 0; k < N; k++)
          {
            double x = x1 * k;
            cos_mat[l * N + k] = cos(x);
            sin_mat[l * N + k] = sin(x);
          }
        }
        cached_N = N;
      }
    }
  }

  fac = 1. / M;
  if (inv == 0)
  {
    for (l = 0; l <= M; l++)
    {
      const double *crow = cos_mat + l * N;
      const double *srow = sin_mat + l * N;
      a[l] = 0;
      if (l > 0 && l < M)
        b[l - 1] = 0;
      for (k = 0; k < N; k++)
      {
        a[l] += fac * u[k] * crow[k];
        if (l > 0 && l < M)
          b[l - 1] += fac * u[k] * srow[k];
      }
    }
    u[0] = a[0];
    u[M] = a[M];
    for (l = 1; l < M; l++)
    {
      u[l] = a[l];
      u[l + M] = b[l - 1];
    }
  }
  else
  {
    a[0] = u[0];
    a[M] = u[M];
    for (l = 1; l < M; l++)
    {
      a[l] = u[l];
      b[l - 1] = u[M + l];
    }
    iy = 1;
    for (k = 0; k < N; k++)
    {
      double sum = 0.5 * (a[0] + a[M] * iy);
      for (l = 1; l < M; l++)
      {
        const double *crow = cos_mat + l * N;
        const double *srow = sin_mat + l * N;
        sum += a[l] * crow[k] + b[l - 1] * srow[k];
      }
      u[k] = sum;
      iy = -iy;
    }
  }
}

/* -----------------------------------------*/
void TwoPunctures::fourder(double u[], double du[], int N)
{
  int l, M, lpM;

  M = N / 2;
  du[0] = 0.;
  du[M] = 0.;
  for (l = 1; l < M; l++)
  {
    lpM = l + M;
    du[l] = u[lpM] * l;
    du[lpM] = -u[l] * l;
  }
}

/* -----------------------------------------*/
void TwoPunctures::fourder2(double u[], double d2u[], int N)
{
  int l, l2, M, lpM;

  d2u[0] = 0.;
  M = N / 2;
  for (l = 1; l <= M; l++)
  {
    l2 = l * l;
    lpM = l + M;
    d2u[l] = -u[l] * l2;
    if (l < M)
      d2u[lpM] = -u[lpM] * l2;
  }
}

/* ----------------------------------------- */
double TwoPunctures::fourev(double *u, int N, double x)
{
  int l, M = N / 2;
  double xl, result;

  result = 0.5 * (u[0] + u[M] * cos(x * M));
  for (l = 1; l < M; l++)
  {
    xl = x * l;
    result += u[l] * cos(xl) + u[M + l] * sin(xl);
  }
  return result;
}

/* ------------------------------------------------------------------------*/
double TwoPunctures::norm1(double *v, int n)
{
  int i;
  double result = -1;

  for (i = 0; i < n; i++)
    if (fabs(v[i]) > result)
      result = fabs(v[i]);

  return result;
}

/* -------------------------------------------------------------------------*/
double TwoPunctures::norm2(double *v, int n)
{
  int i;
  double result = 0;

  for (i = 0; i < n; i++)
    result += v[i] * v[i];

  return sqrt(result);
}

/* -------------------------------------------------------------------------*/
double TwoPunctures::scalarproduct(double *v, double *w, int n)
{
  int i;
  double result = 0;

  for (i = 0; i < n; i++)
    result += v[i] * w[i];

  return result;
}

/* -------------------------------------------------------------------------*/
/* Calculates the value of v at an arbitrary position (x,y,z)*/
double TwoPunctures::PunctIntPolAtArbitPosition(int ivar, int nvar, int n1,
                                                int n2, int n3, derivs v, double x, double y,
                                                double z)
{
  double xs, ys, zs, rs2, phi, X, R, A, B, aux1, aux2, result, Ui;

  xs = x / par_b;
  ys = y / par_b;
  zs = z / par_b;
  rs2 = ys * ys + zs * zs;
  phi = atan2(z, y);
  if (phi < 0)
    phi += 2 * Pi;

  aux1 = 0.5 * (xs * xs + rs2 - 1);
  aux2 = sqrt(aux1 * aux1 + rs2);
  X = asinh(sqrt(aux1 + aux2));
  R = asin(min(1.0, sqrt(-aux1 + aux2)));
  if (x < 0)
    R = Pi - R;

  A = 2 * tanh(0.5 * X) - 1;
  B = tan(0.5 * R - Piq);

  result = PunctEvalAtArbitPosition(v.d0, ivar, A, B, phi, nvar, n1, n2, n3);

  Ui = (A - 1) * result;

  return Ui;
}
/* Calculates the value of v at an arbitrary position (A,B,phi)*/
double TwoPunctures::PunctEvalAtArbitPosition(double *v, int ivar, double A, double B, double phi,
                                              int nvar, int n1, int n2, int n3)
{
  int i, j, k, N;
  double *p, *values1, **values2, result;

  N = maximum3(n1, n2, n3);
  p = dvector(0, N);
  values1 = dvector(0, N);
  values2 = dmatrix(0, N, 0, N);

  for (k = 0; k < n3; k++)
  {
    for (j = 0; j < n2; j++)
    {
      for (i = 0; i < n1; i++)
        p[i] = v[ivar + nvar * (i + n1 * (j + n2 * k))];
      chebft_Zeros(p, n1, 0);
      values2[j][k] = chebev(-1, 1, p, n1, A);
    }
  }

  for (k = 0; k < n3; k++)
  {
    for (j = 0; j < n2; j++)
      p[j] = values2[j][k];
    chebft_Zeros(p, n2, 0);
    values1[k] = chebev(-1, 1, p, n2, B);
  }

  fourft(values1, n3, 0);
  result = fourev(values1, n3, phi);

  free_dvector(p, 0, N);
  free_dvector(values1, 0, N);
  free_dmatrix(values2, 0, N, 0, N);

  return result;
}
/*-----------------------------------------------------------*/
void TwoPunctures::AB_To_XR(int nvar, double A, double B, double *X, double *R,
                            derivs U)
/* On Entrance: U.d0[]=U[]; U.d1[] =U[]_A;  U.d2[] =U[]_B;  U.d3[] =U[]_3;  */
/*                          U.d11[]=U[]_AA; U.d12[]=U[]_AB; U.d13[]=U[]_A3; */
/*                          U.d22[]=U[]_BB; U.d23[]=U[]_B3; U.d33[]=U[]_33; */
/* At Exit:     U.d0[]=U[]; U.d1[] =U[]_X;  U.d2[] =U[]_R;  U.d3[] =U[]_3;  */
/*                          U.d11[]=U[]_XX; U.d12[]=U[]_XR; U.d13[]=U[]_X3; */
/*                          U.d22[]=U[]_RR; U.d23[]=U[]_R3; U.d33[]=U[]_33; */
{
  double At = 0.5 * (A + 1), A_X, A_XX, B_R, B_RR;
  int ivar;

  *X = 2 * atanh(At);
  *R = Pih + 2 * atan(B);

  A_X = 1 - At * At;
  A_XX = -At * A_X;
  B_R = 0.5 * (1 + B * B);
  B_RR = B * B_R;

  for (ivar = 0; ivar < nvar; ivar++)
  {
    U.d11[ivar] = A_X * A_X * U.d11[ivar] + A_XX * U.d1[ivar];
    U.d12[ivar] = A_X * B_R * U.d12[ivar];
    U.d13[ivar] = A_X * U.d13[ivar];
    U.d22[ivar] = B_R * B_R * U.d22[ivar] + B_RR * U.d2[ivar];
    U.d23[ivar] = B_R * U.d23[ivar];
    U.d1[ivar] = A_X * U.d1[ivar];
    U.d2[ivar] = B_R * U.d2[ivar];
  }
}
/*-----------------------------------------------------------*/
void TwoPunctures::C_To_c(int nvar, double X, double R, double *x, double *r,
                          derivs U)
/* On Entrance: U.d0[]=U[]; U.d1[] =U[]_X;  U.d2[] =U[]_R;  U.d3[] =U[]_3;  */
/*                          U.d11[]=U[]_XX; U.d12[]=U[]_XR; U.d13[]=U[]_X3; */
/*                          U.d22[]=U[]_RR; U.d23[]=U[]_R3; U.d33[]=U[]_33; */
/* At Exit:     U.d0[]=U[]; U.d1[] =U[]_x;  U.d2[] =U[]_r;  U.d3[] =U[]_3;  */
/*                          U.d11[]=U[]_xx; U.d12[]=U[]_xr; U.d13[]=U[]_x3; */
/*                          U.d22[]=U[]_rr; U.d23[]=U[]_r3; U.d33[]=U[]_33; */
{
  double C_c2, U_cb, U_CB;
  complex<double> C, C_c, C_cc, c, c_C, c_CC, U_c, U_cc, U_C, U_CC;
  int ivar;

  C = complex<double>(X, R);

  c = cosh(C) * par_b; /* c=b*cosh(C)*/
  c_C = sinh(C) * par_b;
  c_CC = c;

  C_c = complex<double>(1, 0) / c_C;
  C_cc = -C_c * C_c * C_c * c_CC;
  C_c2 = abs(C_c);
  C_c2 = C_c2 * C_c2;

  for (ivar = 0; ivar < nvar; ivar++)
  {
    /* U_C = 0.5*(U_X3-i*U_R3)*/
    /* U_c = U_C*C_c = 0.5*(U_x3-i*U_r3)*/
    U_C = complex<double>(0.5 * U.d13[ivar], -0.5 * U.d23[ivar]);
    U_c = U_C * C_c;
    U.d13[ivar] = 2. * real(U_c);
    U.d23[ivar] = -2. * imag(U_c);

    /* U_C = 0.5*(U_X-i*U_R)*/
    /* U_c = U_C*C_c = 0.5*(U_x-i*U_r)*/
    U_C = complex<double>(0.5 * U.d1[ivar], -0.5 * U.d2[ivar]);
    U_c = U_C * C_c;
    U.d1[ivar] = 2. * real(U_c);
    U.d2[ivar] = -2. * imag(U_c);

    /* U_CC = 0.25*(U_XX-U_RR-2*i*U_XR)*/
    /* U_CB = d^2(U)/(dC*d\bar{C}) = 0.25*(U_XX+U_RR)*/
    U_CC = complex<double>(0.25 * (U.d11[ivar] - U.d22[ivar]), -0.5 * U.d12[ivar]);
    U_CB = 0.25 * (U.d11[ivar] + U.d22[ivar]);

    /* U_cc = C_cc*U_C+(C_c)^2*U_CC*/
    U_cb = U_CB * C_c2;
    U_cc = C_cc * U_C + C_c * C_c * U_CC;

    /* U_xx = 2*(U_cb+Re[U_cc])*/
    /* U_rr = 2*(U_cb-Re[U_cc])*/
    /* U_rx = -2*Im[U_cc]*/
    U.d11[ivar] = 2 * (U_cb + real(U_cc));
    U.d22[ivar] = 2 * (U_cb - real(U_cc));
    U.d12[ivar] = -2 * imag(U_cc);
  }

  *x = real(c);
  *r = imag(c);
}
/*-----------------------------------------------------------*/
void TwoPunctures::rx3_To_xyz(int nvar, double x, double r, double phi,
                              double *y, double *z, derivs U)
/* On Entrance: U.d0[]=U[]; U.d1[] =U[]_x;  U.d2[] =U[]_r;  U.d3[] =U[]_3;  */
/*                          U.d11[]=U[]_xx; U.d12[]=U[]_xr; U.d13[]=U[]_x3; */
/*                          U.d22[]=U[]_rr; U.d23[]=U[]_r3; U.d33[]=U[]_33; */
/* At Exit:     U.d0[]=U[]; U.d1[] =U[]_x;  U.d2[] =U[]_y;  U.dz[] =U[]_z;  */
/*                          U.d11[]=U[]_xx; U.d12[]=U[]_xy; U.d1z[]=U[]_xz; */
/*                          U.d22[]=U[]_yy; U.d2z[]=U[]_yz; U.dzz[]=U[]_zz; */
{
  int jvar;
  double
      sin_phi = sin(phi),
      cos_phi = cos(phi),
      sin2_phi = sin_phi * sin_phi,
      cos2_phi = cos_phi * cos_phi,
      sin_2phi = 2 * sin_phi * cos_phi,
      cos_2phi = cos2_phi - sin2_phi, r_inv = 1 / r, r_inv2 = r_inv * r_inv;

  *y = r * cos_phi;
  *z = r * sin_phi;

  for (jvar = 0; jvar < nvar; jvar++)
  {
    double U_x = U.d1[jvar], U_r = U.d2[jvar], U_3 = U.d3[jvar],
           U_xx = U.d11[jvar], U_xr = U.d12[jvar], U_x3 = U.d13[jvar],
           U_rr = U.d22[jvar], U_r3 = U.d23[jvar], U_33 = U.d33[jvar];
    U.d1[jvar] = U_x;                                                    /* U_x*/
    U.d2[jvar] = U_r * cos_phi - U_3 * r_inv * sin_phi;                  /* U_y*/
    U.d3[jvar] = U_r * sin_phi + U_3 * r_inv * cos_phi;                  /* U_z*/
    U.d11[jvar] = U_xx;                                                  /* U_xx*/
    U.d12[jvar] = U_xr * cos_phi - U_x3 * r_inv * sin_phi;               /* U_xy*/
    U.d13[jvar] = U_xr * sin_phi + U_x3 * r_inv * cos_phi;               /* U_xz*/
    U.d22[jvar] = U_rr * cos2_phi + r_inv2 * sin2_phi * (U_33 + r * U_r) /* U_yy*/
                  + sin_2phi * r_inv2 * (U_3 - r * U_r3);
    U.d23[jvar] = 0.5 * sin_2phi * (U_rr - r_inv * U_r - r_inv2 * U_33) /* U_yz*/
                  - cos_2phi * r_inv2 * (U_3 - r * U_r3);
    U.d33[jvar] = U_rr * sin2_phi + r_inv2 * cos2_phi * (U_33 + r * U_r) /* U_zz*/
                  - sin_2phi * r_inv2 * (U_3 - r * U_r3);
  }
}
/* --------------------------------------------------------------------------*/
void TwoPunctures::Derivatives_AB3(int nvar, int n1, int n2, int n3, derivs v)
{
  int i, j, k, ivar, N;

  N = maximum3(n1, n2, n3);

  /* Three direction passes, run in fixed order (A -> B -> phi) because each
     reads the previous pass' output (v.d1, v.d2).  Within each pass every
     (ivar, row) combination writes disjoint v.* entries, so the pass body is
     collapsed over (ivar, row) for parallelism; the implicit barrier between
     the three `#pragma omp for` blocks preserves the serial data order and
     gives bit-identical results.  All working arrays (indx, p, dp, d2p, q, dq,
     r, dr, plus the chebft/fourft scratch) are allocated ONCE per OpenMP
     thread and reused across every line that thread visits, eliminating ~77
     malloc/free pairs per line (the spectral transforms previously allocated
     their own scratch on every call). */

#pragma omp parallel shared(nvar, n1, n2, n3, N, v) private(i, j, k)
  {
    int *indx = ivector(0, N);
    double *p = dvector(0, N);
    double *dp = dvector(0, N);
    double *d2p = dvector(0, N);
    double *q = dvector(0, N);
    double *dq = dvector(0, N);
    double *r = dvector(0, N);
    double *dr = dvector(0, N);
    double *cb = dvector(0, N);
    double *ab = dvector(0, N);
    double *bb = dvector(0, N);

    /* Derivatives w.r.t. A-Dir. (Chebyshev along i) */
#pragma omp for collapse(2) schedule(static)
    for (ivar = 0; ivar < nvar; ivar++)
      for (k = 0; k < n3; k++)
      {
        for (j = 0; j < n2; j++)
        {
          for (i = 0; i < n1; i++)
          {
            indx[i] = Index(ivar, i, j, k, nvar, n1, n2, n3);
            p[i] = v.d0[indx[i]];
          }
          chebft_Zeros_sc(p, n1, 0, cb);
          chder(p, dp, n1);
          chder(dp, d2p, n1);
          chebft_Zeros_sc(dp, n1, 1, cb);
          chebft_Zeros_sc(d2p, n1, 1, cb);
          for (i = 0; i < n1; i++)
          {
            v.d1[indx[i]] = dp[i];
            v.d11[indx[i]] = d2p[i];
          }
        }
      }
    /* Derivatives w.r.t. B-Dir. (Chebyshev along j) */
#pragma omp for collapse(2) schedule(static)
    for (ivar = 0; ivar < nvar; ivar++)
      for (k = 0; k < n3; k++)
      {
        for (i = 0; i < n1; i++)
        {
          for (j = 0; j < n2; j++)
          {
            indx[j] = Index(ivar, i, j, k, nvar, n1, n2, n3);
            p[j] = v.d0[indx[j]];
            q[j] = v.d1[indx[j]];
          }
          chebft_Zeros_sc(p, n2, 0, cb);
          chebft_Zeros_sc(q, n2, 0, cb);
          chder(p, dp, n2);
          chder(dp, d2p, n2);
          chder(q, dq, n2);
          chebft_Zeros_sc(dp, n2, 1, cb);
          chebft_Zeros_sc(d2p, n2, 1, cb);
          chebft_Zeros_sc(dq, n2, 1, cb);
          for (j = 0; j < n2; j++)
          {
            v.d2[indx[j]] = dp[j];
            v.d22[indx[j]] = d2p[j];
            v.d12[indx[j]] = dq[j];
          }
        }
      }
    /* Derivatives w.r.t. phi-Dir. (Fourier along k) */
#pragma omp for collapse(2) schedule(static)
    for (ivar = 0; ivar < nvar; ivar++)
      for (i = 0; i < n1; i++)
      {
        for (j = 0; j < n2; j++)
        {
          for (k = 0; k < n3; k++)
          {
            indx[k] = Index(ivar, i, j, k, nvar, n1, n2, n3);
            p[k] = v.d0[indx[k]];
            q[k] = v.d1[indx[k]];
            r[k] = v.d2[indx[k]];
          }
          fourft_sc(p, n3, 0, ab, bb);
          fourder(p, dp, n3);
          fourder2(p, d2p, n3);
          fourft_sc(dp, n3, 1, ab, bb);
          fourft_sc(d2p, n3, 1, ab, bb);
          fourft_sc(q, n3, 0, ab, bb);
          fourder(q, dq, n3);
          fourft_sc(dq, n3, 1, ab, bb);
          fourft_sc(r, n3, 0, ab, bb);
          fourder(r, dr, n3);
          fourft_sc(dr, n3, 1, ab, bb);
          for (k = 0; k < n3; k++)
          {
            v.d3[indx[k]] = dp[k];
            v.d33[indx[k]] = d2p[k];
            v.d13[indx[k]] = dq[k];
            v.d23[indx[k]] = dr[k];
          }
        }
      }

    free_ivector(indx, 0, N);
    free_dvector(p, 0, N);
    free_dvector(dp, 0, N);
    free_dvector(d2p, 0, N);
    free_dvector(q, 0, N);
    free_dvector(dq, 0, N);
    free_dvector(r, 0, N);
    free_dvector(dr, 0, N);
    free_dvector(cb, 0, N);
    free_dvector(ab, 0, N);
    free_dvector(bb, 0, N);
  }
}
/* --------------------------------------------------------------------------*/
void TwoPunctures::Newton(int const nvar, int const n1, int const n2, int const n3,
                          derivs v, double const tol, int const itmax)
{
  int ntotal = n1 * n2 * n3 * nvar, ii, it;
  double *F, dmax, normres;
  derivs u, dv;
  const double _nw0 = omp_get_wtime();

  F = dvector(0, ntotal - 1);
  allocate_derivs(&dv, ntotal);
  allocate_derivs(&u, ntotal);

  it = 0;
  dmax = 1;
  while (dmax > tol && it < itmax)
  {
    if (it == 0)
    {
      F_of_v(nvar, n1, n2, n3, v, F, u);
      dmax = norm_inf(F, ntotal);
    }
    for (int j = 0; j < ntotal; j++)
      dv.d0[j] = 0;

    {
      printf("Newton: it=%d \t |F|=%e\n", it, (double)dmax);
      printf("bare mass: mp=%g \t mm=%g\n", (double)par_m_plus, (double)par_m_minus);
    }

    fflush(stdout);
    ii = bicgstab(nvar, n1, n2, n3, v, dv, 100, dmax * 1.e-3, &normres);

    for (int j = 0; j < ntotal; j++)
      v.d0[j] -= dv.d0[j];
    F_of_v(nvar, n1, n2, n3, v, F, u);
    dmax = norm_inf(F, ntotal);
    it += 1;
  }
  if (itmax == 0)
  {
    F_of_v(nvar, n1, n2, n3, v, F, u);
    dmax = norm_inf(F, ntotal);
  }

  printf("Newton: it=%d \t |F|=%e \n", it, (double)dmax);
  printf("Newton wall: %.3f s\n", omp_get_wtime() - _nw0);

  fflush(stdout);

  free_dvector(F, 0, ntotal - 1);
  free_derivs(&dv, ntotal);
  free_derivs(&u, ntotal);
}
#define FAC sin(al) * sin(be) * sin(al) * sin(be) * sin(al) * sin(be)
/* --------------------------------------------------------------------------*/
void TwoPunctures::F_of_v(int nvar, int n1, int n2, int n3, derivs v, double *F,
                          derivs u)
{
  /*      Calculates the left hand sides of the non-linear equations F_m(v_n)=0*/
  /*      and the function u (u.d0[]) as well as its derivatives*/
  /*      (u.d1[], u.d2[], u.d3[], u.d11[], u.d12[], u.d13[], u.d22[], u.d23[], u.d33[])*/
  /*      at interior points and at the boundaries "+/-"*/

  int i, j, k, ivar, indx;
  double al, be, A, B, X, R, x, r, phi, y, z, Am1, *values;
  derivs U;
  double *sources;

  values = dvector(0, nvar - 1);
  allocate_derivs(&U, nvar);

  sources = (double *)calloc(n1 * n2 * n3, sizeof(double));
  if (0)
  {
    double *s_x, *s_y, *s_z;
    int i3D;
    s_x = (double *)calloc(n1 * n2 * n3, sizeof(double));
    s_y = (double *)calloc(n1 * n2 * n3, sizeof(double));
    s_z = (double *)calloc(n1 * n2 * n3, sizeof(double));
    for (i = 0; i < n1; i++)
      for (j = 0; j < n2; j++)
        for (k = 0; k < n3; k++)
        {
          i3D = Index(0, i, j, k, 1, n1, n2, n3);

          al = Pih * (2 * i + 1) / n1;
          A = -cos(al);
          be = Pih * (2 * j + 1) / n2;
          B = -cos(be);
          phi = 2. * Pi * k / n3;

          Am1 = A - 1;
          for (ivar = 0; ivar < nvar; ivar++)
          {
            indx = Index(ivar, i, j, k, nvar, n1, n2, n3);
            U.d0[ivar] = Am1 * v.d0[indx];                    /* U*/
            U.d1[ivar] = v.d0[indx] + Am1 * v.d1[indx];       /* U_A*/
            U.d2[ivar] = Am1 * v.d2[indx];                    /* U_B*/
            U.d3[ivar] = Am1 * v.d3[indx];                    /* U_3*/
            U.d11[ivar] = 2 * v.d1[indx] + Am1 * v.d11[indx]; /* U_AA*/
            U.d12[ivar] = v.d2[indx] + Am1 * v.d12[indx];     /* U_AB*/
            U.d13[ivar] = v.d3[indx] + Am1 * v.d13[indx];     /* U_AB*/
            U.d22[ivar] = Am1 * v.d22[indx];                  /* U_BB*/
            U.d23[ivar] = Am1 * v.d23[indx];                  /* U_B3*/
            U.d33[ivar] = Am1 * v.d33[indx];                  /* U_33*/
          }
          /* Calculation of (X,R) and*/
          /* (U_X, U_R, U_3, U_XX, U_XR, U_X3, U_RR, U_R3, U_33)*/
          AB_To_XR(nvar, A, B, &X, &R, U);
          /* Calculation of (x,r) and*/
          /* (U, U_x, U_r, U_3, U_xx, U_xr, U_x3, U_rr, U_r3, U_33)*/
          C_To_c(nvar, X, R, &(s_x[i3D]), &r, U);
          /* Calculation of (y,z) and*/
          /* (U, U_x, U_y, U_z, U_xx, U_xy, U_xz, U_yy, U_yz, U_zz)*/
          rx3_To_xyz(nvar, s_x[i3D], r, phi, &(s_y[i3D]), &(s_z[i3D]), U);
        }
    free(s_z);
    free(s_y);
    free(s_x);
  }
  else
    for (i = 0; i < n1; i++)
      for (j = 0; j < n2; j++)
        for (k = 0; k < n3; k++)
          sources[Index(0, i, j, k, 1, n1, n2, n3)] = 0.0;

  Derivatives_AB3(nvar, n1, n2, n3, v);
  double psi, psi2, psi4, psi7, r_plus, r_minus;
  FILE *debugfile = NULL;
  if (0)
  {
    debugfile = fopen("res.dat", "w");
    assert(debugfile);
  }
  for (i = 0; i < n1; i++)
  {
    for (j = 0; j < n2; j++)
    {
      for (k = 0; k < n3; k++)
      {

        al = Pih * (2 * i + 1) / n1;
        A = -cos(al);
        be = Pih * (2 * j + 1) / n2;
        B = -cos(be);
        phi = 2. * Pi * k / n3;

        Am1 = A - 1;
        for (ivar = 0; ivar < nvar; ivar++)
        {
          indx = Index(ivar, i, j, k, nvar, n1, n2, n3);
          U.d0[ivar] = Am1 * v.d0[indx];                    /* U*/
          U.d1[ivar] = v.d0[indx] + Am1 * v.d1[indx];       /* U_A*/
          U.d2[ivar] = Am1 * v.d2[indx];                    /* U_B*/
          U.d3[ivar] = Am1 * v.d3[indx];                    /* U_3*/
          U.d11[ivar] = 2 * v.d1[indx] + Am1 * v.d11[indx]; /* U_AA*/
          U.d12[ivar] = v.d2[indx] + Am1 * v.d12[indx];     /* U_AB*/
          U.d13[ivar] = v.d3[indx] + Am1 * v.d13[indx];     /* U_AB*/
          U.d22[ivar] = Am1 * v.d22[indx];                  /* U_BB*/
          U.d23[ivar] = Am1 * v.d23[indx];                  /* U_B3*/
          U.d33[ivar] = Am1 * v.d33[indx];                  /* U_33*/
        }
        /* Calculation of (X,R) and*/
        /* (U_X, U_R, U_3, U_XX, U_XR, U_X3, U_RR, U_R3, U_33)*/
        AB_To_XR(nvar, A, B, &X, &R, U);
        /* Calculation of (x,r) and*/
        /* (U, U_x, U_r, U_3, U_xx, U_xr, U_x3, U_rr, U_r3, U_33)*/
        C_To_c(nvar, X, R, &x, &r, U);
        /* Calculation of (y,z) and*/
        /* (U, U_x, U_y, U_z, U_xx, U_xy, U_xz, U_yy, U_yz, U_zz)*/
        rx3_To_xyz(nvar, x, r, phi, &y, &z, U);
        NonLinEquations(sources[Index(0, i, j, k, 1, n1, n2, n3)],
                        A, B, X, R, x, r, phi, y, z, U, values);
        for (ivar = 0; ivar < nvar; ivar++)
        {
          indx = Index(ivar, i, j, k, nvar, n1, n2, n3);
          F[indx] = values[ivar] * FAC;
          /* if ((i<5) && ((j<5) || (j>n2-5)))*/
          /*     F[indx] = 0.0;*/
          u.d0[indx] = U.d0[ivar];   /*  U*/
          u.d1[indx] = U.d1[ivar];   /*      U_x*/
          u.d2[indx] = U.d2[ivar];   /*      U_y*/
          u.d3[indx] = U.d3[ivar];   /*      U_z*/
          u.d11[indx] = U.d11[ivar]; /*      U_xx*/
          u.d12[indx] = U.d12[ivar]; /*      U_xy*/
          u.d13[indx] = U.d13[ivar]; /*      U_xz*/
          u.d22[indx] = U.d22[ivar]; /*      U_yy*/
          u.d23[indx] = U.d23[ivar]; /*      U_yz*/
          u.d33[indx] = U.d33[ivar]; /*      U_zz*/
        }
        if (debugfile && (k == 0))
        {
          r_plus = sqrt((x - par_b) * (x - par_b) + y * y + z * z);
          r_minus = sqrt((x + par_b) * (x + par_b) + y * y + z * z);
          psi = 1. +
                0.5 * par_m_plus / r_plus +
                0.5 * par_m_minus / r_minus +
                U.d0[0];
          psi2 = psi * psi;
          psi4 = psi2 * psi2;
          psi7 = psi * psi2 * psi4;
          fprintf(debugfile,
                  "%.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g\n",
                  (double)x, (double)y, (double)A, (double)B,
                  (double)(U.d11[0] +
                           U.d22[0] +
                           U.d33[0] +
                           /*                      0.125 * BY_KKofxyz (x, y, z) / psi7 +*/
                           (2.0 * Pi / psi2 / psi * sources[indx]) * FAC),
                  (double)((U.d11[0] +
                            U.d22[0] +
                            U.d33[0]) *
                           FAC),
                  (double)(-(2.0 * Pi / psi2 / psi * sources[indx]) * FAC),
                  (double)sources[indx]
                  /*(double)F[indx]*/
          );
        }
      }
    }
  }
  if (debugfile)
  {
    fclose(debugfile);
  }
  free(sources);
  free_dvector(values, 0, nvar - 1);
  free_derivs(&U, nvar);
}
/* --------------------------------------------------------------------------*/
double TwoPunctures::norm_inf(double const *F, int const ntotal)
{
  double dmax = -1;
  {
    double dmax1 = -1;
    for (int j = 0; j < ntotal; j++)
      if (fabs(F[j]) > dmax1)
        dmax1 = fabs(F[j]);
    if (dmax1 > dmax)
      dmax = dmax1;
  }
  return dmax;
}
/* --------------------------------------------------------------------------*/
int TwoPunctures::bicgstab(int const nvar, int const n1, int const n2, int const n3,
                           derivs v, derivs dv, int const itmax, double const tol,
                           double *normres)
{
  int const output = 1;
  int ntotal = n1 * n2 * n3 * nvar, ii;
  double alpha = 0, beta = 0;
  double rho = 0, rho1 = 1, rhotol = 1e-50;
  double omega = 0, omegatol = 1e-50;
  double *p, *rt, *s, *t, *r, *vv;
  double **JFD;
  int **cols, *ncols, maxcol = StencilSize * nvar;
  double *F;
  derivs u, ph, sh;

  F = dvector(0, ntotal - 1);
  allocate_derivs(&u, ntotal);

  JFD = dmatrix(0, ntotal - 1, 0, maxcol - 1);
  cols = imatrix(0, ntotal - 1, 0, maxcol - 1);
  ncols = ivector(0, ntotal - 1);

  {
    double _a = omp_get_wtime();
    F_of_v(nvar, n1, n2, n3, v, F, u);
    t_F_of_v += omp_get_wtime() - _a;
  }
  {
    double _a = omp_get_wtime();
    SetMatrix_JFD(nvar, n1, n2, n3, u, ncols, cols, JFD);
    t_SetJFD += omp_get_wtime() - _a;
  }
  {
    double _a = omp_get_wtime();
    BuildLineFactors(nvar, n1, n2, n3, ncols, cols, JFD);
    t_BuildLF += omp_get_wtime() - _a;
  }

  /* temporary storage */
  r = dvector(0, ntotal - 1);
  p = dvector(0, ntotal - 1);
  allocate_derivs(&ph, ntotal);
  /*      ph  = dvector(0, ntotal-1);*/
  rt = dvector(0, ntotal - 1);
  s = dvector(0, ntotal - 1);
  allocate_derivs(&sh, ntotal);
  /*      sh  = dvector(0, ntotal-1);*/
  t = dvector(0, ntotal - 1);
  vv = dvector(0, ntotal - 1);

  /* check */
  if (output == 1)
  {
    printf("bicgstab:  itmax %d, tol %e\n", itmax, (double)tol);
    fflush(stdout);
  }

  /* compute initial residual rt = r = F - J*dv */
  {
    double _a = omp_get_wtime();
    J_times_dv(nvar, n1, n2, n3, dv, r, u);
    t_J_times_dv += omp_get_wtime() - _a;
  }
  for (int j = 0; j < ntotal; j++)
    rt[j] = r[j] = F[j] - r[j];

  *normres = norm2(r, ntotal);
  if (output == 1)
  {
    printf("bicgstab: %5d  %10.3e\n", 0, (double)*normres);
    fflush(stdout);
  }

  if (*normres <= tol)
    return 0;

  /* cgs iteration */
  for (ii = 0; ii < itmax; ii++)
  {
    rho = scalarproduct(rt, r, ntotal);
    if (fabs(rho) < rhotol)
      break;

    /* compute direction vector p */
    if (ii == 0)
    {
      for (int j = 0; j < ntotal; j++)
        p[j] = r[j];
    }
    else
    {
      beta = (rho / rho1) * (alpha / omega);
      for (int j = 0; j < ntotal; j++)
        p[j] = r[j] + beta * (p[j] - omega * vv[j]);
    }

    /* compute direction adjusting vector ph and scalar alpha */
    for (int j = 0; j < ntotal; j++)
      ph.d0[j] = 0;
    for (int j = 0; j < NRELAX; j++) /* solves JFD*ph = p by relaxation*/
      relax(ph.d0, nvar, n1, n2, n3, p, ncols, cols, JFD);

    {
      double _a = omp_get_wtime();
      J_times_dv(nvar, n1, n2, n3, ph, vv, u); /* vv=J*ph*/
      t_J_times_dv += omp_get_wtime() - _a;
    }
    alpha = rho / scalarproduct(rt, vv, ntotal);
    for (int j = 0; j < ntotal; j++)
      s[j] = r[j] - alpha * vv[j];

    /* early check of tolerance */
    *normres = norm2(s, ntotal);
    if (*normres <= tol)
    {
      for (int j = 0; j < ntotal; j++)
        dv.d0[j] += alpha * ph.d0[j];
      if (output == 1)
      {
        printf("bicgstab: %5d  %10.3e  %10.3e  %10.3e  %10.3e\n",
               ii + 1, (double)*normres, (double)alpha, (double)beta, (double)omega);
        fflush(stdout);
      }
      break;
    }

    /* compute stabilizer vector sh and scalar omega */
    for (int j = 0; j < ntotal; j++)
      sh.d0[j] = 0;
    for (int j = 0; j < NRELAX; j++) /* solves JFD*sh = s by relaxation*/
      relax(sh.d0, nvar, n1, n2, n3, s, ncols, cols, JFD);

    {
      double _a = omp_get_wtime();
      J_times_dv(nvar, n1, n2, n3, sh, t, u); /* t=J*sh*/
      t_J_times_dv += omp_get_wtime() - _a;
    }
    omega = scalarproduct(t, s, ntotal) / scalarproduct(t, t, ntotal);

    /* compute new solution approximation */
    for (int j = 0; j < ntotal; j++)
    {
      dv.d0[j] += alpha * ph.d0[j] + omega * sh.d0[j];
      r[j] = s[j] - omega * t[j];
    }
    /* are we done? */
    *normres = norm2(r, ntotal);
    if (output == 1)
    {
      printf("bicgstab: %5d  %10.3e  %10.3e  %10.3e  %10.3e\n",
             ii + 1, (double)*normres, (double)alpha, (double)beta, (double)omega);
      fflush(stdout);
    }
    if (*normres <= tol)
      break;
    rho1 = rho;
    if (fabs(omega) < omegatol)
      break;
  }

  /* free temporary storage */
  free_dvector(r, 0, ntotal - 1);
  free_dvector(p, 0, ntotal - 1);
  /*      free_dvector(ph,  0, ntotal-1);*/
  free_derivs(&ph, ntotal);
  free_dvector(rt, 0, ntotal - 1);
  free_dvector(s, 0, ntotal - 1);
  /*      free_dvector(sh,  0, ntotal-1);*/
  free_derivs(&sh, ntotal);
  free_dvector(t, 0, ntotal - 1);
  free_dvector(vv, 0, ntotal - 1);

  free_dvector(F, 0, ntotal - 1);
  free_derivs(&u, ntotal);

  free_dmatrix(JFD, 0, ntotal - 1, 0, maxcol - 1);
  free_imatrix(cols, 0, ntotal - 1, 0, maxcol - 1);
  free_ivector(ncols, 0, ntotal - 1);

  printf("relax profile: calls=%ld  omp-wall=%.3f s\n", relax_calls, relax_wall);
  printf("serial profile: F_of_v=%.3f  SetJFD=%.3f  BuildLF=%.3f  J_times_dv=%.3f  (spec=%.3f) s\n",
         t_F_of_v, t_SetJFD, t_BuildLF, t_J_times_dv, t_spec);
  fflush(stdout);

  /* iteration failed */
  if (ii > itmax)
    return -1;

  /* breakdown */
  if (fabs(rho) < rhotol)
    return -10;
  if (fabs(omega) < omegatol)
    return -11;

  /* success! */
  return ii + 1;
}
/* --------------------------------------------------------------------------*/
void TwoPunctures::allocate_derivs(derivs *v, int n)
{
  int m = n - 1;
  (*v).d0 = dvector(0, m);
  (*v).d1 = dvector(0, m);
  (*v).d2 = dvector(0, m);
  (*v).d3 = dvector(0, m);
  (*v).d11 = dvector(0, m);
  (*v).d12 = dvector(0, m);
  (*v).d13 = dvector(0, m);
  (*v).d22 = dvector(0, m);
  (*v).d23 = dvector(0, m);
  (*v).d33 = dvector(0, m);
}

/* --------------------------------------------------------------------------*/
void TwoPunctures::free_derivs(derivs *v, int n)
{
  int m = n - 1;
  free_dvector((*v).d0, 0, m);
  free_dvector((*v).d1, 0, m);
  free_dvector((*v).d2, 0, m);
  free_dvector((*v).d3, 0, m);
  free_dvector((*v).d11, 0, m);
  free_dvector((*v).d12, 0, m);
  free_dvector((*v).d13, 0, m);
  free_dvector((*v).d22, 0, m);
  free_dvector((*v).d23, 0, m);
  free_dvector((*v).d33, 0, m);
}
/* --------------------------------------------------------------------------*/
int TwoPunctures::Index(int ivar, int i, int j, int k, int nvar, int n1, int n2, int n3)
{
  int i1 = i, j1 = j, k1 = k;

  if (i1 < 0)
    i1 = -(i1 + 1);
  if (i1 >= n1)
    i1 = 2 * n1 - (i1 + 1);

  if (j1 < 0)
    j1 = -(j1 + 1);
  if (j1 >= n2)
    j1 = 2 * n2 - (j1 + 1);

  if (k1 < 0)
    k1 = k1 + n3;
  if (k1 >= n3)
    k1 = k1 - n3;

  return ivar + nvar * (i1 + n1 * (j1 + n2 * k1));
}
/*-----------------------------------------------------------*/
/********           Nonlinear Equations                ***********/
/*-----------------------------------------------------------*/
void TwoPunctures::NonLinEquations(double rho_adm,
                                   double A, double B, double X, double R,
                                   double x, double r, double phi,
                                   double y, double z, derivs U, double *values)
{
  double r_plus, r_minus, psi, psi2, psi4, psi7;
  double mu;

  r_plus = sqrt((x - par_b) * (x - par_b) + y * y + z * z);
  r_minus = sqrt((x + par_b) * (x + par_b) + y * y + z * z);

  psi = 1. + 0.5 * par_m_plus / r_plus + 0.5 * par_m_minus / r_minus + U.d0[0];
  psi2 = psi * psi;
  psi4 = psi2 * psi2;
  psi7 = psi * psi2 * psi4;

  values[0] = U.d11[0] + U.d22[0] + U.d33[0] + 0.125 * BY_KKofxyz(x, y, z) / psi7 + 2.0 * Pi / psi2 / psi * rho_adm;
}
double TwoPunctures::BY_KKofxyz(double x, double y, double z)
{
  int i, j;
  double r_plus, r2_plus, r3_plus, r_minus, r2_minus, r3_minus, np_Pp, nm_Pm,
      Aij, AijAij, n_plus[3], n_minus[3], np_Sp[3], nm_Sm[3];

  r2_plus = (x - par_b) * (x - par_b) + y * y + z * z;
  r2_minus = (x + par_b) * (x + par_b) + y * y + z * z;
  r_plus = sqrt(r2_plus);
  r_minus = sqrt(r2_minus);
  r3_plus = r_plus * r2_plus;
  r3_minus = r_minus * r2_minus;

  n_plus[0] = (x - par_b) / r_plus;
  n_minus[0] = (x + par_b) / r_minus;
  n_plus[1] = y / r_plus;
  n_minus[1] = y / r_minus;
  n_plus[2] = z / r_plus;
  n_minus[2] = z / r_minus;

  /* dot product: np_Pp = (n_+).(P_+); nm_Pm = (n_-).(P_-) */
  np_Pp = 0;
  nm_Pm = 0;
  for (i = 0; i < 3; i++)
  {
    np_Pp += n_plus[i] * par_P_plus[i];
    nm_Pm += n_minus[i] * par_P_minus[i];
  }
  /* cross product: np_Sp[i] = [(n_+) x (S_+)]_i; nm_Sm[i] = [(n_-) x (S_-)]_i*/
  np_Sp[0] = n_plus[1] * par_S_plus[2] - n_plus[2] * par_S_plus[1];
  np_Sp[1] = n_plus[2] * par_S_plus[0] - n_plus[0] * par_S_plus[2];
  np_Sp[2] = n_plus[0] * par_S_plus[1] - n_plus[1] * par_S_plus[0];
  nm_Sm[0] = n_minus[1] * par_S_minus[2] - n_minus[2] * par_S_minus[1];
  nm_Sm[1] = n_minus[2] * par_S_minus[0] - n_minus[0] * par_S_minus[2];
  nm_Sm[2] = n_minus[0] * par_S_minus[1] - n_minus[1] * par_S_minus[0];
  AijAij = 0;
  for (i = 0; i < 3; i++)
  {
    for (j = 0; j < 3; j++)
    { /* Bowen-York-Curvature :*/
      Aij =
          +1.5 * (par_P_plus[i] * n_plus[j] + par_P_plus[j] * n_plus[i] + np_Pp * n_plus[i] * n_plus[j]) / r2_plus + 1.5 * (par_P_minus[i] * n_minus[j] + par_P_minus[j] * n_minus[i] + nm_Pm * n_minus[i] * n_minus[j]) / r2_minus - 3.0 * (np_Sp[i] * n_plus[j] + np_Sp[j] * n_plus[i]) / r3_plus - 3.0 * (nm_Sm[i] * n_minus[j] + nm_Sm[j] * n_minus[i]) / r3_minus;
      if (i == j)
        Aij -= +1.5 * (np_Pp / r2_plus + nm_Pm / r2_minus);
      AijAij += Aij * Aij;
    }
  }

  return AijAij;
}
void TwoPunctures::SetMatrix_JFD(int nvar, int n1, int n2, int n3, derivs u,
                                 int *ncols, int **cols, double **Matrix)
{
  int column, row, mcol;
  int i, i1, i_0, i_1, j, j1, j_0, j_1, k, k1, k_0, k_1, N1, N2, N3,
      ivar, ivar1, ntotal = nvar * n1 * n2 * n3;
  double *values;
  derivs dv;
  const int maxnbr = 27; /* 3x3x3 stencil */

  values = dvector(0, nvar - 1);
  allocate_derivs(&dv, ntotal);

  /* Two-phase construction so the expensive JFD_times_dv evaluations run in
     parallel while the final table fill keeps the exact serial order:
       Phase 1 (parallel): each column independently evaluates its 27-stencil
         contributions using a private dv.d0 probe and a private values
         buffer, storing raw (row, value) pairs in tmp_row/tmp_val.
       Phase 2 (serial): walk the columns in the original (i,j,k) order and
         append the cached pairs into cols/Matrix/ncols.  Because phase 2
         visits rows in exactly the serial order, the resulting CSR layout is
         bit-identical to the serial code.  JFD_times_dv only reads dv.d0, so
         the private probe needs no other derivative arrays. */

  N1 = n1 - 1;
  N2 = n2 - 1;
  N3 = n3 - 1;

  for (i = 0; i < n1; i++)
  {
    for (j = 0; j < n2; j++)
    {
      for (k = 0; k < n3; k++)
      {
        for (ivar = 0; ivar < nvar; ivar++)
        {
          row = Index(ivar, i, j, k, nvar, n1, n2, n3);
          ncols[row] = 0;
          dv.d0[row] = 0;
        }
      }
    }
  }

  int *tmp_row = new int[ntotal * maxnbr];
  double *tmp_val = new double[ntotal * maxnbr];
  int *tmp_cnt = new int[ntotal];

#pragma omp parallel default(none)                            \
    shared(nvar, n1, n2, n3, u, ntotal, tmp_row, tmp_val, tmp_cnt, \
           N1, N2, N3) private(i, j, k, ivar, column, row, i1, j1, k1, \
           i_0, i_1, j_0, j_1, k_0, k_1, ivar1, mcol)
  {
    /* Phase 1: evaluate all stencil contributions in parallel. */
    double *d0p = new double[ntotal];
    for (i = 0; i < ntotal; i++)
      d0p[i] = 0.0;
    double *vals = dvector(0, nvar - 1);
    derivs dpv, dU, U;
    dpv.d0 = d0p;
    /* JFD_times_dv reads only dv.d0; the other derivs fields are unused. */
    dpv.d1 = dpv.d2 = dpv.d3 = 0;
    dpv.d11 = dpv.d12 = dpv.d13 = 0;
    dpv.d22 = dpv.d23 = dpv.d33 = 0;
    /* Scratch buffers reused across the whole thread's stencil sweep,
       avoiding ~20 malloc/free pairs per JFD_times_dv call. */
    allocate_derivs(&dU, nvar);
    allocate_derivs(&U, nvar);

#pragma omp for schedule(static)
    for (i = 0; i < n1; i++)
    {
      for (j = 0; j < n2; j++)
      {
        for (k = 0; k < n3; k++)
        {
          for (ivar = 0; ivar < nvar; ivar++)
          {
            column = Index(ivar, i, j, k, nvar, n1, n2, n3);
            d0p[column] = 1.0;

            i_0 = maximum2(0, i - 1);
            i_1 = minimum2(N1, i + 1);
            j_0 = maximum2(0, j - 1);
            j_1 = minimum2(N2, j + 1);
            k_0 = k - 1;
            k_1 = k + 1;

            int cnt = 0;
            for (i1 = i_0; i1 <= i_1; i1++)
            {
              for (j1 = j_0; j1 <= j_1; j1++)
              {
                for (k1 = k_0; k1 <= k_1; k1++)
                {
                  JFD_times_dv_sc(i1, j1, k1, nvar, n1, n2, n3,
                                  dpv, u, vals, dU, U);
                  for (ivar1 = 0; ivar1 < nvar; ivar1++)
                  {
                    if (vals[ivar1] != 0)
                    {
                      row = Index(ivar1, i1, j1, k1, nvar, n1, n2, n3);
                      tmp_row[column * maxnbr + cnt] = row;
                      tmp_val[column * maxnbr + cnt] = vals[ivar1];
                      cnt += 1;
                    }
                  }
                }
              }
            }
            tmp_cnt[column] = cnt;
            d0p[column] = 0.0;
          }
        }
      }
    }

    free_dvector(vals, 0, nvar - 1);
    free_derivs(&dU, nvar);
    free_derivs(&U, nvar);
    delete[] d0p;
  }

  /* Phase 2: serial fill preserving the original (i,j,k) column order. */
  for (i = 0; i < n1; i++)
  {
    for (j = 0; j < n2; j++)
    {
      for (k = 0; k < n3; k++)
      {
        for (ivar = 0; ivar < nvar; ivar++)
        {
          column = Index(ivar, i, j, k, nvar, n1, n2, n3);
          for (mcol = 0; mcol < tmp_cnt[column]; mcol++)
          {
            row = tmp_row[column * maxnbr + mcol];
            cols[row][ncols[row]] = column;
            Matrix[row][ncols[row]] = tmp_val[column * maxnbr + mcol];
            ncols[row] += 1;
          }
        }
      }
    }
  }

  delete[] tmp_row;
  delete[] tmp_val;
  delete[] tmp_cnt;

  free_derivs(&dv, ntotal);
  free_dvector(values, 0, nvar - 1);
}
/* --------------------------------------------------------------------------*/
void TwoPunctures::BuildLineFactors(int nvar, int n1, int n2, int n3,
                                    int *ncols, int **cols, double **JFD)
/* Precompute the LU factorization (with reciprocal diagonals) of every
   tridiagonal line used by the relax() preconditioner. The tridiagonal
   coefficients only depend on JFD, which is constant within a bicgstab
   iteration, so this is done once per bicgstab call. The relax sweeps then
   only assemble the right-hand side and do forward/backward substitution.
   be-lines run along j for each (i,k); al-lines run along i for each (j,k). */
{
  int i, j, k, m, col, Ic, Ip, Im;

  for (k = 0; k < n3; k++)
  {
    for (i = 0; i < n1; i++)
    {
      double diag[MAX_LINE_SIZE], e[MAX_LINE_SIZE], f[MAX_LINE_SIZE];
      for (j = 0; j < n2; j++)
      {
        Ip = Index(0, i, j + 1, k, nvar, n1, n2, n3);
        Ic = Index(0, i, j, k, nvar, n1, n2, n3);
        Im = Index(0, i, j - 1, k, nvar, n1, n2, n3);
        diag[j] = e[j] = f[j] = 0.0;
        for (m = 0; m < ncols[Ic]; m++)
        {
          col = cols[Ic][m];
          if (col == Im && j > 0)
            f[j - 1] = JFD[Ic][m];
          if (col == Ic)
            diag[j] = JFD[Ic][m];
          if (col == Ip && j < n2 - 1)
            e[j] = JFD[Ic][m];
        }
      }
      double *l = lu_be_l + (i * n3 + k) * n2;
      double *u = lu_be_u + (i * n3 + k) * n2;
      double *r = lu_be_r + (i * n3 + k) * n2;
      u[0] = e[0];
      r[0] = 1.0 / diag[0];
      for (j = 0; j < n2 - 2; j++)
      {
        l[j] = f[j] * r[j];
        u[j + 1] = e[j + 1];
        r[j + 1] = 1.0 / (diag[j + 1] - l[j] * u[j]);
      }
      l[n2 - 2] = f[n2 - 2] * r[n2 - 2];
      r[n2 - 1] = 1.0 / (diag[n2 - 1] - l[n2 - 2] * u[n2 - 2]);
    }
  }

  for (k = 0; k < n3; k++)
  {
    for (j = 0; j < n2; j++)
    {
      double diag[MAX_LINE_SIZE], e[MAX_LINE_SIZE], f[MAX_LINE_SIZE];
      for (i = 0; i < n1; i++)
      {
        Ip = Index(0, i + 1, j, k, nvar, n1, n2, n3);
        Ic = Index(0, i, j, k, nvar, n1, n2, n3);
        Im = Index(0, i - 1, j, k, nvar, n1, n2, n3);
        diag[i] = e[i] = f[i] = 0.0;
        for (m = 0; m < ncols[Ic]; m++)
        {
          col = cols[Ic][m];
          if (col == Im && i > 0)
            f[i - 1] = JFD[Ic][m];
          if (col == Ic)
            diag[i] = JFD[Ic][m];
          if (col == Ip && i < n1 - 1)
            e[i] = JFD[Ic][m];
        }
      }
      double *l = lu_al_l + (j * n3 + k) * n1;
      double *u = lu_al_u + (j * n3 + k) * n1;
      double *r = lu_al_r + (j * n3 + k) * n1;
      u[0] = e[0];
      r[0] = 1.0 / diag[0];
      for (i = 0; i < n1 - 2; i++)
      {
        l[i] = f[i] * r[i];
        u[i + 1] = e[i + 1];
        r[i + 1] = 1.0 / (diag[i + 1] - l[i] * u[i]);
      }
      l[n1 - 2] = f[n1 - 2] * r[n1 - 2];
      r[n1 - 1] = 1.0 / (diag[n1 - 1] - l[n1 - 2] * u[n1 - 2]);
    }
  }
}
/* --------------------------------------------------------------------------*/
void TwoPunctures::J_times_dv(int nvar, int n1, int n2, int n3, derivs dv, double *Jdv, derivs u)
{ /*      Calculates the left hand sides of the non-linear equations F_m(v_n)=0*/
  /*      and the function u (u.d0[]) as well as its derivatives*/
  /*      (u.d1[], u.d2[], u.d3[], u.d11[], u.d12[], u.d13[], u.d22[], u.d23[], u.d33[])*/
  /*      at interior points and at the boundaries "+/-"*/
  int i, j, k, ivar, indx;
  double al, be, A, B, X, R, x, r, phi, y, z, Am1, *values;
  derivs dU, U;

  {
    double _a = omp_get_wtime();
    Derivatives_AB3(nvar, n1, n2, n3, dv);
    t_spec += omp_get_wtime() - _a;
  }

  /* Each (i,j,k) writes only Jdv[indx], so the point loop is embarrassingly
     parallel.  values/dU/U are per-iteration temporaries; they are allocated
     ONCE per OpenMP thread (outside the k loop) and reused across all points
     that the thread visits, eliminating ~21 malloc/free calls per grid point.
     Derivatives_AB3 above must stay serial: it writes the shared dv.d1..d33
     arrays that every point reads.  Works with any nvar via heap buffers that
     are hoisted per thread. */
#pragma omp parallel shared(dv, Jdv, u, nvar, n1, n2, n3) private(i, j, k, ivar, indx)
  {
    double *values = dvector(0, nvar - 1);
    derivs dU, U;
    allocate_derivs(&dU, nvar);
    allocate_derivs(&U, nvar);

#pragma omp for schedule(static)
    for (i = 0; i < n1; i++)
    {
      for (j = 0; j < n2; j++)
      {
        for (k = 0; k < n3; k++)
        {
          double al, be, A, B, X, R, x, r, phi, y, z, Am1;

          al = Pih * (2 * i + 1) / n1;
          A = -cos(al);
          be = Pih * (2 * j + 1) / n2;
          B = -cos(be);
          phi = 2. * Pi * k / n3;

          Am1 = A - 1;
          for (ivar = 0; ivar < nvar; ivar++)
          {
            indx = Index(ivar, i, j, k, nvar, n1, n2, n3);
            dU.d0[ivar] = Am1 * dv.d0[indx];                     /* dU*/
            dU.d1[ivar] = dv.d0[indx] + Am1 * dv.d1[indx];       /* dU_A*/
            dU.d2[ivar] = Am1 * dv.d2[indx];                     /* dU_B*/
            dU.d3[ivar] = Am1 * dv.d3[indx];                     /* dU_3*/
            dU.d11[ivar] = 2 * dv.d1[indx] + Am1 * dv.d11[indx]; /* dU_AA*/
            dU.d12[ivar] = dv.d2[indx] + Am1 * dv.d12[indx];     /* dU_AB*/
            dU.d13[ivar] = dv.d3[indx] + Am1 * dv.d13[indx];     /* dU_AB*/
            dU.d22[ivar] = Am1 * dv.d22[indx];                   /* dU_BB*/
            dU.d23[ivar] = Am1 * dv.d23[indx];                   /* dU_B3*/
            dU.d33[ivar] = Am1 * dv.d33[indx];                   /* dU_33*/
            U.d0[ivar] = u.d0[indx];                             /* U   */
            U.d1[ivar] = u.d1[indx];                             /* U_x*/
            U.d2[ivar] = u.d2[indx];                             /* U_y*/
            U.d3[ivar] = u.d3[indx];                             /* U_z*/
            U.d11[ivar] = u.d11[indx];                           /* U_xx*/
            U.d12[ivar] = u.d12[indx];                           /* U_xy*/
            U.d13[ivar] = u.d13[indx];                           /* U_xz*/
            U.d22[ivar] = u.d22[indx];                           /* U_yy*/
            U.d23[ivar] = u.d23[indx];                           /* U_yz*/
            U.d33[ivar] = u.d33[indx];                           /* U_zz*/
          }
          /* Calculation of (X,R) and*/
          /* (dU_X, dU_R, dU_3, dU_XX, dU_XR, dU_X3, dU_RR, dU_R3, dU_33)*/
          AB_To_XR(nvar, A, B, &X, &R, dU);
          /* Calculation of (x,r) and*/
          /* (dU, dU_x, dU_r, dU_3, dU_xx, dU_xr, dU_x3, dU_rr, dU_r3, dU_33)*/
          C_To_c(nvar, X, R, &x, &r, dU);
          /* Calculation of (y,z) and*/
          /* (dU, dU_x, dU_y, dU_z, dU_xx, dU_xy, dU_xz, dU_yy, dU_yz, dU_zz)*/
          rx3_To_xyz(nvar, x, r, phi, &y, &z, dU);
          LinEquations(A, B, X, R, x, r, phi, y, z, dU, U, values);
          for (ivar = 0; ivar < nvar; ivar++)
          {
            indx = Index(ivar, i, j, k, nvar, n1, n2, n3);
            Jdv[indx] = values[ivar] * FAC;
          }
        }
      }
    }

    free_dvector(values, 0, nvar - 1);
    free_derivs(&dU, nvar);
    free_derivs(&U, nvar);
  }
}
/* --------------------------------------------------------------------------*/
void TwoPunctures::relax(double *dv, int const nvar, int const n1, int const n2, int const n3,
                         double const *rhs, int const *ncols, int **cols, double **JFD)
{
  /* OpenMP red-black line relaxation.
     The relaxation is a 3D red-black sweep: the k-planes are split into
     even/odd passes, and within each plane the be-lines (along j) and
     al-lines (along i) are ordered by red-black parity in i and j.  Every
     phase only reads values of opposite parity, which are NOT modified
     during that phase, so all lines of one phase are mutually independent
     and can run in parallel.  The implicit barriers of the consecutive
     `#pragma omp for` constructs preserve the exact sequential sweep order
     (bit-identical results).  Each phase is collapsed over (k, parity-i)
     or (k, parity-j) to expose ~(n3/2)*(n1/2) independent lines.
     N_PlaneRelax sweeps are repeated per parity pass as in the serial code. */

#pragma omp parallel default(none)                                    \
    shared(dv, nvar, n1, n2, n3, rhs, ncols, cols, JFD, relax_wall, relax_calls)
  {
    const double t0 = omp_get_wtime();
    for (int n = 0; n < N_PlaneRelax; n++)
    {
#pragma omp for collapse(2) schedule(static)
      for (int k = 0; k < n3; k += 2)
        for (int i = 2; i < n1; i += 2)
          LineRelax_be(dv, i, k, nvar, n1, n2, n3, rhs, ncols, cols, JFD);
#pragma omp for collapse(2) schedule(static)
      for (int k = 0; k < n3; k += 2)
        for (int i = 1; i < n1; i += 2)
          LineRelax_be(dv, i, k, nvar, n1, n2, n3, rhs, ncols, cols, JFD);
#pragma omp for collapse(2) schedule(static)
      for (int k = 0; k < n3; k += 2)
        for (int j = 1; j < n2; j += 2)
          LineRelax_al(dv, j, k, nvar, n1, n2, n3, rhs, ncols, cols, JFD);
#pragma omp for collapse(2) schedule(static)
      for (int k = 0; k < n3; k += 2)
        for (int j = 0; j < n2; j += 2)
          LineRelax_al(dv, j, k, nvar, n1, n2, n3, rhs, ncols, cols, JFD);
    }
    for (int n = 0; n < N_PlaneRelax; n++)
    {
#pragma omp for collapse(2) schedule(static)
      for (int k = 1; k < n3; k += 2)
        for (int i = 0; i < n1; i += 2)
          LineRelax_be(dv, i, k, nvar, n1, n2, n3, rhs, ncols, cols, JFD);
#pragma omp for collapse(2) schedule(static)
      for (int k = 1; k < n3; k += 2)
        for (int i = 1; i < n1; i += 2)
          LineRelax_be(dv, i, k, nvar, n1, n2, n3, rhs, ncols, cols, JFD);
#pragma omp for collapse(2) schedule(static)
      for (int k = 1; k < n3; k += 2)
        for (int j = 1; j < n2; j += 2)
          LineRelax_al(dv, j, k, nvar, n1, n2, n3, rhs, ncols, cols, JFD);
#pragma omp for collapse(2) schedule(static)
      for (int k = 1; k < n3; k += 2)
        for (int j = 0; j < n2; j += 2)
          LineRelax_al(dv, j, k, nvar, n1, n2, n3, rhs, ncols, cols, JFD);
    }
    const double t1 = omp_get_wtime();
#pragma omp master
    {
      relax_wall += t1 - t0;
      relax_calls += 1;
    }
  }
}
/* --------------------------------------------------------------------------*/
void TwoPunctures::LineRelax_be(double *dv,
                                int const i, int const k, int const nvar,
                                int const n1, int const n2, int const n3,
                                double const *rhs, int const *ncols, int **cols,
                                double **JFD)
{
  int j, m, Ic, Ip, Im, col, ivar;

  double b[MAX_LINE_SIZE];
  double x[MAX_LINE_SIZE];
  double y[MAX_LINE_SIZE];

  /* Precomputed LU factors for line (i,k) (see BuildLineFactors) */
  const double *l = lu_be_l + (i * n3 + k) * n2;
  const double *u = lu_be_u + (i * n3 + k) * n2;
  const double *r = lu_be_r + (i * n3 + k) * n2;

  for (ivar = 0; ivar < nvar; ivar++)
  {
    for (j = 0; j < n2; j++)
    {
      Ip = Index(ivar, i, j + 1, k, nvar, n1, n2, n3);
      Ic = Index(ivar, i, j, k, nvar, n1, n2, n3);
      Im = Index(ivar, i, j - 1, k, nvar, n1, n2, n3);
      b[j] = rhs[Ic];
      for (m = 0; m < ncols[Ic]; m++)
      {
        col = cols[Ic][m];
        if (col != Ip && col != Ic && col != Im)
          b[j] -= JFD[Ic][m] * dv[col];
      }
    }
    /* Forward substitution [L][y] = [b] */
    y[0] = b[0];
    for (j = 1; j < n2; j++)
      y[j] = b[j] - l[j - 1] * y[j - 1];
    /* Backward substitution [U][x] = [y] */
    x[n2 - 1] = y[n2 - 1] * r[n2 - 1];
    for (j = n2 - 2; j >= 0; j--)
      x[j] = (y[j] - u[j] * x[j + 1]) * r[j];
    for (j = 0; j < n2; j++)
    {
      Ic = Index(ivar, i, j, k, nvar, n1, n2, n3);
      dv[Ic] = x[j];
    }
  }
}
/* --------------------------------------------------------------------------*/
void TwoPunctures::JFD_times_dv(int i, int j, int k, int nvar, int n1, int n2,
                                int n3, derivs dv, derivs u, double *values)
{ /* Calculates rows of the vector 'J(FD)*dv'.*/
  /* First row to be calculated: row = Index(0,      i, j, k; nvar, n1, n2, n3)*/
  /* Last  row to be calculated: row = Index(nvar-1, i, j, k; nvar, n1, n2, n3)*/
  /* These rows are stored in the vector JFDdv[0] ... JFDdv[nvar-1].*/
  derivs dU, U;
  allocate_derivs(&dU, nvar);
  allocate_derivs(&U, nvar);
  JFD_times_dv_sc(i, j, k, nvar, n1, n2, n3, dv, u, values, dU, U);
  free_derivs(&dU, nvar);
  free_derivs(&U, nvar);
}

void TwoPunctures::JFD_times_dv_sc(int i, int j, int k, int nvar, int n1, int n2,
                                   int n3, derivs dv, derivs u, double *values,
                                   derivs dU, derivs U)
{ /* Scratch variant of JFD_times_dv: dU and U must be caller-owned derivs
     buffers of size nvar.  This lets SetMatrix_JFD's hot 3x3x3 stencil loop
     allocate every working array once per OpenMP thread instead of ~20
     malloc/free pairs per stencil point (~35M allocations per matrix build). */
  int ivar, indx;
  double al, be, A, B, X, R, x, r, phi, y, z, Am1;
  double sin_al, sin_al_i1, sin_al_i2, sin_al_i3, cos_al;
  double sin_be, sin_be_i1, sin_be_i2, sin_be_i3, cos_be;
  double dV0, dV1, dV2, dV3, dV11, dV12, dV13, dV22, dV23, dV33,
      ha, ga, ga2, hb, gb, gb2, hp, gp, gp2, gagb, gagp, gbgp;

  if (k < 0)
    k = k + n3;
  if (k >= n3)
    k = k - n3;

  ha = Pi / n1; /* ha: Stepsize with respect to (al)*/
  al = ha * (i + 0.5);
  A = -cos(al);
  ga = 1 / ha;
  ga2 = ga * ga;

  hb = Pi / n2; /* hb: Stepsize with respect to (be)*/
  be = hb * (j + 0.5);
  B = -cos(be);
  gb = 1 / hb;
  gb2 = gb * gb;
  gagb = ga * gb;

  hp = 2 * Pi / n3; /* hp: Stepsize with respect to (phi)*/
  phi = hp * j;
  gp = 1 / hp;
  gp2 = gp * gp;
  gagp = ga * gp;
  gbgp = gb * gp;

  sin_al = sin(al);
  sin_be = sin(be);
  sin_al_i1 = 1 / sin_al;
  sin_be_i1 = 1 / sin_be;
  sin_al_i2 = sin_al_i1 * sin_al_i1;
  sin_be_i2 = sin_be_i1 * sin_be_i1;
  sin_al_i3 = sin_al_i1 * sin_al_i2;
  sin_be_i3 = sin_be_i1 * sin_be_i2;
  cos_al = -A;
  cos_be = -B;

  Am1 = A - 1;
  for (ivar = 0; ivar < nvar; ivar++)
  {
    int iccc = Index(ivar, i, j, k, nvar, n1, n2, n3),
        ipcc = Index(ivar, i + 1, j, k, nvar, n1, n2, n3),
        imcc = Index(ivar, i - 1, j, k, nvar, n1, n2, n3),
        icpc = Index(ivar, i, j + 1, k, nvar, n1, n2, n3),
        icmc = Index(ivar, i, j - 1, k, nvar, n1, n2, n3),
        iccp = Index(ivar, i, j, k + 1, nvar, n1, n2, n3),
        iccm = Index(ivar, i, j, k - 1, nvar, n1, n2, n3),
        icpp = Index(ivar, i, j + 1, k + 1, nvar, n1, n2, n3),
        icmp = Index(ivar, i, j - 1, k + 1, nvar, n1, n2, n3),
        icpm = Index(ivar, i, j + 1, k - 1, nvar, n1, n2, n3),
        icmm = Index(ivar, i, j - 1, k - 1, nvar, n1, n2, n3),
        ipcp = Index(ivar, i + 1, j, k + 1, nvar, n1, n2, n3),
        imcp = Index(ivar, i - 1, j, k + 1, nvar, n1, n2, n3),
        ipcm = Index(ivar, i + 1, j, k - 1, nvar, n1, n2, n3),
        imcm = Index(ivar, i - 1, j, k - 1, nvar, n1, n2, n3),
        ippc = Index(ivar, i + 1, j + 1, k, nvar, n1, n2, n3),
        impc = Index(ivar, i - 1, j + 1, k, nvar, n1, n2, n3),
        ipmc = Index(ivar, i + 1, j - 1, k, nvar, n1, n2, n3),
        immc = Index(ivar, i - 1, j - 1, k, nvar, n1, n2, n3);
    /* Derivatives of (dv) w.r.t. (al,be,phi):*/
    dV0 = dv.d0[iccc];
    dV1 = 0.5 * ga * (dv.d0[ipcc] - dv.d0[imcc]);
    dV2 = 0.5 * gb * (dv.d0[icpc] - dv.d0[icmc]);
    dV3 = 0.5 * gp * (dv.d0[iccp] - dv.d0[iccm]);
    dV11 = ga2 * (dv.d0[ipcc] + dv.d0[imcc] - 2 * dv.d0[iccc]);
    dV22 = gb2 * (dv.d0[icpc] + dv.d0[icmc] - 2 * dv.d0[iccc]);
    dV33 = gp2 * (dv.d0[iccp] + dv.d0[iccm] - 2 * dv.d0[iccc]);
    dV12 =
        0.25 * gagb * (dv.d0[ippc] - dv.d0[ipmc] + dv.d0[immc] - dv.d0[impc]);
    dV13 =
        0.25 * gagp * (dv.d0[ipcp] - dv.d0[imcp] + dv.d0[imcm] - dv.d0[ipcm]);
    dV23 =
        0.25 * gbgp * (dv.d0[icpp] - dv.d0[icpm] + dv.d0[icmm] - dv.d0[icmp]);
    /* Derivatives of (dv) w.r.t. (A,B,phi):*/
    dV11 = sin_al_i3 * (sin_al * dV11 - cos_al * dV1);
    dV12 = sin_al_i1 * sin_be_i1 * dV12;
    dV13 = sin_al_i1 * dV13;
    dV22 = sin_be_i3 * (sin_be * dV22 - cos_be * dV2);
    dV23 = sin_be_i1 * dV23;
    dV1 = sin_al_i1 * dV1;
    dV2 = sin_be_i1 * dV2;
    /* Derivatives of (dU) w.r.t. (A,B,phi):*/
    dU.d0[ivar] = Am1 * dV0;
    dU.d1[ivar] = dV0 + Am1 * dV1;
    dU.d2[ivar] = Am1 * dV2;
    dU.d3[ivar] = Am1 * dV3;
    dU.d11[ivar] = 2 * dV1 + Am1 * dV11;
    dU.d12[ivar] = dV2 + Am1 * dV12;
    dU.d13[ivar] = dV3 + Am1 * dV13;
    dU.d22[ivar] = Am1 * dV22;
    dU.d23[ivar] = Am1 * dV23;
    dU.d33[ivar] = Am1 * dV33;

    indx = Index(ivar, i, j, k, nvar, n1, n2, n3);
    U.d0[ivar] = u.d0[indx];   /* U   */
    U.d1[ivar] = u.d1[indx];   /* U_x*/
    U.d2[ivar] = u.d2[indx];   /* U_y*/
    U.d3[ivar] = u.d3[indx];   /* U_z*/
    U.d11[ivar] = u.d11[indx]; /* U_xx*/
    U.d12[ivar] = u.d12[indx]; /* U_xy*/
    U.d13[ivar] = u.d13[indx]; /* U_xz*/
    U.d22[ivar] = u.d22[indx]; /* U_yy*/
    U.d23[ivar] = u.d23[indx]; /* U_yz*/
    U.d33[ivar] = u.d33[indx]; /* U_zz*/
  }
  /* Calculation of (X,R) and*/
  /* (dU_X, dU_R, dU_3, dU_XX, dU_XR, dU_X3, dU_RR, dU_R3, dU_33)*/
  AB_To_XR(nvar, A, B, &X, &R, dU);
  /* Calculation of (x,r) and*/
  /* (dU, dU_x, dU_r, dU_3, dU_xx, dU_xr, dU_x3, dU_rr, dU_r3, dU_33)*/
  C_To_c(nvar, X, R, &x, &r, dU);
  /* Calculation of (y,z) and*/
  /* (dU, dU_x, dU_y, dU_z, dU_xx, dU_xy, dU_xz, dU_yy, dU_yz, dU_zz)*/
  rx3_To_xyz(nvar, x, r, phi, &y, &z, dU);
  LinEquations(A, B, X, R, x, r, phi, y, z, dU, U, values);
  for (ivar = 0; ivar < nvar; ivar++)
    values[ivar] *= FAC;
}
#undef FAC
/*-----------------------------------------------------------*/
/********               Linear Equations                ***********/
/*-----------------------------------------------------------*/
void TwoPunctures::LinEquations(double A, double B, double X, double R,
                                double x, double r, double phi,
                                double y, double z, derivs dU, derivs U, double *values)
{
  double r_plus, r_minus, psi, psi2, psi4, psi8;

  r_plus = sqrt((x - par_b) * (x - par_b) + y * y + z * z);
  r_minus = sqrt((x + par_b) * (x + par_b) + y * y + z * z);

  psi =
      1. + 0.5 * par_m_plus / r_plus + 0.5 * par_m_minus / r_minus + U.d0[0];
  psi2 = psi * psi;
  psi4 = psi2 * psi2;
  psi8 = psi4 * psi4;

  values[0] = dU.d11[0] + dU.d22[0] + dU.d33[0] - 0.875 * BY_KKofxyz(x, y, z) / psi8 * dU.d0[0];
}
/* -------------------------------------------------------------------------*/
void TwoPunctures::LineRelax_al(double *dv,
                                int const j, int const k, int const nvar,
                                int const n1, int const n2, int const n3,
                                double const *rhs, int const *ncols,
                                int **cols, double **JFD)
{
  int i, m, Ic, Ip, Im, col, ivar;

  double b[MAX_LINE_SIZE];
  double x[MAX_LINE_SIZE];
  double y[MAX_LINE_SIZE];

  /* Precomputed LU factors for line (j,k) (see BuildLineFactors) */
  const double *l = lu_al_l + (j * n3 + k) * n1;
  const double *u = lu_al_u + (j * n3 + k) * n1;
  const double *r = lu_al_r + (j * n3 + k) * n1;

  for (ivar = 0; ivar < nvar; ivar++)
  {
    for (i = 0; i < n1; i++)
    {
      Ip = Index(ivar, i + 1, j, k, nvar, n1, n2, n3);
      Ic = Index(ivar, i, j, k, nvar, n1, n2, n3);
      Im = Index(ivar, i - 1, j, k, nvar, n1, n2, n3);
      b[i] = rhs[Ic];
      for (m = 0; m < ncols[Ic]; m++)
      {
        col = cols[Ic][m];
        if (col != Ip && col != Ic && col != Im)
          b[i] -= JFD[Ic][m] * dv[col];
      }
    }
    /* Forward substitution [L][y] = [b] */
    y[0] = b[0];
    for (i = 1; i < n1; i++)
      y[i] = b[i] - l[i - 1] * y[i - 1];
    /* Backward substitution [U][x] = [y] */
    x[n1 - 1] = y[n1 - 1] * r[n1 - 1];
    for (i = n1 - 2; i >= 0; i--)
      x[i] = (y[i] - u[i] * x[i + 1]) * r[i];
    for (i = 0; i < n1; i++)
    {
      Ic = Index(ivar, i, j, k, nvar, n1, n2, n3);
      dv[Ic] = x[i];
    }
  }
}
/* -------------------------------------------------------------------------*/
// --------------------------------------------------------------------------*/
// Calculates the value of v at an arbitrary position (x,y,z) if the spectral coefficients are know*/*/
/* --------------------------------------------------------------------------*/
/* Calculates the value of v at an arbitrary position (A,B,phi)*/
double TwoPunctures::Spec_IntPolABphiFast(parameters par, double *v, int ivar, double A, double B, double phi)
{
  int i, j, k, N;
  double *p, *values1, **values2, result;

  int nvar = par.nvar;
  int n1 = par.n1;
  int n2 = par.n2;
  int n3 = par.n3;
  N = maximum3(n1, n2, n3);

  p = dvector(0, N);
  values1 = dvector(0, N);
  values2 = dmatrix(0, N, 0, N);

  for (k = 0; k < n3; k++)
  {
    for (j = 0; j < n2; j++)
    {
      for (i = 0; i < n1; i++)
        p[i] = v[ivar + nvar * (i + n1 * (j + n2 * k))];
      //      chebft_Zeros (p, n1, 0);
      values2[j][k] = chebev(-1, 1, p, n1, A);
    }
  }

  for (k = 0; k < n3; k++)
  {
    for (j = 0; j < n2; j++)
      p[j] = values2[j][k];
    //    chebft_Zeros (p, n2, 0);
    values1[k] = chebev(-1, 1, p, n2, B);
  }

  //  fourft (values1, n3, 0);
  result = fourev(values1, n3, phi);

  free_dvector(p, 0, N);
  free_dvector(values1, 0, N);
  free_dmatrix(values2, 0, N, 0, N);

  return result;
  //  */
  //  return 0.;
}

/* Calculates the value of v at an arbitrary position (x,y,z) given the spectral coefficients*/
double TwoPunctures::Spec_IntPolFast(parameters par, int ivar, double *v, double x, double y, double z)
{
  double xs, ys, zs, rs2, phi, X, R, A, B, aux1, aux2, result, Ui;

  int nvar = par.nvar;
  int n1 = par.n1;
  int n2 = par.n2;
  int n3 = par.n3;
  double par_b = par.b;

  xs = x / par.b;
  ys = y / par.b;
  zs = z / par.b;
  rs2 = ys * ys + zs * zs;
  phi = atan2(z, y);
  if (phi < 0)
    phi += 2. * Pi;

  aux1 = 0.5 * (xs * xs + rs2 - 1.);
  aux2 = sqrt(aux1 * aux1 + rs2);

  // Note from YT: aux2-aux1 can be equal to 1. When that happens, numerical
  // truncation may make it slightly larger than 1. This makes
  // R NAN! I also worry that aux2-aux1 and aux1+axu2 may become negative due to
  // truncation error, which gives rise to NAN for X and R.
  // The following few lines attempt to fix these.
  double aux2_plus_aux1, aux2_minus_aux1;
  if (aux1 < 0)
  {
    aux2_plus_aux1 = rs2 / (aux2 - aux1);
    aux2_minus_aux1 = aux2 - aux1;
  }
  else
  {
    aux2_plus_aux1 = aux2 + aux1;
    aux2_minus_aux1 = rs2 / (aux2 + aux1);
  }
  if (fabs(aux1) + fabs(aux2) < 1.e-20)
  {
    aux2_plus_aux1 = 0.0;
    aux2_minus_aux1 = 0.0;
  }
  double sqrt_aux2_minus_aux1 = sqrt(fabs(aux2_minus_aux1));

  // Note from YT: The following two lines have replaced by the 6 lines belows.
  //      X = asinhd(sqrt(aux1+aux2));
  //      R = asin(sqrt(fabs(-aux1+aux2)));

  X = asinh(sqrt(aux2_plus_aux1));
  if (sqrt_aux2_minus_aux1 > 1.0)
  {
    R = 0.5 * Pi;
  }
  else
  {
    R = asin(sqrt_aux2_minus_aux1);
  }

  if (x < 0)
    R = Pi - R;

  A = 2. * tanh(0.5 * X) - 1.;
  B = tan(0.5 * R - Piq);

  result = Spec_IntPolABphiFast(par, v, ivar, A, B, phi);

  Ui = (A - 1) * result;

  return Ui;
}

// Evaluates the spectral expansion coefficients of v
void TwoPunctures::SpecCoef(parameters par, int ivar, double *v, double *cf)
{
  // Here v is a pointer to the values of the variable v at the collocation points
  int i, j, k, N, n, l, m;
  double *p, ***values3, ***values4;

  int nvar = par.nvar;
  int n1 = par.n1;
  int n2 = par.n2;
  int n3 = par.n3;

  N = maximum3(n1, n2, n3);
  p = dvector(0, N);
  values3 = d3tensor(0, n1, 0, n2, 0, n3);
  values4 = d3tensor(0, n1, 0, n2, 0, n3);

  // Caclulate values3[n,j,k] = a_n^{j,k} = (sum_i^(n1-1) f(A_i,B_j,phi_k) Tn(-A_i))/k_n , k_n = N/2 or N
  for (k = 0; k < n3; k++)
  {
    for (j = 0; j < n2; j++)
    {
      for (i = 0; i < n1; i++)
        p[i] = v[ivar + (i + n1 * (j + n2 * k))];
      chebft_Zeros(p, n1, 0);
      for (n = 0; n < n1; n++)
      {
        values3[n][j][k] = p[n];
      }
    }
  }

  // Caclulate values4[n,l,k] = a_{n,l}^{k} = (sum_j^(n2-1) a_n^{j,k} Tn(B_j))/k_l , k_l = N/2 or N

  for (n = 0; n < n1; n++)
  {
    for (k = 0; k < n3; k++)
    {
      for (j = 0; j < n2; j++)
        p[j] = values3[n][j][k];
      chebft_Zeros(p, n2, 0);
      for (l = 0; l < n2; l++)
      {
        values4[n][l][k] = p[l];
      }
    }
  }

  // Caclulate coefficients  a_{n,l,m} = (sum_k^(n3-1) a_{n,m}^{k} fourier(phi_k))/k_m , k_m = N/2 or N
  for (i = 0; i < n1; i++)
  {
    for (j = 0; j < n2; j++)
    {
      for (k = 0; k < n3; k++)
        p[k] = values4[i][j][k];
      fourft(p, n3, 0);
      for (k = 0; k < n3; k++)
      {
        cf[ivar + (i + n1 * (j + n2 * k))] = p[k];
      }
    }
  }

  free_dvector(p, 0, N);
  free_d3tensor(values3, 0, n1, 0, n2, 0, n3);
  free_d3tensor(values4, 0, n1, 0, n2, 0, n3);
}
