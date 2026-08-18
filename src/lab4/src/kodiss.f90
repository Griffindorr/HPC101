

#include "macrodef.fh"

! we need only distinguish different finite difference order
! Vertex or Cell is distinguished in routine symmetry_bd which locates in
! file "fmisc.f90"

! fourth order code

!---------------------------------------------------------------------------------------------
!usual type Kreiss-Oliger type numerical dissipation
!We support cell center only
! Note the notation D_+ and D_- [P240 of B. Gustafsson, H.-O. Kreiss, and J. Oliger, Time
! Dependent Problems and Difference Methods (Wiley, New York, 1995).]
! D_+ = (f(i+1) - f(i))/h
! D_- = (f(i) - f(i-1))/h
! then we have D_+D_- = D_-D_+
!              D_+^3D_-^3 = (D_+D_-)^3 =
!    f(i-3) - 6 f(i-2) + 15 f(i-1) - 20 f(i) + 15 f(i+1) - 6 f(i+2) + f(i+3)
! -----------------------------------------------------------------------------
!                                    dx^6
! this is for 4th order accurate finite difference scheme
!---------------------------------------------------------------------------------------------
subroutine kodis(ex,X,Y,Z,f,f_rhs,SoA,Symmetry,eps)

implicit none
! argument variables
integer,intent(in) :: Symmetry
integer,dimension(3),intent(in)::ex
real*8, dimension(1:3), intent(in) :: SoA
double precision,intent(in),dimension(ex(1))::X
double precision,intent(in),dimension(ex(2))::Y
double precision,intent(in),dimension(ex(3))::Z
double precision,intent(in),dimension(ex(1),ex(2),ex(3))::f
double precision,intent(inout),dimension(ex(1),ex(2),ex(3))::f_rhs
real*8,intent(in) :: eps
! local variables
integer :: imin,jmin,kmin,imax,jmax,kmax
integer :: i,j,k
real*8  :: dX,dY,dZ
real*8, parameter :: ONE=1.d0,SIX=6.d0,FIT=1.5d1,TWT=2.d1
real*8,parameter::cof=6.4d1   ! 2^6
integer, parameter :: NO_SYMM=0, OCTANT=2
!rhs_i = rhs_i + eps/dx/cof*(f_i-3 - 6*f_i-2 + 15*f_i-1 - 20*f_i + 15*f_i+1 - 6*f_i+2 + f_i+3)
  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)
  
  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1

  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -2
  if(Symmetry == OCTANT .and. dabs(X(1)) < dX) imin = -2
  if(Symmetry == OCTANT .and. dabs(Y(1)) < dY) jmin = -2

! fused version: no fh copy; interior points gather directly from f,
! the thin boundary band uses the symmetry-mirrored gather gf()
  do k=4,ex(3)-3
  do j=4,ex(2)-3
  do i=4,ex(1)-3
   f_rhs(i,j,k)       = f_rhs(i,j,k) + eps/cof *( (     &
                              (f(i-3,j,k)+f(i+3,j,k)) - &
                          SIX*(f(i-2,j,k)+f(i+2,j,k)) + &
                          FIT*(f(i-1,j,k)+f(i+1,j,k)) - &
                          TWT* f(i,j,k)            )/dX + &
                                                  (     &
                              (f(i,j-3,k)+f(i,j+3,k)) - &
                          SIX*(f(i,j-2,k)+f(i,j+2,k)) + &
                          FIT*(f(i,j-1,k)+f(i,j+1,k)) - &
                          TWT* f(i,j,k)            )/dY + &
                                                  (     &
                              (f(i,j,k-3)+f(i,j,k+3)) - &
                          SIX*(f(i,j,k-2)+f(i,j,k+2)) + &
                          FIT*(f(i,j,k-1)+f(i,j,k+1)) - &
                          TWT* f(i,j,k)            )/dZ )
  enddo
  enddo
  enddo

  do k=1,ex(3)
  do j=1,ex(2)
  do i=1,ex(1)
  if(i.ge.4 .and. i.le.ex(1)-3 .and. j.ge.4 .and. j.le.ex(2)-3 .and. &
     k.ge.4 .and. k.le.ex(3)-3) cycle
  if(i-3 >= imin .and. i+3 <= imax .and. &
     j-3 >= jmin .and. j+3 <= jmax .and. &
     k-3 >= kmin .and. k+3 <= kmax) then
   f_rhs(i,j,k)       = f_rhs(i,j,k) + eps/cof *( (     &
                              (gf(i-3,j,k)+gf(i+3,j,k)) - &
                          SIX*(gf(i-2,j,k)+gf(i+2,j,k)) + &
                          FIT*(gf(i-1,j,k)+gf(i+1,j,k)) - &
                          TWT* gf(i,j,k)            )/dX + &
                                                  (     &
                              (gf(i,j-3,k)+gf(i,j+3,k)) - &
                          SIX*(gf(i,j-2,k)+gf(i,j+2,k)) + &
                          FIT*(gf(i,j-1,k)+gf(i,j+1,k)) - &
                          TWT* gf(i,j,k)            )/dY + &
                                                  (     &
                              (gf(i,j,k-3)+gf(i,j,k+3)) - &
                          SIX*(gf(i,j,k-2)+gf(i,j,k+2)) + &
                          FIT*(gf(i,j,k-1)+gf(i,j,k+1)) - &
                          TWT* gf(i,j,k)            )/dZ )
  endif
  enddo
  enddo
  enddo
  return

  contains

! symmetry-mirrored gather: index ii<1 reads f(1-ii,...)*SoA
! (identical to the ghost values produced by symmetry_bd)
    function gf(ii,jj,kk) result(v)
      integer, intent(in) :: ii,jj,kk
      real*8 :: v
      integer :: i2,j2,k2
      real*8 :: sv
      i2 = ii
      j2 = jj
      k2 = kk
      sv = 1.d0
      if(ii .lt. 1) then
         i2 = 1 - ii
         sv = sv * SoA(1)
      endif
      if(jj .lt. 1) then
         j2 = 1 - jj
         sv = sv * SoA(2)
      endif
      if(kk .lt. 1) then
         k2 = 1 - kk
         sv = sv * SoA(3)
      endif
      v = f(i2,j2,k2) * sv
    end function gf

  end subroutine kodis