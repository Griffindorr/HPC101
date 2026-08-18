
! Compute advection terms in right hand sides of field equations

#include "macrodef.fh"

! we need only distinguish different finite difference order
! Vertex or Cell is distinguished in routine symmetry_bd which locates in
! file "fmisc.f90"

! fourth order code

!-----------------------------------------------------------------------------
!
! Compute advection terms in right hand sides of field equations
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
!
! where
!
!        i
!      |B |
! v = -----
!        i
!       B
!
!-----------------------------------------------------------------------------

subroutine lopsided(ex,X,Y,Z,f,f_rhs,Sfx,Sfy,Sfz,Symmetry,SoA)
  implicit none

!~~~~~~> Input parameters:

  integer, intent(in)  :: ex(1:3),Symmetry
  real*8,  intent(in)  :: X(1:ex(1)),Y(1:ex(2)),Z(1:ex(3))
  real*8,dimension(ex(1),ex(2),ex(3)),intent(in)   :: f,Sfx,Sfy,Sfz

  real*8,dimension(ex(1),ex(2),ex(3)),intent(inout):: f_rhs
  real*8,dimension(3),intent(in) ::SoA

!~~~~~~> local variables:
! note index -2,-1,0, so we have 3 extra points
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8 :: dX,dY,dZ
  real*8 :: d12dx,d12dy,d12dz,d2dx,d2dy,d2dz
  real*8,  parameter :: ZEO=0.d0,ONE=1.d0, F3=3.d0
  real*8,  parameter :: TWO=2.d0,F6=6.0d0,F18=1.8d1
  real*8,  parameter :: F12=1.2d1, F10=1.d1,EIT=8.d0
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)

  d12dx = ONE/F12/dX
  d12dy = ONE/F12/dY
  d12dz = ONE/F12/dZ

  d2dx = ONE/TWO/dX
  d2dy = ONE/TWO/dY
  d2dz = ONE/TWO/dZ

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -2
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -2
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -2

! upper bound set ex-1 only for efficiency, 
! the loop body will set ex 0 also
#if 0
  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
!! old code
! x direction   
    if(Sfx(i,j,k) >= ZEO .and. i+3 <= imax .and. i-1 >= imin)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*fh(i-1,j,k)-F10*fh(i,j,k)+F18*fh(i+1,j,k) &
                                    -F6*fh(i+2,j,k)+    fh(i+3,j,k))

    elseif(Sfx(i,j,k) <= ZEO .and. i-3 >= imin .and. i+1 <= imax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*fh(i+1,j,k)-F10*fh(i,j,k)+F18*fh(i-1,j,k) &
                                    -F6*fh(i-2,j,k)+    fh(i-3,j,k))

     elseif(i+2 <= imax .and. i-2 >= imin)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfx(i,j,k)*d12dx*(fh(i-2,j,k)-EIT*fh(i-1,j,k)+EIT*fh(i+1,j,k)-fh(i+2,j,k))

     elseif(i+1 <= imax .and. i-1 >= imin)then
!
!              - f(i-1) + f(i+1)
!  fx(i) = --------------------------------
!                     2 dx
     f_rhs(i,j,k)=f_rhs(i,j,k) + Sfx(i,j,k)*d2dx*(-fh(i-1,j,k)+fh(i+1,j,k))

! set imax and imin 0
    endif

! y direction   
    if(Sfy(i,j,k) >= ZEO .and. j+3 <= jmax .and. j-1 >= jmin)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*fh(i,j-1,k)-F10*fh(i,j,k)+F18*fh(i,j+1,k) &
                                    -F6*fh(i,j+2,k)+    fh(i,j+3,k))

    elseif(Sfy(i,j,k) <= ZEO .and. j-3 >= jmin .and. j+1 <= jmax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*fh(i,j+1,k)-F10*fh(i,j,k)+F18*fh(i,j-1,k) &
                                    -F6*fh(i,j-2,k)+    fh(i,j-3,k))

     elseif(j+2 <= jmax .and. j-2 >= jmin)then

     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                            &
                  Sfy(i,j,k)*d12dy*(fh(i,j-2,k)-EIT*fh(i,j-1,k)+EIT*fh(i,j+1,k)-fh(i,j+2,k))

     elseif(j+1 <= jmax .and. j-1 >= jmin)then

     f_rhs(i,j,k)=f_rhs(i,j,k) + Sfy(i,j,k)*d2dy*(-fh(i,j-1,k)+fh(i,j+1,k))
! set jmin and jmax 0
     endif
!! z direction   
    if(Sfz(i,j,k) >= ZEO .and. k+3 <= kmax .and. k-1 >= kmin)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*fh(i,j,k-1)-F10*fh(i,j,k)+F18*fh(i,j,k+1) &
                                    -F6*fh(i,j,k+2)+    fh(i,j,k+3))

    elseif(Sfz(i,j,k) <= ZEO .and. k-3 >= kmin .and. k+1 <= kmax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*fh(i,j,k+1)-F10*fh(i,j,k)+F18*fh(i,j,k-1) &
                                    -F6*fh(i,j,k-2)+    fh(i,j,k-3))

     elseif(k+2 <= kmax .and. k-2 >= kmin)then

     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                            &
                  Sfz(i,j,k)*d12dz*(fh(i,j,k-2)-EIT*fh(i,j,k-1)+EIT*fh(i,j,k+1)-fh(i,j,k+2))

     elseif(k+1 <= kmax .and. k-1 >= kmin)then

     f_rhs(i,j,k)=f_rhs(i,j,k)+Sfz(i,j,k)*d2dz*(-fh(i,j,k-1)+fh(i,j,k+1))
! set kmin and kmax 0
     endif
#else
!! new code, 2012dec27, based on bam
! fused version: no fh copy; interior gathers directly from f, the thin
! boundary band uses the symmetry-mirrored gather gf()
! upper bound set ex-1 only for efficiency, 
! the loop body will set ex 0 also
  do k=4,ex(3)-3
  do j=4,ex(2)-3
  do i=4,ex(1)-3
! x direction (4th-order upwind; boundary conditions always hold in interior)
    if(Sfx(i,j,k) > ZEO)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*f(i-1,j,k)-F10*f(i,j,k)+F18*f(i+1,j,k) &
                                    -F6*f(i+2,j,k)+    f(i+3,j,k))
    elseif(Sfx(i,j,k) < ZEO)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*f(i+1,j,k)-F10*f(i,j,k)+F18*f(i-1,j,k) &
                                    -F6*f(i-2,j,k)+    f(i-3,j,k))
    endif
! y direction
    if(Sfy(i,j,k) > ZEO)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*f(i,j-1,k)-F10*f(i,j,k)+F18*f(i,j+1,k) &
                                    -F6*f(i,j+2,k)+    f(i,j+3,k))
    elseif(Sfy(i,j,k) < ZEO)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*f(i,j+1,k)-F10*f(i,j,k)+F18*f(i,j-1,k) &
                                    -F6*f(i,j-2,k)+    f(i,j-3,k))
    endif
! z direction
    if(Sfz(i,j,k) > ZEO)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*f(i,j,k-1)-F10*f(i,j,k)+F18*f(i,j,k+1) &
                                    -F6*f(i,j,k+2)+    f(i,j,k+3))
    elseif(Sfz(i,j,k) < ZEO)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*f(i,j,k+1)-F10*f(i,j,k)+F18*f(i,j,k-1) &
                                    -F6*f(i,j,k-2)+    f(i,j,k-3))
    endif
  enddo
  enddo
  enddo
! boundary band: full original logic with mirrored gather
  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
    if(i.ge.4 .and. i.le.ex(1)-3 .and. j.ge.4 .and. j.le.ex(2)-3 .and. &
       k.ge.4 .and. k.le.ex(3)-3) cycle
! x direction   
    if(Sfx(i,j,k) > ZEO)then
      if(i+3 <= imax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*gf(i-1,j,k)-F10*gf(i,j,k)+F18*gf(i+1,j,k) &
                                    -F6*gf(i+2,j,k)+    gf(i+3,j,k))
     elseif(i+2 <= imax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfx(i,j,k)*d12dx*(gf(i-2,j,k)-EIT*gf(i-1,j,k)+EIT*gf(i+1,j,k)-gf(i+2,j,k))
     elseif(i+1 <= imax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*gf(i+1,j,k)-F10*gf(i,j,k)+F18*gf(i-1,j,k) &
                                    -F6*gf(i-2,j,k)+    gf(i-3,j,k))
     endif
   elseif(Sfx(i,j,k) < ZEO)then
      if(i-3 >= imin)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*gf(i+1,j,k)-F10*gf(i,j,k)+F18*gf(i-1,j,k) &
                                    -F6*gf(i-2,j,k)+    gf(i-3,j,k))
     elseif(i-2 >= imin)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfx(i,j,k)*d12dx*(gf(i-2,j,k)-EIT*gf(i-1,j,k)+EIT*gf(i+1,j,k)-gf(i+2,j,k))
     elseif(i-1 >= imin)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*gf(i-1,j,k)-F10*gf(i,j,k)+F18*gf(i+1,j,k) &
                                    -F6*gf(i+2,j,k)+    gf(i+3,j,k))
     endif
   endif
! y direction   
    if(Sfy(i,j,k) > ZEO)then
      if(j+3 <= jmax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*gf(i,j-1,k)-F10*gf(i,j,k)+F18*gf(i,j+1,k) &
                                    -F6*gf(i,j+2,k)+    gf(i,j+3,k))
     elseif(j+2 <= jmax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfy(i,j,k)*d12dy*(gf(i,j-2,k)-EIT*gf(i,j-1,k)+EIT*gf(i,j+1,k)-gf(i,j+2,k))
     elseif(j+1 <= jmax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*gf(i,j+1,k)-F10*gf(i,j,k)+F18*gf(i,j-1,k) &
                                    -F6*gf(i,j-2,k)+    gf(i,j-3,k))
     endif
   elseif(Sfy(i,j,k) < ZEO)then
      if(j-3 >= jmin)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*gf(i,j+1,k)-F10*gf(i,j,k)+F18*gf(i,j-1,k) &
                                    -F6*gf(i,j-2,k)+    gf(i,j-3,k))
     elseif(j-2 >= jmin)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfy(i,j,k)*d12dy*(gf(i,j-2,k)-EIT*gf(i,j-1,k)+EIT*gf(i,j+1,k)-gf(i,j+2,k))
     elseif(j-1 >= jmin)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*gf(i,j-1,k)-F10*gf(i,j,k)+F18*gf(i,j+1,k) &
                                    -F6*gf(i,j+2,k)+    gf(i,j+3,k))
     endif
   endif
! z direction   
    if(Sfz(i,j,k) > ZEO)then
      if(k+3 <= kmax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*gf(i,j,k-1)-F10*gf(i,j,k)+F18*gf(i,j,k+1) &
                                    -F6*gf(i,j,k+2)+    gf(i,j,k+3))
     elseif(k+2 <= kmax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfz(i,j,k)*d12dz*(gf(i,j,k-2)-EIT*gf(i,j,k-1)+EIT*gf(i,j,k+1)-gf(i,j,k+2))
     elseif(k+1 <= kmax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*gf(i,j,k+1)-F10*gf(i,j,k)+F18*gf(i,j,k-1) &
                                    -F6*gf(i,j,k-2)+    gf(i,j,k-3))
     endif
   elseif(Sfz(i,j,k) < ZEO)then
      if(k-3 >= kmin)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*gf(i,j,k+1)-F10*gf(i,j,k)+F18*gf(i,j,k-1) &
                                    -F6*gf(i,j,k-2)+    gf(i,j,k-3))
     elseif(k-2 >= kmin)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfz(i,j,k)*d12dz*(gf(i,j,k-2)-EIT*gf(i,j,k-1)+EIT*gf(i,j,k+1)-gf(i,j,k+2))
     elseif(k-1 >= kmin)then
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*gf(i,j,k-1)-F10*gf(i,j,k)+F18*gf(i,j,k+1) &
                                    -F6*gf(i,j,k+2)+    gf(i,j,k+3))
     endif
   endif
  enddo
  enddo
  enddo
#endif
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

  end subroutine lopsided

