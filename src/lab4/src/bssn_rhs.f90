

#include "macrodef.fh"

  function compute_rhs_bssn(ex, T,X, Y, Z,                                     &
               chi    ,   trK    ,                                             &
               dxx    ,   gxy    ,   gxz    ,   dyy    ,   gyz    ,   dzz,     &
               Axx    ,   Axy    ,   Axz    ,   Ayy    ,   Ayz    ,   Azz,     &
               Gamx   ,  Gamy    ,  Gamz    ,                                  &
               Lap    ,  betax   ,  betay   ,  betaz   ,                       &
               dtSfx  ,  dtSfy   ,  dtSfz   ,                                  &
               chi_rhs,   trK_rhs,                                             &
               gxx_rhs,   gxy_rhs,   gxz_rhs,   gyy_rhs,   gyz_rhs,   gzz_rhs, &
               Axx_rhs,   Axy_rhs,   Axz_rhs,   Ayy_rhs,   Ayz_rhs,   Azz_rhs, &
               Gamx_rhs,  Gamy_rhs,  Gamz_rhs,                                 &
               Lap_rhs,  betax_rhs,  betay_rhs,  betaz_rhs,                    &
               dtSfx_rhs,  dtSfy_rhs,  dtSfz_rhs,                              &
               rho,Sx,Sy,Sz,Sxx,Sxy,Sxz,Syy,Syz,Szz,                           &
               Gamxxx,Gamxxy,Gamxxz,Gamxyy,Gamxyz,Gamxzz,                      &
               Gamyxx,Gamyxy,Gamyxz,Gamyyy,Gamyyz,Gamyzz,                      &
               Gamzxx,Gamzxy,Gamzxz,Gamzyy,Gamzyz,Gamzzz,                      &
               Rxx,Rxy,Rxz,Ryy,Ryz,Rzz,                                        &
               ham_Res, movx_Res, movy_Res, movz_Res,                          &
                        Gmx_Res, Gmy_Res, Gmz_Res,                             &
               Symmetry,Lev,eps,co)  result(gont)
! calculate constraint violation when co=0               
  implicit none

!~~~~~~> Input parameters:

  integer,intent(in ):: ex(1:3), Symmetry,Lev,co
  real*8, intent(in ):: T
  real*8, intent(in ):: X(1:ex(1)),Y(1:ex(2)),Z(1:ex(3))
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) :: chi,dxx,dyy,dzz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: trK
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: gxy,gxz,gyz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: Gamx,Gamy,Gamz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) :: Lap, betax, betay, betaz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: dtSfx,  dtSfy,  dtSfz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: chi_rhs,trK_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: gxx_rhs,gxy_rhs,gxz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: gyy_rhs,gyz_rhs,gzz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Axx_rhs,Axy_rhs,Axz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Ayy_rhs,Ayz_rhs,Azz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamx_rhs,Gamy_rhs,Gamz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Lap_rhs, betax_rhs, betay_rhs, betaz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: dtSfx_rhs,dtSfy_rhs,dtSfz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: rho,Sx,Sy,Sz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: Sxx,Sxy,Sxz,Syy,Syz,Szz
! when out, physical second kind of connection  
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamxxx, Gamxxy, Gamxxz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamxyy, Gamxyz, Gamxzz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamyxx, Gamyxy, Gamyxz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamyyy, Gamyyz, Gamyzz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamzxx, Gamzxy, Gamzxz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamzyy, Gamzyz, Gamzzz
! when out, physical Ricci tensor  
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Rxx,Rxy,Rxz,Ryy,Ryz,Rzz
  real*8,intent(in) :: eps
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) :: ham_Res, movx_Res, movy_Res, movz_Res
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) :: Gmx_Res, Gmy_Res, Gmz_Res
!  gont = 0: success; gont = 1: something wrong
  integer::gont

!~~~~~~> Other variables:

  real*8, dimension(ex(1),ex(2),ex(3)) :: gxx,gyy,gzz
  real*8, dimension(ex(1),ex(2),ex(3)) :: chix,chiy,chiz
  real*8, dimension(ex(1),ex(2),ex(3)) :: gxxx,gxyx,gxzx,gyyx,gyzx,gzzx
  real*8, dimension(ex(1),ex(2),ex(3)) :: gxxy,gxyy,gxzy,gyyy,gyzy,gzzy
  real*8, dimension(ex(1),ex(2),ex(3)) :: gxxz,gxyz,gxzz,gyyz,gyzz,gzzz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Lapx,Lapy,Lapz
  real*8, dimension(ex(1),ex(2),ex(3)) :: betaxx,betaxy,betaxz
  real*8, dimension(ex(1),ex(2),ex(3)) :: betayx,betayy,betayz
  real*8, dimension(ex(1),ex(2),ex(3)) :: betazx,betazy,betazz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Gamxx,Gamxy,Gamxz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Gamyx,Gamyy,Gamyz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Gamzx,Gamzy,Gamzz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Kx,Ky,Kz,div_beta,S
  real*8, dimension(ex(1),ex(2),ex(3)) :: f,fxx,fxy,fxz,fyy,fyz,fzz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Gamxa,Gamya,Gamza,alpn1,chin1
  real*8, dimension(ex(1),ex(2),ex(3)) :: gupxx,gupxy,gupxz
  real*8, dimension(ex(1),ex(2),ex(3)) :: gupyy,gupyz,gupzz

  real*8,dimension(3) ::SSS,AAS,ASA,SAA,ASS,SAS,SSA
  real*8            :: dX, dY, dZ, PI
  real*8            :: al, ch1, gx, gy, gz, gxy1, gxz1, gyz1, detg
  integer           :: i, j, k
  real*8, parameter :: ZEO = 0.d0,ONE = 1.D0, TWO = 2.D0, FOUR = 4.D0
  real*8, parameter :: EIGHT = 8.D0, HALF = 0.5D0, THR = 3.d0
  real*8, parameter :: SYM = 1.D0, ANTI= - 1.D0
  double precision,parameter::FF = 0.75d0,eta=2.d0
  real*8, parameter :: F1o3 = 1.D0/3.D0, F2o3 = 2.D0/3.D0,F3o2=1.5d0, F1o6 = 1.D0/6.D0
  real*8, parameter :: F16=1.6d1,F8=8.d0


!!! sanity check
  dX = sum(chi)+sum(trK)+sum(dxx)+sum(gxy)+sum(gxz)+sum(dyy)+sum(gyz)+sum(dzz) &
      +sum(Axx)+sum(Axy)+sum(Axz)+sum(Ayy)+sum(Ayz)+sum(Azz)                   &
      +sum(Gamx)+sum(Gamy)+sum(Gamz)                                           &
      +sum(Lap)+sum(betax)+sum(betay)+sum(betaz)
  if(dX.ne.dX) then
     if(sum(chi).ne.sum(chi))write(*,*)"bssn.f90: find NaN in chi"
     if(sum(trK).ne.sum(trK))write(*,*)"bssn.f90: find NaN in trk"
     if(sum(dxx).ne.sum(dxx))write(*,*)"bssn.f90: find NaN in dxx"
     if(sum(gxy).ne.sum(gxy))write(*,*)"bssn.f90: find NaN in gxy"
     if(sum(gxz).ne.sum(gxz))write(*,*)"bssn.f90: find NaN in gxz"
     if(sum(dyy).ne.sum(dyy))write(*,*)"bssn.f90: find NaN in dyy"
     if(sum(gyz).ne.sum(gyz))write(*,*)"bssn.f90: find NaN in gyz"
     if(sum(dzz).ne.sum(dzz))write(*,*)"bssn.f90: find NaN in dzz"
     if(sum(Axx).ne.sum(Axx))write(*,*)"bssn.f90: find NaN in Axx"
     if(sum(Axy).ne.sum(Axy))write(*,*)"bssn.f90: find NaN in Axy"
     if(sum(Axz).ne.sum(Axz))write(*,*)"bssn.f90: find NaN in Axz"
     if(sum(Ayy).ne.sum(Ayy))write(*,*)"bssn.f90: find NaN in Ayy"
     if(sum(Ayz).ne.sum(Ayz))write(*,*)"bssn.f90: find NaN in Ayz"
     if(sum(Azz).ne.sum(Azz))write(*,*)"bssn.f90: find NaN in Azz"
     if(sum(Gamx).ne.sum(Gamx))write(*,*)"bssn.f90: find NaN in Gamx"
     if(sum(Gamy).ne.sum(Gamy))write(*,*)"bssn.f90: find NaN in Gamy"
     if(sum(Gamz).ne.sum(Gamz))write(*,*)"bssn.f90: find NaN in Gamz"
     if(sum(Lap).ne.sum(Lap))write(*,*)"bssn.f90: find NaN in Lap"
     if(sum(betax).ne.sum(betax))write(*,*)"bssn.f90: find NaN in betax"
     if(sum(betay).ne.sum(betay))write(*,*)"bssn.f90: find NaN in betay"
     if(sum(betaz).ne.sum(betaz))write(*,*)"bssn.f90: find NaN in betaz"
     gont = 1
     return
  endif

  PI = dacos(-ONE)

  dX = X(2) - X(1)
  dY = Y(2) - Y(1)
  dZ = Z(2) - Z(1)

  call fderivs(ex,betax,betaxx,betaxy,betaxz,X,Y,Z,ANTI, SYM, SYM,Symmetry,Lev)
  call fderivs(ex,betay,betayx,betayy,betayz,X,Y,Z, SYM,ANTI, SYM,Symmetry,Lev)
  call fderivs(ex,betaz,betazx,betazy,betazz,X,Y,Z, SYM, SYM,ANTI,Symmetry,Lev)
 
  call fderivs(ex,chi,chix,chiy,chiz,X,Y,Z,SYM,SYM,SYM,symmetry,Lev)

  call fderivs(ex,dxx,gxxx,gxxy,gxxz,X,Y,Z,SYM ,SYM ,SYM ,Symmetry,Lev)
  call fderivs(ex,gxy,gxyx,gxyy,gxyz,X,Y,Z,ANTI,ANTI,SYM ,Symmetry,Lev)
  call fderivs(ex,gxz,gxzx,gxzy,gxzz,X,Y,Z,ANTI,SYM ,ANTI,Symmetry,Lev)
  call fderivs(ex,dyy,gyyx,gyyy,gyyz,X,Y,Z,SYM ,SYM ,SYM ,Symmetry,Lev)
  call fderivs(ex,gyz,gyzx,gyzy,gyzz,X,Y,Z,SYM ,ANTI,ANTI,Symmetry,Lev)
  call fderivs(ex,dzz,gzzx,gzzy,gzzz,X,Y,Z,SYM ,SYM ,SYM ,Symmetry,Lev)

  alpn1 = Lap + ONE
  chin1 = chi + ONE
  gxx = dxx + ONE
  gyy = dyy + ONE
  gzz = dzz + ONE

  call fderivs(ex,betax,betaxx,betaxy,betaxz,X,Y,Z,ANTI, SYM, SYM,Symmetry,Lev)
  call fderivs(ex,betay,betayx,betayy,betayz,X,Y,Z, SYM,ANTI, SYM,Symmetry,Lev)
  call fderivs(ex,betaz,betazx,betazy,betazz,X,Y,Z, SYM, SYM,ANTI,Symmetry,Lev)
  
  div_beta = betaxx + betayy + betazz
 
  call fderivs(ex,chi,chix,chiy,chiz,X,Y,Z,SYM,SYM,SYM,symmetry,Lev)

  chi_rhs = F2o3 *chin1*( alpn1 * trK - div_beta ) !rhs for chi

  call fderivs(ex,dxx,gxxx,gxxy,gxxz,X,Y,Z,SYM ,SYM ,SYM ,Symmetry,Lev)
  call fderivs(ex,gxy,gxyx,gxyy,gxyz,X,Y,Z,ANTI,ANTI,SYM ,Symmetry,Lev)
  call fderivs(ex,gxz,gxzx,gxzy,gxzz,X,Y,Z,ANTI,SYM ,ANTI,Symmetry,Lev)
  call fderivs(ex,dyy,gyyx,gyyy,gyyz,X,Y,Z,SYM ,SYM ,SYM ,Symmetry,Lev)
  call fderivs(ex,gyz,gyzx,gyzy,gyzz,X,Y,Z,SYM ,ANTI,ANTI,Symmetry,Lev)
  call fderivs(ex,dzz,gzzx,gzzy,gzzz,X,Y,Z,SYM ,SYM ,SYM ,Symmetry,Lev)

  gxx_rhs = - TWO * alpn1 * Axx    -  F2o3 * gxx * div_beta          + &
              TWO *(  gxx * betaxx +   gxy * betayx +   gxz * betazx)

  gyy_rhs = - TWO * alpn1 * Ayy    -  F2o3 * gyy * div_beta          + &
              TWO *(  gxy * betaxy +   gyy * betayy +   gyz * betazy)

  gzz_rhs = - TWO * alpn1 * Azz    -  F2o3 * gzz * div_beta          + &
              TWO *(  gxz * betaxz +   gyz * betayz +   gzz * betazz)

  gxy_rhs = - TWO * alpn1 * Axy    +  F1o3 * gxy    * div_beta       + &
                      gxx * betaxy                  +   gxz * betazy + &
                                       gyy * betayx +   gyz * betazx   &
                                                    -   gxy * betazz

  gyz_rhs = - TWO * alpn1 * Ayz    +  F1o3 * gyz    * div_beta       + &
                      gxy * betaxz +   gyy * betayz                  + &
                      gxz * betaxy                  +   gzz * betazy   &
                                                    -   gyz * betaxx
 
  gxz_rhs = - TWO * alpn1 * Axz    +  F1o3 * gxz    * div_beta       + &
                      gxx * betaxz +   gxy * betayz                  + &
                                       gyz * betayx +   gzz * betazx   &
                                                    -   gxz * betayy     !rhs for gij

! invert tilted metric
  gupzz =  gxx * gyy * gzz + gxy * gyz * gxz + gxz * gxy * gyz - &
           gxz * gyy * gxz - gxy * gxy * gzz - gxx * gyz * gyz
  gupxx =   ( gyy * gzz - gyz * gyz ) / gupzz
  gupxy = - ( gxy * gzz - gyz * gxz ) / gupzz
  gupxz =   ( gxy * gyz - gyy * gxz ) / gupzz
  gupyy =   ( gxx * gzz - gxz * gxz ) / gupzz
  gupyz = - ( gxx * gyz - gxy * gxz ) / gupzz
  gupzz =   ( gxx * gyy - gxy * gxy ) / gupzz


  if(co == 0)then
! Gam^i_Res = Gam^i + gup^ij_,j
  Gmx_Res = Gamx - (gupxx*(gupxx*gxxx+gupxy*gxyx+gupxz*gxzx)&
                   +gupxy*(gupxx*gxyx+gupxy*gyyx+gupxz*gyzx)&
                   +gupxz*(gupxx*gxzx+gupxy*gyzx+gupxz*gzzx)&
                   +gupxx*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy)&
                   +gupxy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy)&
                   +gupxz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy)&
                   +gupxx*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)&
                   +gupxy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)&
                   +gupxz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz))
  Gmy_Res = Gamy - (gupxx*(gupxy*gxxx+gupyy*gxyx+gupyz*gxzx)&
                   +gupxy*(gupxy*gxyx+gupyy*gyyx+gupyz*gyzx)&
                   +gupxz*(gupxy*gxzx+gupyy*gyzx+gupyz*gzzx)&
                   +gupxy*(gupxy*gxxy+gupyy*gxyy+gupyz*gxzy)&
                   +gupyy*(gupxy*gxyy+gupyy*gyyy+gupyz*gyzy)&
                   +gupyz*(gupxy*gxzy+gupyy*gyzy+gupyz*gzzy)&
                   +gupxy*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)&
                   +gupyy*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)&
                   +gupyz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz))
  Gmz_Res = Gamz - (gupxx*(gupxz*gxxx+gupyz*gxyx+gupzz*gxzx)&
                   +gupxy*(gupxz*gxyx+gupyz*gyyx+gupzz*gyzx)&
                   +gupxz*(gupxz*gxzx+gupyz*gyzx+gupzz*gzzx)&
                   +gupxy*(gupxz*gxxy+gupyz*gxyy+gupzz*gxzy)&
                   +gupyy*(gupxz*gxyy+gupyz*gyyy+gupzz*gyzy)&
                   +gupyz*(gupxz*gxzy+gupyz*gyzy+gupzz*gzzy)&
                   +gupxz*(gupxz*gxxz+gupyz*gxyz+gupzz*gxzz)&
                   +gupyz*(gupxz*gxyz+gupyz*gyyz+gupzz*gyzz)&
                   +gupzz*(gupxz*gxzz+gupyz*gyzz+gupzz*gzzz))
  endif

! second kind of connection
  Gamxxx =HALF*( gupxx*gxxx + gupxy*(TWO*gxyx - gxxy ) + gupxz*(TWO*gxzx - gxxz ))
  Gamyxx =HALF*( gupxy*gxxx + gupyy*(TWO*gxyx - gxxy ) + gupyz*(TWO*gxzx - gxxz ))
  Gamzxx =HALF*( gupxz*gxxx + gupyz*(TWO*gxyx - gxxy ) + gupzz*(TWO*gxzx - gxxz ))
 
  Gamxyy =HALF*( gupxx*(TWO*gxyy - gyyx ) + gupxy*gyyy + gupxz*(TWO*gyzy - gyyz ))
  Gamyyy =HALF*( gupxy*(TWO*gxyy - gyyx ) + gupyy*gyyy + gupyz*(TWO*gyzy - gyyz ))
  Gamzyy =HALF*( gupxz*(TWO*gxyy - gyyx ) + gupyz*gyyy + gupzz*(TWO*gyzy - gyyz ))

  Gamxzz =HALF*( gupxx*(TWO*gxzz - gzzx ) + gupxy*(TWO*gyzz - gzzy ) + gupxz*gzzz)
  Gamyzz =HALF*( gupxy*(TWO*gxzz - gzzx ) + gupyy*(TWO*gyzz - gzzy ) + gupyz*gzzz)
  Gamzzz =HALF*( gupxz*(TWO*gxzz - gzzx ) + gupyz*(TWO*gyzz - gzzy ) + gupzz*gzzz)

  Gamxxy =HALF*( gupxx*gxxy + gupxy*gyyx + gupxz*( gxzy + gyzx - gxyz ) )
  Gamyxy =HALF*( gupxy*gxxy + gupyy*gyyx + gupyz*( gxzy + gyzx - gxyz ) )
  Gamzxy =HALF*( gupxz*gxxy + gupyz*gyyx + gupzz*( gxzy + gyzx - gxyz ) )

  Gamxxz =HALF*( gupxx*gxxz + gupxy*( gxyz + gyzx - gxzy ) + gupxz*gzzx )
  Gamyxz =HALF*( gupxy*gxxz + gupyy*( gxyz + gyzx - gxzy ) + gupyz*gzzx )
  Gamzxz =HALF*( gupxz*gxxz + gupyz*( gxyz + gyzx - gxzy ) + gupzz*gzzx )

  Gamxyz =HALF*( gupxx*( gxyz + gxzy - gyzx ) + gupxy*gyyz + gupxz*gzzy )
  Gamyyz =HALF*( gupxy*( gxyz + gxzy - gyzx ) + gupyy*gyyz + gupyz*gzzy )
  Gamzyz =HALF*( gupxz*( gxyz + gxzy - gyzx ) + gupyz*gyyz + gupzz*gzzy )
! Raise indices of \tilde A_{ij} and store in R_ij

  Rxx =    gupxx * gupxx * Axx + gupxy * gupxy * Ayy + gupxz * gupxz * Azz + &
      TWO*(gupxx * gupxy * Axy + gupxx * gupxz * Axz + gupxy * gupxz * Ayz)

  Ryy =    gupxy * gupxy * Axx + gupyy * gupyy * Ayy + gupyz * gupyz * Azz + &
      TWO*(gupxy * gupyy * Axy + gupxy * gupyz * Axz + gupyy * gupyz * Ayz)

  Rzz =    gupxz * gupxz * Axx + gupyz * gupyz * Ayy + gupzz * gupzz * Azz + &
      TWO*(gupxz * gupyz * Axy + gupxz * gupzz * Axz + gupyz * gupzz * Ayz)

  Rxy =    gupxx * gupxy * Axx + gupxy * gupyy * Ayy + gupxz * gupyz * Azz + &
          (gupxx * gupyy       + gupxy * gupxy)* Axy                       + &
          (gupxx * gupyz       + gupxz * gupxy)* Axz                       + &
          (gupxy * gupyz       + gupxz * gupyy)* Ayz

  Rxz =    gupxx * gupxz * Axx + gupxy * gupyz * Ayy + gupxz * gupzz * Azz + &
          (gupxx * gupyz       + gupxy * gupxz)* Axy                       + &
          (gupxx * gupzz       + gupxz * gupxz)* Axz                       + &
          (gupxy * gupzz       + gupxz * gupyz)* Ayz

  Ryz =    gupxy * gupxz * Axx + gupyy * gupyz * Ayy + gupyz * gupzz * Azz + &
          (gupxy * gupyz       + gupyy * gupxz)* Axy                       + &
          (gupxy * gupzz       + gupyz * gupxz)* Axz                       + &
          (gupyy * gupzz       + gupyz * gupyz)* Ayz

! Right hand side for Gam^i without shift terms...
  call fderivs(ex,Lap,Lapx,Lapy,Lapz,X,Y,Z,SYM,SYM,SYM,Symmetry,Lev)
  call fderivs(ex,trK,Kx,Ky,Kz,X,Y,Z,SYM,SYM,SYM,symmetry,Lev)

  do k=1,ex(3)
  do j=1,ex(2)
  do i=1,ex(1)
   Gamx_rhs(i,j,k) = - TWO * (   Lapx(i,j,k) * Rxx(i,j,k) +   Lapy(i,j,k) * Rxy(i,j,k) +   Lapz(i,j,k) * Rxz(i,j,k) ) + &
        TWO * alpn1(i,j,k) * (                                                &
        -F3o2/chin1(i,j,k) * (   chix(i,j,k) * Rxx(i,j,k) +   chiy(i,j,k) * Rxy(i,j,k) +   chiz(i,j,k) * Rxz(i,j,k) ) - &
              gupxx(i,j,k) * (   F2o3 * Kx(i,j,k)  +  EIGHT * PI * Sx(i,j,k)            ) - &
              gupxy(i,j,k) * (   F2o3 * Ky(i,j,k)  +  EIGHT * PI * Sy(i,j,k)            ) - &
              gupxz(i,j,k) * (   F2o3 * Kz(i,j,k)  +  EIGHT * PI * Sz(i,j,k)            ) + &
                        Gamxxx(i,j,k) * Rxx(i,j,k) + Gamxyy(i,j,k) * Ryy(i,j,k) + Gamxzz(i,j,k) * Rzz(i,j,k)   + &
                TWO * ( Gamxxy(i,j,k) * Rxy(i,j,k) + Gamxxz(i,j,k) * Rxz(i,j,k) + Gamxyz(i,j,k) * Ryz(i,j,k) ) )

   Gamy_rhs(i,j,k) = - TWO * (   Lapx(i,j,k) * Rxy(i,j,k) +   Lapy(i,j,k) * Ryy(i,j,k) +   Lapz(i,j,k) * Ryz(i,j,k) ) + &
        TWO * alpn1(i,j,k) * (                                                &
        -F3o2/chin1(i,j,k) * (   chix(i,j,k) * Rxy(i,j,k) +  chiy(i,j,k) * Ryy(i,j,k) +    chiz(i,j,k) * Ryz(i,j,k) ) - &
              gupxy(i,j,k) * (   F2o3 * Kx(i,j,k)  +  EIGHT * PI * Sx(i,j,k)            ) - &
              gupyy(i,j,k) * (   F2o3 * Ky(i,j,k)  +  EIGHT * PI * Sy(i,j,k)            ) - &
              gupyz(i,j,k) * (   F2o3 * Kz(i,j,k)  +  EIGHT * PI * Sz(i,j,k)            ) + &
                        Gamyxx(i,j,k) * Rxx(i,j,k) + Gamyyy(i,j,k) * Ryy(i,j,k) + Gamyzz(i,j,k) * Rzz(i,j,k)   + &
                TWO * ( Gamyxy(i,j,k) * Rxy(i,j,k) + Gamyxz(i,j,k) * Rxz(i,j,k) + Gamyyz(i,j,k) * Ryz(i,j,k) ) )

   Gamz_rhs(i,j,k) = - TWO * (   Lapx(i,j,k) * Rxz(i,j,k) +   Lapy(i,j,k) * Ryz(i,j,k) +   Lapz(i,j,k) * Rzz(i,j,k) ) + &
        TWO * alpn1(i,j,k) * (                                                &
        -F3o2/chin1(i,j,k) * (   chix(i,j,k) * Rxz(i,j,k) +  chiy(i,j,k) * Ryz(i,j,k) +    chiz(i,j,k) * Rzz(i,j,k) ) - &
              gupxz(i,j,k) * (   F2o3 * Kx(i,j,k)  +  EIGHT * PI * Sx(i,j,k)            ) - &
              gupyz(i,j,k) * (   F2o3 * Ky(i,j,k)  +  EIGHT * PI * Sy(i,j,k)            ) - &
              gupzz(i,j,k) * (   F2o3 * Kz(i,j,k)  +  EIGHT * PI * Sz(i,j,k)            ) + &
                        Gamzxx(i,j,k) * Rxx(i,j,k) + Gamzyy(i,j,k) * Ryy(i,j,k) + Gamzzz(i,j,k) * Rzz(i,j,k)   + &
                TWO * ( Gamzxy(i,j,k) * Rxy(i,j,k) + Gamzxz(i,j,k) * Rxz(i,j,k) + Gamzyz(i,j,k) * Ryz(i,j,k) ) )
  enddo
  enddo
  enddo

  call fdderivs(ex,betax,gxxx,gxyx,gxzx,gyyx,gyzx,gzzx,&
                X,Y,Z,ANTI,SYM, SYM ,Symmetry,Lev)
  call fdderivs(ex,betay,gxxy,gxyy,gxzy,gyyy,gyzy,gzzy,&
                X,Y,Z,SYM ,ANTI,SYM ,Symmetry,Lev)
  call fdderivs(ex,betaz,gxxz,gxyz,gxzz,gyyz,gyzz,gzzz,&
                X,Y,Z,SYM ,SYM, ANTI,Symmetry,Lev)

  do k=1,ex(3)
  do j=1,ex(2)
  do i=1,ex(1)
  fxx(i,j,k) = gxxx(i,j,k) + gxyy(i,j,k) + gxzz(i,j,k)
  fxy(i,j,k) = gxyx(i,j,k) + gyyy(i,j,k) + gyzz(i,j,k)
  fxz(i,j,k) = gxzx(i,j,k) + gyzy(i,j,k) + gzzz(i,j,k)

  Gamxa(i,j,k) =       gupxx(i,j,k) * Gamxxx(i,j,k) + gupyy(i,j,k) * Gamxyy(i,j,k) + gupzz(i,j,k) * Gamxzz(i,j,k) + &
          TWO*( gupxy(i,j,k) * Gamxxy(i,j,k) + gupxz(i,j,k) * Gamxxz(i,j,k) + gupyz(i,j,k) * Gamxyz(i,j,k) )
  Gamya(i,j,k) =       gupxx(i,j,k) * Gamyxx(i,j,k) + gupyy(i,j,k) * Gamyyy(i,j,k) + gupzz(i,j,k) * Gamyzz(i,j,k) + &
          TWO*( gupxy(i,j,k) * Gamyxy(i,j,k) + gupxz(i,j,k) * Gamyxz(i,j,k) + gupyz(i,j,k) * Gamyyz(i,j,k) )
  Gamza(i,j,k) =       gupxx(i,j,k) * Gamzxx(i,j,k) + gupyy(i,j,k) * Gamzyy(i,j,k) + gupzz(i,j,k) * Gamzzz(i,j,k) + &
          TWO*( gupxy(i,j,k) * Gamzxy(i,j,k) + gupxz(i,j,k) * Gamzxz(i,j,k) + gupyz(i,j,k) * Gamzyz(i,j,k) )
  enddo
  enddo
  enddo

  call fderivs(ex,Gamx,Gamxx,Gamxy,Gamxz,X,Y,Z,ANTI,SYM ,SYM ,Symmetry,Lev)
  call fderivs(ex,Gamy,Gamyx,Gamyy,Gamyz,X,Y,Z,SYM ,ANTI,SYM ,Symmetry,Lev)
  call fderivs(ex,Gamz,Gamzx,Gamzy,Gamzz,X,Y,Z,SYM ,SYM ,ANTI,Symmetry,Lev)

  do k=1,ex(3)
  do j=1,ex(2)
  do i=1,ex(1)
  Gamx_rhs(i,j,k) =               Gamx_rhs(i,j,k) +  F2o3 *  Gamxa(i,j,k) * &
        ( betaxx(i,j,k) + betayy(i,j,k) + betazz(i,j,k) ) - &
                     Gamxa(i,j,k) * betaxx(i,j,k) - Gamya(i,j,k) * betaxy(i,j,k) - Gamza(i,j,k) * betaxz(i,j,k)  + &
             F1o3 * (gupxx(i,j,k) * fxx(i,j,k)    + gupxy(i,j,k) * fxy(i,j,k)    + gupxz(i,j,k) * fxz(i,j,k)    ) + &
                     gupxx(i,j,k) * gxxx(i,j,k)   + gupyy(i,j,k) * gyyx(i,j,k)   + gupzz(i,j,k) * gzzx(i,j,k)    + &
              TWO * (gupxy(i,j,k) * gxyx(i,j,k)   + gupxz(i,j,k) * gxzx(i,j,k)   + gupyz(i,j,k) * gyzx(i,j,k)  )

  Gamy_rhs(i,j,k) =               Gamy_rhs(i,j,k) +  F2o3 *  Gamya(i,j,k) * &
        ( betaxx(i,j,k) + betayy(i,j,k) + betazz(i,j,k) ) - &
                     Gamxa(i,j,k) * betayx(i,j,k) - Gamya(i,j,k) * betayy(i,j,k) - Gamza(i,j,k) * betayz(i,j,k)  + &
             F1o3 * (gupxy(i,j,k) * fxx(i,j,k)    + gupyy(i,j,k) * fxy(i,j,k)    + gupyz(i,j,k) * fxz(i,j,k)    ) + &
                     gupxx(i,j,k) * gxxy(i,j,k)   + gupyy(i,j,k) * gyyy(i,j,k)   + gupzz(i,j,k) * gzzy(i,j,k)    + &
              TWO * (gupxy(i,j,k) * gxyy(i,j,k)   + gupxz(i,j,k) * gxzy(i,j,k)   + gupyz(i,j,k) * gyzy(i,j,k)  )

  Gamz_rhs(i,j,k) =               Gamz_rhs(i,j,k) +  F2o3 *  Gamza(i,j,k) * &
        ( betaxx(i,j,k) + betayy(i,j,k) + betazz(i,j,k) ) - &
                     Gamxa(i,j,k) * betazx(i,j,k) - Gamya(i,j,k) * betazy(i,j,k) - Gamza(i,j,k) * betazz(i,j,k)  + &
             F1o3 * (gupxz(i,j,k) * fxx(i,j,k)    + gupyz(i,j,k) * fxy(i,j,k)    + gupzz(i,j,k) * fxz(i,j,k)    ) + &
                     gupxx(i,j,k) * gxxz(i,j,k)   + gupyy(i,j,k) * gyyz(i,j,k)   + gupzz(i,j,k) * gzzz(i,j,k)    + &
              TWO * (gupxy(i,j,k) * gxyz(i,j,k)   + gupxz(i,j,k) * gxzz(i,j,k)   + gupyz(i,j,k) * gyzz(i,j,k)  )    !rhs for Gam^i
  enddo
  enddo
  enddo


!first kind of connection stored in gij,k
  gxxx = gxx * Gamxxx + gxy * Gamyxx + gxz * Gamzxx
  gxyx = gxx * Gamxxy + gxy * Gamyxy + gxz * Gamzxy
  gxzx = gxx * Gamxxz + gxy * Gamyxz + gxz * Gamzxz
  gyyx = gxx * Gamxyy + gxy * Gamyyy + gxz * Gamzyy
  gyzx = gxx * Gamxyz + gxy * Gamyyz + gxz * Gamzyz
  gzzx = gxx * Gamxzz + gxy * Gamyzz + gxz * Gamzzz

  gxxy = gxy * Gamxxx + gyy * Gamyxx + gyz * Gamzxx
  gxyy = gxy * Gamxxy + gyy * Gamyxy + gyz * Gamzxy
  gxzy = gxy * Gamxxz + gyy * Gamyxz + gyz * Gamzxz
  gyyy = gxy * Gamxyy + gyy * Gamyyy + gyz * Gamzyy
  gyzy = gxy * Gamxyz + gyy * Gamyyz + gyz * Gamzyz
  gzzy = gxy * Gamxzz + gyy * Gamyzz + gyz * Gamzzz

  gxxz = gxz * Gamxxx + gyz * Gamyxx + gzz * Gamzxx
  gxyz = gxz * Gamxxy + gyz * Gamyxy + gzz * Gamzxy
  gxzz = gxz * Gamxxz + gyz * Gamyxz + gzz * Gamzxz
  gyyz = gxz * Gamxyy + gyz * Gamyyy + gzz * Gamzyy
  gyzz = gxz * Gamxyz + gyz * Gamyyz + gzz * Gamzyz
  gzzz = gxz * Gamxzz + gyz * Gamyzz + gzz * Gamzzz


!compute Ricci tensor for tilted metric
   call fdderivs(ex,dxx,fxx,fxy,fxz,fyy,fyz,fzz,X,Y,Z,SYM ,SYM ,SYM ,symmetry,Lev)
   Rxx =   gupxx * fxx + gupyy * fyy + gupzz * fzz + &
         ( gupxy * fxy + gupxz * fxz + gupyz * fyz ) * TWO

   call fdderivs(ex,dyy,fxx,fxy,fxz,fyy,fyz,fzz,X,Y,Z,SYM ,SYM ,SYM ,symmetry,Lev)
   Ryy =   gupxx * fxx + gupyy * fyy + gupzz * fzz + &
         ( gupxy * fxy + gupxz * fxz + gupyz * fyz ) * TWO

   call fdderivs(ex,dzz,fxx,fxy,fxz,fyy,fyz,fzz,X,Y,Z,SYM ,SYM ,SYM ,symmetry,Lev)
   Rzz =   gupxx * fxx + gupyy * fyy + gupzz * fzz + &
         ( gupxy * fxy + gupxz * fxz + gupyz * fyz ) * TWO

   call fdderivs(ex,gxy,fxx,fxy,fxz,fyy,fyz,fzz,X,Y,Z,ANTI, ANTI,SYM ,symmetry,Lev)
   Rxy =   gupxx * fxx + gupyy * fyy + gupzz * fzz + &
         ( gupxy * fxy + gupxz * fxz + gupyz * fyz ) * TWO

   call fdderivs(ex,gxz,fxx,fxy,fxz,fyy,fyz,fzz,X,Y,Z,ANTI ,SYM ,ANTI,symmetry,Lev)
   Rxz =   gupxx * fxx + gupyy * fyy + gupzz * fzz + &
         ( gupxy * fxy + gupxz * fxz + gupyz * fyz ) * TWO

   call fdderivs(ex,gyz,fxx,fxy,fxz,fyy,fyz,fzz,X,Y,Z,SYM ,ANTI ,ANTI,symmetry,Lev)
   Ryz =   gupxx * fxx + gupyy * fyy + gupzz * fzz + &
         ( gupxy * fxy + gupxz * fxz + gupyz * fyz ) * TWO


! six Ricci formulas as array statements (HEAD)
  Rxx =     - HALF * Rxx                                   + &
               gxx * Gamxx+ gxy * Gamyx   +    gxz * Gamzx + &
             Gamxa * gxxx +  Gamya * gxyx +  Gamza * gxzx  + &
   gupxx *(                                                  &
       TWO*(Gamxxx * gxxx + Gamyxx * gxyx + Gamzxx * gxzx) + &
            Gamxxx * gxxx + Gamyxx * gxxy + Gamzxx * gxxz )+ &
   gupxy *(                                                  &
       TWO*(Gamxxx * gxyx + Gamyxx * gyyx + Gamzxx * gyzx  + &
            Gamxxy * gxxx + Gamyxy * gxyx + Gamzxy * gxzx) + &
            Gamxxy * gxxx + Gamyxy * gxxy + Gamzxy * gxxz  + &
            Gamxxx * gxyx + Gamyxx * gxyy + Gamzxx * gxyz )+ &
   gupxz *(                                                  &
       TWO*(Gamxxx * gxzx + Gamyxx * gyzx + Gamzxx * gzzx  + &
            Gamxxz * gxxx + Gamyxz * gxyx + Gamzxz * gxzx) + &
            Gamxxz * gxxx + Gamyxz * gxxy + Gamzxz * gxxz  + &
            Gamxxx * gxzx + Gamyxx * gxzy + Gamzxx * gxzz )+ &
   gupyy *(                                                  &
       TWO*(Gamxxy * gxyx + Gamyxy * gyyx + Gamzxy * gyzx) + &
            Gamxxy * gxyx + Gamyxy * gxyy + Gamzxy * gxyz )+ &
   gupyz *(                                                  &
       TWO*(Gamxxy * gxzx + Gamyxy * gyzx + Gamzxy * gzzx  + &
            Gamxxz * gxyx + Gamyxz * gyyx + Gamzxz * gyzx) + &
            Gamxxz * gxyx + Gamyxz * gxyy + Gamzxz * gxyz  + &
            Gamxxy * gxzx + Gamyxy * gxzy + Gamzxy * gxzz )+ &
   gupzz *(                                                  &
       TWO*(Gamxxz * gxzx + Gamyxz * gyzx + Gamzxz * gzzx) + &
            Gamxxz * gxzx + Gamyxz * gxzy + Gamzxz * gxzz )

  Ryy =     - HALF * Ryy                                   + &
               gxy * Gamxy+  gyy * Gamyy  +  gyz * Gamzy   + &
             Gamxa * gxyy +  Gamya * gyyy +  Gamza * gyzy  + &
   gupxx *(                                                  &
       TWO*(Gamxxy * gxxy + Gamyxy * gxyy + Gamzxy * gxzy) + &
            Gamxxy * gxyx + Gamyxy * gxyy + Gamzxy * gxyz )+ &
   gupxy *(                                                  &
       TWO*(Gamxxy * gxyy + Gamyxy * gyyy + Gamzxy * gyzy  + &
            Gamxyy * gxxy + Gamyyy * gxyy + Gamzyy * gxzy) + &
            Gamxyy * gxyx + Gamyyy * gxyy + Gamzyy * gxyz  + &
            Gamxxy * gyyx + Gamyxy * gyyy + Gamzxy * gyyz )+ &
   gupxz *(                                                  &
       TWO*(Gamxxy * gxzy + Gamyxy * gyzy + Gamzxy * gzzy  + &
            Gamxyz * gxxy + Gamyyz * gxyy + Gamzyz * gxzy) + &
            Gamxyz * gxyx + Gamyyz * gxyy + Gamzyz * gxyz  + &
            Gamxxy * gyzx + Gamyxy * gyzy + Gamzxy * gyzz )+ &
   gupyy *(                                                  &
       TWO*(Gamxyy * gxyy + Gamyyy * gyyy + Gamzyy * gyzy) + &
            Gamxyy * gyyx + Gamyyy * gyyy + Gamzyy * gyyz )+ &
   gupyz *(                                                  &
       TWO*(Gamxyy * gxzy + Gamyyy * gyzy + Gamzyy * gzzy  + &
            Gamxyz * gxyy + Gamyyz * gyyy + Gamzyz * gyzy) + &
            Gamxyz * gyyx + Gamyyz * gyyy + Gamzyz * gyyz  + &
            Gamxyy * gyzx + Gamyyy * gyzy + Gamzyy * gyzz )+ &
   gupzz *(                                                  &
       TWO*(Gamxyz * gxzy + Gamyyz * gyzy + Gamzyz * gzzy) + &
            Gamxyz * gyzx + Gamyyz * gyzy + Gamzyz * gyzz )

  Rzz =     - HALF * Rzz                                   + &
               gxz * Gamxz+ gyz * Gamyz  +    gzz * Gamzz  + &
             Gamxa * gxzz +  Gamya * gyzz +  Gamza * gzzz  + &
   gupxx *(                                                  &
       TWO*(Gamxxz * gxxz + Gamyxz * gxyz + Gamzxz * gxzz) + &
            Gamxxz * gxzx + Gamyxz * gxzy + Gamzxz * gxzz )+ &
   gupxy *(                                                  &
       TWO*(Gamxxz * gxyz + Gamyxz * gyyz + Gamzxz * gyzz  + &
            Gamxyz * gxxz + Gamyyz * gxyz + Gamzyz * gxzz) + &
            Gamxyz * gxzx + Gamyyz * gxzy + Gamzyz * gxzz  + &
            Gamxxz * gyzx + Gamyxz * gyzy + Gamzxz * gyzz )+ &
   gupxz *(                                                  &
       TWO*(Gamxxz * gxzz + Gamyxz * gyzz + Gamzxz * gzzz  + &
            Gamxzz * gxxz + Gamyzz * gxyz + Gamzzz * gxzz) + &
            Gamxzz * gxzx + Gamyzz * gxzy + Gamzzz * gxzz  + &
            Gamxxz * gzzx + Gamyxz * gzzy + Gamzxz * gzzz )+ &
   gupyy *(                                                  &
       TWO*(Gamxyz * gxyz + Gamyyz * gyyz + Gamzyz * gyzz) + &
            Gamxyz * gyzx + Gamyyz * gyzy + Gamzyz * gyzz )+ &
   gupyz *(                                                  &
       TWO*(Gamxyz * gxzz + Gamyyz * gyzz + Gamzyz * gzzz  + &
            Gamxzz * gxyz + Gamyzz * gyyz + Gamzzz * gyzz) + &
            Gamxzz * gyzx + Gamyzz * gyzy + Gamzzz * gyzz  + &
            Gamxyz * gzzx + Gamyyz * gzzy + Gamzyz * gzzz )+ &
   gupzz *(                                                  &
       TWO*(Gamxzz * gxzz + Gamyzz * gyzz + Gamzzz * gzzz) + &
            Gamxzz * gzzx + Gamyzz * gzzy + Gamzzz * gzzz )

  Rxy = HALF*(     - Rxy                                   + &
               gxx * Gamxy +    gxy * Gamyy + gxz * Gamzy  + &
               gxy * Gamxx +    gyy * Gamyx + gyz * Gamzx  + &
             Gamxa * gxyx +  Gamya * gyyx +  Gamza * gyzx  + &
             Gamxa * gxxy +  Gamya * gxyy +  Gamza * gxzy )+ &
   gupxx *(                                                  &
            Gamxxx * gxxy + Gamyxx * gxyy + Gamzxx * gxzy  + &
            Gamxxy * gxxx + Gamyxy * gxyx + Gamzxy * gxzx  + &
            Gamxxx * gxyx + Gamyxx * gxyy + Gamzxx * gxyz )+ &
   gupxy *(                                                  &
            Gamxxx * gxyy + Gamyxx * gyyy + Gamzxx * gyzy  + &
            Gamxxy * gxyx + Gamyxy * gyyx + Gamzxy * gyzx  + &
            Gamxxy * gxyx + Gamyxy * gxyy + Gamzxy * gxyz  + &
            Gamxxy * gxxy + Gamyxy * gxyy + Gamzxy * gxzy  + &
            Gamxyy * gxxx + Gamyyy * gxyx + Gamzyy * gxzx  + &
            Gamxxx * gyyx + Gamyxx * gyyy + Gamzxx * gyyz )+ &
   gupxz *(                                                  &
            Gamxxx * gxzy + Gamyxx * gyzy + Gamzxx * gzzy  + &
            Gamxxy * gxzx + Gamyxy * gyzx + Gamzxy * gzzx  + &
            Gamxxz * gxyx + Gamyxz * gxyy + Gamzxz * gxyz  + &
            Gamxxz * gxxy + Gamyxz * gxyy + Gamzxz * gxzy  + &
            Gamxyz * gxxx + Gamyyz * gxyx + Gamzyz * gxzx  + &
            Gamxxx * gyzx + Gamyxx * gyzy + Gamzxx * gyzz )+ &
   gupyy *(                                                  &
            Gamxxy * gxyy + Gamyxy * gyyy + Gamzxy * gyzy  + &
            Gamxyy * gxyx + Gamyyy * gyyx + Gamzyy * gyzx  + &
            Gamxxy * gyyx + Gamyxy * gyyy + Gamzxy * gyyz )+ &
   gupyz *(                                                  &
            Gamxxy * gxzy + Gamyxy * gyzy + Gamzxy * gzzy  + &
            Gamxyy * gxzx + Gamyyy * gyzx + Gamzyy * gzzx  + &
            Gamxxz * gyyx + Gamyxz * gyyy + Gamzxz * gyyz  + &
            Gamxxz * gxyy + Gamyxz * gyyy + Gamzxz * gyzy  + &
            Gamxyz * gxyx + Gamyyz * gyyx + Gamzyz * gyzx  + &
            Gamxxy * gyzx + Gamyxy * gyzy + Gamzxy * gyzz )+ &
   gupzz *(                                                  &
            Gamxxz * gxzy + Gamyxz * gyzy + Gamzxz * gzzy  + &
            Gamxyz * gxzx + Gamyyz * gyzx + Gamzyz * gzzx  + &
            Gamxxz * gyzx + Gamyxz * gyzy + Gamzxz * gyzz )

  Rxz = HALF*(     - Rxz                                   + &
               gxx * Gamxz +  gxy * Gamyz + gxz * Gamzz    + &
               gxz * Gamxx +  gyz * Gamyx + gzz * Gamzx    + &
             Gamxa * gxzx +  Gamya * gyzx +  Gamza * gzzx  + &
             Gamxa * gxxz +  Gamya * gxyz +  Gamza * gxzz )+ &
   gupxx *(                                                  &
            Gamxxx * gxxz + Gamyxx * gxyz + Gamzxx * gxzz  + &
            Gamxxz * gxxx + Gamyxz * gxyx + Gamzxz * gxzx  + &
            Gamxxx * gxzx + Gamyxx * gxzy + Gamzxx * gxzz )+ &
   gupxy *(                                                  &
            Gamxxx * gxyz + Gamyxx * gyyz + Gamzxx * gyzz  + &
            Gamxxz * gxyx + Gamyxz * gyyx + Gamzxz * gyzx  + &
            Gamxxy * gxzx + Gamyxy * gxzy + Gamzxy * gxzz  + &
            Gamxxy * gxxz + Gamyxy * gxyz + Gamzxy * gxzz  + &
            Gamxyz * gxxx + Gamyyz * gxyx + Gamzyz * gxzx  + &
            Gamxxx * gyzx + Gamyxx * gyzy + Gamzxx * gyzz )+ &
   gupxz *(                                                  &
            Gamxxx * gxzz + Gamyxx * gyzz + Gamzxx * gzzz  + &
            Gamxxz * gxzx + Gamyxz * gyzx + Gamzxz * gzzx  + &
            Gamxxz * gxzx + Gamyxz * gxzy + Gamzxz * gxzz  + &
            Gamxxz * gxxz + Gamyxz * gxyz + Gamzxz * gxzz  + &
            Gamxzz * gxxx + Gamyzz * gxyx + Gamzzz * gxzx  + &
            Gamxxx * gzzx + Gamyxx * gzzy + Gamzxx * gzzz )+ &
   gupyy *(                                                  &
            Gamxxy * gxyz + Gamyxy * gyyz + Gamzxy * gyzz  + &
            Gamxyz * gxyx + Gamyyz * gyyx + Gamzyz * gyzx  + &
            Gamxxy * gyzx + Gamyxy * gyzy + Gamzxy * gyzz )+ &
   gupyz *(                                                  &
            Gamxxy * gxzz + Gamyxy * gyzz + Gamzxy * gzzz  + &
            Gamxyz * gxzx + Gamyyz * gyzx + Gamzyz * gzzx  + &
            Gamxxz * gyzx + Gamyxz * gyzy + Gamzxz * gyzz  + &
            Gamxxz * gxyz + Gamyxz * gyyz + Gamzxz * gyzz  + &
            Gamxzz * gxyx + Gamyzz * gyyx + Gamzzz * gyzx  + &
            Gamxxy * gzzx + Gamyxy * gzzy + Gamzxy * gzzz )+ &
   gupzz *(                                                  &
            Gamxxz * gxzz + Gamyxz * gyzz + Gamzxz * gzzz  + &
            Gamxzz * gxzx + Gamyzz * gyzx + Gamzzz * gzzx  + &
            Gamxxz * gzzx + Gamyxz * gzzy + Gamzxz * gzzz )

  Ryz = HALF*(     - Ryz                                   + &
               gxy * Gamxz + gyy * Gamyz + gyz * Gamzz     + &
               gxz * Gamxy + gyz * Gamyy + gzz * Gamzy     + &
             Gamxa * gxzy +  Gamya * gyzy +  Gamza * gzzy  + &
             Gamxa * gxyz +  Gamya * gyyz +  Gamza * gyzz )+ &
   gupxx *(                                                  &
            Gamxxy * gxxz + Gamyxy * gxyz + Gamzxy * gxzz  + &
            Gamxxz * gxxy + Gamyxz * gxyy + Gamzxz * gxzy  + &
            Gamxxy * gxzx + Gamyxy * gxzy + Gamzxy * gxzz )+ &
   gupxy *(                                                  &
            Gamxxy * gxyz + Gamyxy * gyyz + Gamzxy * gyzz  + &
            Gamxxz * gxyy + Gamyxz * gyyy + Gamzxz * gyzy  + &
            Gamxyy * gxzx + Gamyyy * gxzy + Gamzyy * gxzz  + &
            Gamxyy * gxxz + Gamyyy * gxyz + Gamzyy * gxzz  + &
            Gamxyz * gxxy + Gamyyz * gxyy + Gamzyz * gxzy  + &
            Gamxxy * gyzx + Gamyxy * gyzy + Gamzxy * gyzz )+ &
   gupxz *(                                                  &
            Gamxxy * gxzz + Gamyxy * gyzz + Gamzxy * gzzz  + &
            Gamxxz * gxzy + Gamyxz * gyzy + Gamzxz * gzzy  + &
            Gamxyz * gxzx + Gamyyz * gxzy + Gamzyz * gxzz  + &
            Gamxyz * gxxz + Gamyyz * gxyz + Gamzyz * gxzz  + &
            Gamxzz * gxxy + Gamyzz * gxyy + Gamzzz * gxzy  + &
            Gamxxy * gzzx + Gamyxy * gzzy + Gamzxy * gzzz )+ &
   gupyy *(                                                  &
            Gamxyy * gxyz + Gamyyy * gyyz + Gamzyy * gyzz  + &
            Gamxyz * gxyy + Gamyyz * gyyy + Gamzyz * gyzy  + &
            Gamxyy * gyzx + Gamyyy * gyzy + Gamzyy * gyzz )+ &
   gupyz *(                                                  &
            Gamxyy * gxzz + Gamyyy * gyzz + Gamzyy * gzzz  + &
            Gamxyz * gxzy + Gamyyz * gyzy + Gamzyz * gzzy  + &
            Gamxyz * gyzx + Gamyyz * gyzy + Gamzyz * gyzz  + &
            Gamxyz * gxyz + Gamyyz * gyyz + Gamzyz * gyzz  + &
            Gamxzz * gxyy + Gamyzz * gyyy + Gamzzz * gyzy  + &
            Gamxyy * gzzx + Gamyyy * gzzy + Gamzyy * gzzz )+ &
   gupzz *(                                                  &
            Gamxyz * gxzz + Gamyyz * gyzz + Gamzyz * gzzz  + &
            Gamxzz * gxzy + Gamyzz * gyzy + Gamzzz * gzzy  + &
            Gamxyz * gzzx + Gamyyz * gzzy + Gamzyz * gzzz )
!covariant second derivative of chi respect to tilted metric
  call fdderivs(ex,chi,fxx,fxy,fxz,fyy,fyz,fzz,X,Y,Z,SYM,SYM,SYM,Symmetry,Lev)

  do k=1,ex(3)
  do j=1,ex(2)
  do i=1,ex(1)
  fxx(i,j,k) = fxx(i,j,k) - Gamxxx(i,j,k) * chix(i,j,k) - Gamyxx(i,j,k) * chiy(i,j,k) - Gamzxx(i,j,k) * chiz(i,j,k)
  fxy(i,j,k) = fxy(i,j,k) - Gamxxy(i,j,k) * chix(i,j,k) - Gamyxy(i,j,k) * chiy(i,j,k) - Gamzxy(i,j,k) * chiz(i,j,k)
  fxz(i,j,k) = fxz(i,j,k) - Gamxxz(i,j,k) * chix(i,j,k) - Gamyxz(i,j,k) * chiy(i,j,k) - Gamzxz(i,j,k) * chiz(i,j,k)
  fyy(i,j,k) = fyy(i,j,k) - Gamxyy(i,j,k) * chix(i,j,k) - Gamyyy(i,j,k) * chiy(i,j,k) - Gamzyy(i,j,k) * chiz(i,j,k)
  fyz(i,j,k) = fyz(i,j,k) - Gamxyz(i,j,k) * chix(i,j,k) - Gamyyz(i,j,k) * chiy(i,j,k) - Gamzyz(i,j,k) * chiz(i,j,k)
  fzz(i,j,k) = fzz(i,j,k) - Gamxzz(i,j,k) * chix(i,j,k) - Gamyzz(i,j,k) * chiy(i,j,k) - Gamzzz(i,j,k) * chiz(i,j,k)
! Store D^l D_l chi - 3/(2*chi) D^l chi D_l chi in f
  f(i,j,k) =        gupxx(i,j,k) * ( fxx(i,j,k) - F3o2/chin1(i,j,k) * chix(i,j,k) * chix(i,j,k) ) + &
             gupyy(i,j,k) * ( fyy(i,j,k) - F3o2/chin1(i,j,k) * chiy(i,j,k) * chiy(i,j,k) ) + &
             gupzz(i,j,k) * ( fzz(i,j,k) - F3o2/chin1(i,j,k) * chiz(i,j,k) * chiz(i,j,k) ) + &
       TWO * gupxy(i,j,k) * ( fxy(i,j,k) - F3o2/chin1(i,j,k) * chix(i,j,k) * chiy(i,j,k) ) + &
       TWO * gupxz(i,j,k) * ( fxz(i,j,k) - F3o2/chin1(i,j,k) * chix(i,j,k) * chiz(i,j,k) ) + &
       TWO * gupyz(i,j,k) * ( fyz(i,j,k) - F3o2/chin1(i,j,k) * chiy(i,j,k) * chiz(i,j,k) )
! Add chi part to Ricci tensor:
  Rxx(i,j,k) = Rxx(i,j,k) + (fxx(i,j,k) - chix(i,j,k)*chix(i,j,k)/chin1(i,j,k)/TWO + gxx(i,j,k) * f(i,j,k))/chin1(i,j,k)/TWO
  Ryy(i,j,k) = Ryy(i,j,k) + (fyy(i,j,k) - chiy(i,j,k)*chiy(i,j,k)/chin1(i,j,k)/TWO + gyy(i,j,k) * f(i,j,k))/chin1(i,j,k)/TWO
  Rzz(i,j,k) = Rzz(i,j,k) + (fzz(i,j,k) - chiz(i,j,k)*chiz(i,j,k)/chin1(i,j,k)/TWO + gzz(i,j,k) * f(i,j,k))/chin1(i,j,k)/TWO
  Rxy(i,j,k) = Rxy(i,j,k) + (fxy(i,j,k) - chix(i,j,k)*chiy(i,j,k)/chin1(i,j,k)/TWO + gxy(i,j,k) * f(i,j,k))/chin1(i,j,k)/TWO
  Rxz(i,j,k) = Rxz(i,j,k) + (fxz(i,j,k) - chix(i,j,k)*chiz(i,j,k)/chin1(i,j,k)/TWO + gxz(i,j,k) * f(i,j,k))/chin1(i,j,k)/TWO
  Ryz(i,j,k) = Ryz(i,j,k) + (fyz(i,j,k) - chiy(i,j,k)*chiz(i,j,k)/chin1(i,j,k)/TWO + gyz(i,j,k) * f(i,j,k))/chin1(i,j,k)/TWO
  enddo
  enddo
  enddo



! covariant second derivatives of the lapse respect to physical metric
  call fdderivs(ex,Lap,fxx,fxy,fxz,fyy,fyz,fzz,X,Y,Z, &
                SYM,SYM,SYM,symmetry,Lev)

  do k=1,ex(3)
  do j=1,ex(2)
  do i=1,ex(1)
  gxxx(i,j,k) = (gupxx(i,j,k) * chix(i,j,k) + gupxy(i,j,k) * chiy(i,j,k) + &
     gupxz(i,j,k) * chiz(i,j,k))/chin1(i,j,k)
  gxxy(i,j,k) = (gupxy(i,j,k) * chix(i,j,k) + gupyy(i,j,k) * chiy(i,j,k) + &
     gupyz(i,j,k) * chiz(i,j,k))/chin1(i,j,k)
  gxxz(i,j,k) = (gupxz(i,j,k) * chix(i,j,k) + gupyz(i,j,k) * chiy(i,j,k) + &
     gupzz(i,j,k) * chiz(i,j,k))/chin1(i,j,k)
! now get physical second kind of connection
  Gamxxx(i,j,k) = Gamxxx(i,j,k) - ( (chix(i,j,k) + chix(i,j,k))/chin1(i,j,k) - gxx(i,j,k) * gxxx(i,j,k) )*HALF
  Gamyxx(i,j,k) = Gamyxx(i,j,k) - (                     - gxx(i,j,k) * gxxy(i,j,k) )*HALF
  Gamzxx(i,j,k) = Gamzxx(i,j,k) - (                     - gxx(i,j,k) * gxxz(i,j,k) )*HALF
  Gamxyy(i,j,k) = Gamxyy(i,j,k) - (                     - gyy(i,j,k) * gxxx(i,j,k) )*HALF
  Gamyyy(i,j,k) = Gamyyy(i,j,k) - ( (chiy(i,j,k) + chiy(i,j,k))/chin1(i,j,k) - gyy(i,j,k) * gxxy(i,j,k) )*HALF
  Gamzyy(i,j,k) = Gamzyy(i,j,k) - (                     - gyy(i,j,k) * gxxz(i,j,k) )*HALF
  Gamxzz(i,j,k) = Gamxzz(i,j,k) - (                     - gzz(i,j,k) * gxxx(i,j,k) )*HALF
  Gamyzz(i,j,k) = Gamyzz(i,j,k) - (                     - gzz(i,j,k) * gxxy(i,j,k) )*HALF
  Gamzzz(i,j,k) = Gamzzz(i,j,k) - ( (chiz(i,j,k) + chiz(i,j,k))/chin1(i,j,k) - gzz(i,j,k) * gxxz(i,j,k) )*HALF
  Gamxxy(i,j,k) = Gamxxy(i,j,k) - (  chiy(i,j,k)        /chin1(i,j,k) - gxy(i,j,k) * gxxx(i,j,k) )*HALF
  Gamyxy(i,j,k) = Gamyxy(i,j,k) - (         chix(i,j,k) /chin1(i,j,k) - gxy(i,j,k) * gxxy(i,j,k) )*HALF
  Gamzxy(i,j,k) = Gamzxy(i,j,k) - (                     - gxy(i,j,k) * gxxz(i,j,k) )*HALF
  Gamxxz(i,j,k) = Gamxxz(i,j,k) - (  chiz(i,j,k)        /chin1(i,j,k) - gxz(i,j,k) * gxxx(i,j,k) )*HALF
  Gamyxz(i,j,k) = Gamyxz(i,j,k) - (                     - gxz(i,j,k) * gxxy(i,j,k) )*HALF
  Gamzxz(i,j,k) = Gamzxz(i,j,k) - (         chix(i,j,k) /chin1(i,j,k) - gxz(i,j,k) * gxxz(i,j,k) )*HALF
  Gamxyz(i,j,k) = Gamxyz(i,j,k) - (                     - gyz(i,j,k) * gxxx(i,j,k) )*HALF
  Gamyyz(i,j,k) = Gamyyz(i,j,k) - (  chiz(i,j,k)        /chin1(i,j,k) - gyz(i,j,k) * gxxy(i,j,k) )*HALF
  Gamzyz(i,j,k) = Gamzyz(i,j,k) - (         chiy(i,j,k) /chin1(i,j,k) - gyz(i,j,k) * gxxz(i,j,k) )*HALF

  fxx(i,j,k) = fxx(i,j,k) - Gamxxx(i,j,k)*Lapx(i,j,k) - Gamyxx(i,j,k)*Lapy(i,j,k) - Gamzxx(i,j,k)*Lapz(i,j,k)
  fyy(i,j,k) = fyy(i,j,k) - Gamxyy(i,j,k)*Lapx(i,j,k) - Gamyyy(i,j,k)*Lapy(i,j,k) - Gamzyy(i,j,k)*Lapz(i,j,k)
  fzz(i,j,k) = fzz(i,j,k) - Gamxzz(i,j,k)*Lapx(i,j,k) - Gamyzz(i,j,k)*Lapy(i,j,k) - Gamzzz(i,j,k)*Lapz(i,j,k)
  fxy(i,j,k) = fxy(i,j,k) - Gamxxy(i,j,k)*Lapx(i,j,k) - Gamyxy(i,j,k)*Lapy(i,j,k) - Gamzxy(i,j,k)*Lapz(i,j,k)
  fxz(i,j,k) = fxz(i,j,k) - Gamxxz(i,j,k)*Lapx(i,j,k) - Gamyxz(i,j,k)*Lapy(i,j,k) - Gamzxz(i,j,k)*Lapz(i,j,k)
  fyz(i,j,k) = fyz(i,j,k) - Gamxyz(i,j,k)*Lapx(i,j,k) - Gamyyz(i,j,k)*Lapy(i,j,k) - Gamzyz(i,j,k)*Lapz(i,j,k)

! store D^i D_i Lap in trK_rhs upto chi
  trK_rhs(i,j,k) =    gupxx(i,j,k) * fxx(i,j,k) + gupyy(i,j,k) * fyy(i,j,k) + gupzz(i,j,k) * fzz(i,j,k) + &
        TWO* ( gupxy(i,j,k) * fxy(i,j,k) + gupxz(i,j,k) * fxz(i,j,k) + gupyz(i,j,k) * fyz(i,j,k) )
#if 1
!! follow bam code
  S(i,j,k) =  chin1(i,j,k) * ( gupxx(i,j,k) * Sxx(i,j,k) + gupyy(i,j,k) * Syy(i,j,k) + &
     gupzz(i,j,k) * Szz(i,j,k) + &
     TWO * ( gupxy(i,j,k) * Sxy(i,j,k) + gupxz(i,j,k) * Sxz(i,j,k) + gupyz(i,j,k) * Syz(i,j,k) ) )
  f(i,j,k) = F2o3 * trK(i,j,k) * trK(i,j,k) -( &
       gupxx(i,j,k) * ( &
       gupxx(i,j,k) * Axx(i,j,k) * Axx(i,j,k) + gupyy(i,j,k) * Axy(i,j,k) * Axy(i,j,k) + &
     gupzz(i,j,k) * Axz(i,j,k) * Axz(i,j,k) + &
       TWO * (gupxy(i,j,k) * Axx(i,j,k) * Axy(i,j,k) + gupxz(i,j,k) * Axx(i,j,k) * Axz(i,j,k) + &
     gupyz(i,j,k) * Axy(i,j,k) * Axz(i,j,k)) ) + &
       gupyy(i,j,k) * ( &
       gupxx(i,j,k) * Axy(i,j,k) * Axy(i,j,k) + gupyy(i,j,k) * Ayy(i,j,k) * Ayy(i,j,k) + &
     gupzz(i,j,k) * Ayz(i,j,k) * Ayz(i,j,k) + &
       TWO * (gupxy(i,j,k) * Axy(i,j,k) * Ayy(i,j,k) + gupxz(i,j,k) * Axy(i,j,k) * Ayz(i,j,k) + &
     gupyz(i,j,k) * Ayy(i,j,k) * Ayz(i,j,k)) ) + &
       gupzz(i,j,k) * ( &
       gupxx(i,j,k) * Axz(i,j,k) * Axz(i,j,k) + gupyy(i,j,k) * Ayz(i,j,k) * Ayz(i,j,k) + &
     gupzz(i,j,k) * Azz(i,j,k) * Azz(i,j,k) + &
       TWO * (gupxy(i,j,k) * Axz(i,j,k) * Ayz(i,j,k) + gupxz(i,j,k) * Axz(i,j,k) * Azz(i,j,k) + &
     gupyz(i,j,k) * Ayz(i,j,k) * Azz(i,j,k)) ) + &
       TWO * ( &
       gupxy(i,j,k) * ( &
       gupxx(i,j,k) * Axx(i,j,k) * Axy(i,j,k) + gupyy(i,j,k) * Axy(i,j,k) * Ayy(i,j,k) + &
     gupzz(i,j,k) * Axz(i,j,k) * Ayz(i,j,k) + &
       gupxy(i,j,k) * (Axx(i,j,k) * Ayy(i,j,k) + Axy(i,j,k) * Axy(i,j,k)) + &
       gupxz(i,j,k) * (Axx(i,j,k) * Ayz(i,j,k) + Axz(i,j,k) * Axy(i,j,k)) + &
       gupyz(i,j,k) * (Axy(i,j,k) * Ayz(i,j,k) + Axz(i,j,k) * Ayy(i,j,k)) ) + &
       gupxz(i,j,k) * ( &
       gupxx(i,j,k) * Axx(i,j,k) * Axz(i,j,k) + gupyy(i,j,k) * Axy(i,j,k) * Ayz(i,j,k) + &
     gupzz(i,j,k) * Axz(i,j,k) * Azz(i,j,k) + &
       gupxy(i,j,k) * (Axx(i,j,k) * Ayz(i,j,k) + Axy(i,j,k) * Axz(i,j,k)) + &
       gupxz(i,j,k) * (Axx(i,j,k) * Azz(i,j,k) + Axz(i,j,k) * Axz(i,j,k)) + &
       gupyz(i,j,k) * (Axy(i,j,k) * Azz(i,j,k) + Axz(i,j,k) * Ayz(i,j,k)) ) + &
       gupyz(i,j,k) * ( &
       gupxx(i,j,k) * Axy(i,j,k) * Axz(i,j,k) + gupyy(i,j,k) * Ayy(i,j,k) * Ayz(i,j,k) + &
     gupzz(i,j,k) * Ayz(i,j,k) * Azz(i,j,k) + &
       gupxy(i,j,k) * (Axy(i,j,k) * Ayz(i,j,k) + Ayy(i,j,k) * Axz(i,j,k)) + &
       gupxz(i,j,k) * (Axy(i,j,k) * Azz(i,j,k) + Ayz(i,j,k) * Axz(i,j,k)) + &
       gupyz(i,j,k) * (Ayy(i,j,k) * Azz(i,j,k) + Ayz(i,j,k) * Ayz(i,j,k)) ) )) -1.6d1*PI*rho(i,j,k) + &
     EIGHT * PI * S(i,j,k)
  f(i,j,k) = - F1o3 *(  gupxx(i,j,k) * fxx(i,j,k) + gupyy(i,j,k) * fyy(i,j,k) + gupzz(i,j,k) * fzz(i,j,k) + &
        TWO* ( gupxy(i,j,k) * fxy(i,j,k) + gupxz(i,j,k) * fxz(i,j,k) + gupyz(i,j,k) * fyz(i,j,k) ) + &
     alpn1(i,j,k)/chin1(i,j,k)*f(i,j,k))

  fxx(i,j,k) = alpn1(i,j,k) * (Rxx(i,j,k) - EIGHT * PI * Sxx(i,j,k)) - fxx(i,j,k)
  fxy(i,j,k) = alpn1(i,j,k) * (Rxy(i,j,k) - EIGHT * PI * Sxy(i,j,k)) - fxy(i,j,k)
  fxz(i,j,k) = alpn1(i,j,k) * (Rxz(i,j,k) - EIGHT * PI * Sxz(i,j,k)) - fxz(i,j,k)
  fyy(i,j,k) = alpn1(i,j,k) * (Ryy(i,j,k) - EIGHT * PI * Syy(i,j,k)) - fyy(i,j,k)
  fyz(i,j,k) = alpn1(i,j,k) * (Ryz(i,j,k) - EIGHT * PI * Syz(i,j,k)) - fyz(i,j,k)
  fzz(i,j,k) = alpn1(i,j,k) * (Rzz(i,j,k) - EIGHT * PI * Szz(i,j,k)) - fzz(i,j,k)
#else
! Add lapse and S_ij parts to Ricci tensor:

  fxx(i,j,k) = alpn1(i,j,k) * (Rxx(i,j,k) - EIGHT * PI * Sxx(i,j,k)) - fxx(i,j,k)
  fxy(i,j,k) = alpn1(i,j,k) * (Rxy(i,j,k) - EIGHT * PI * Sxy(i,j,k)) - fxy(i,j,k)
  fxz(i,j,k) = alpn1(i,j,k) * (Rxz(i,j,k) - EIGHT * PI * Sxz(i,j,k)) - fxz(i,j,k)
  fyy(i,j,k) = alpn1(i,j,k) * (Ryy(i,j,k) - EIGHT * PI * Syy(i,j,k)) - fyy(i,j,k)
  fyz(i,j,k) = alpn1(i,j,k) * (Ryz(i,j,k) - EIGHT * PI * Syz(i,j,k)) - fyz(i,j,k)
  fzz(i,j,k) = alpn1(i,j,k) * (Rzz(i,j,k) - EIGHT * PI * Szz(i,j,k)) - fzz(i,j,k)

! Compute trace-free part (note: chi^-1 and chi cancel!):

  f(i,j,k) = F1o3 *(  gupxx(i,j,k) * fxx(i,j,k) + gupyy(i,j,k) * fyy(i,j,k) + gupzz(i,j,k) * fzz(i,j,k) + &
        TWO* ( gupxy(i,j,k) * fxy(i,j,k) + gupxz(i,j,k) * fxz(i,j,k) + gupyz(i,j,k) * fyz(i,j,k) ) )
#endif

  Axx_rhs(i,j,k) = fxx(i,j,k) - gxx(i,j,k) * f(i,j,k)
  Ayy_rhs(i,j,k) = fyy(i,j,k) - gyy(i,j,k) * f(i,j,k)
  Azz_rhs(i,j,k) = fzz(i,j,k) - gzz(i,j,k) * f(i,j,k)
  Axy_rhs(i,j,k) = fxy(i,j,k) - gxy(i,j,k) * f(i,j,k)
  Axz_rhs(i,j,k) = fxz(i,j,k) - gxz(i,j,k) * f(i,j,k)
  Ayz_rhs(i,j,k) = fyz(i,j,k) - gyz(i,j,k) * f(i,j,k)

  enddo
  enddo
  enddo


  do k=1,ex(3)
  do j=1,ex(2)
  do i=1,ex(1)
! Now: store A_il A^l_j into fij:

  fxx(i,j,k) =       gupxx(i,j,k) * Axx(i,j,k) * Axx(i,j,k) + gupyy(i,j,k) * Axy(i,j,k) * Axy(i,j,k) + &
     gupzz(i,j,k) * Axz(i,j,k) * Axz(i,j,k) + &
       TWO * (gupxy(i,j,k) * Axx(i,j,k) * Axy(i,j,k) + gupxz(i,j,k) * Axx(i,j,k) * Axz(i,j,k) + &
     gupyz(i,j,k) * Axy(i,j,k) * Axz(i,j,k))
  fyy(i,j,k) =       gupxx(i,j,k) * Axy(i,j,k) * Axy(i,j,k) + gupyy(i,j,k) * Ayy(i,j,k) * Ayy(i,j,k) + &
     gupzz(i,j,k) * Ayz(i,j,k) * Ayz(i,j,k) + &
       TWO * (gupxy(i,j,k) * Axy(i,j,k) * Ayy(i,j,k) + gupxz(i,j,k) * Axy(i,j,k) * Ayz(i,j,k) + &
     gupyz(i,j,k) * Ayy(i,j,k) * Ayz(i,j,k))
  fzz(i,j,k) =       gupxx(i,j,k) * Axz(i,j,k) * Axz(i,j,k) + gupyy(i,j,k) * Ayz(i,j,k) * Ayz(i,j,k) + &
     gupzz(i,j,k) * Azz(i,j,k) * Azz(i,j,k) + &
       TWO * (gupxy(i,j,k) * Axz(i,j,k) * Ayz(i,j,k) + gupxz(i,j,k) * Axz(i,j,k) * Azz(i,j,k) + &
     gupyz(i,j,k) * Ayz(i,j,k) * Azz(i,j,k))
  fxy(i,j,k) =       gupxx(i,j,k) * Axx(i,j,k) * Axy(i,j,k) + gupyy(i,j,k) * Axy(i,j,k) * Ayy(i,j,k) + &
     gupzz(i,j,k) * Axz(i,j,k) * Ayz(i,j,k) + &
              gupxy(i,j,k) *(Axx(i,j,k) * Ayy(i,j,k) + Axy(i,j,k) * Axy(i,j,k))                            + &
              gupxz(i,j,k) *(Axx(i,j,k) * Ayz(i,j,k) + Axz(i,j,k) * Axy(i,j,k))                            + &
              gupyz(i,j,k) *(Axy(i,j,k) * Ayz(i,j,k) + Axz(i,j,k) * Ayy(i,j,k))
  fxz(i,j,k) =       gupxx(i,j,k) * Axx(i,j,k) * Axz(i,j,k) + gupyy(i,j,k) * Axy(i,j,k) * Ayz(i,j,k) + &
     gupzz(i,j,k) * Axz(i,j,k) * Azz(i,j,k) + &
              gupxy(i,j,k) *(Axx(i,j,k) * Ayz(i,j,k) + Axy(i,j,k) * Axz(i,j,k))                            + &
              gupxz(i,j,k) *(Axx(i,j,k) * Azz(i,j,k) + Axz(i,j,k) * Axz(i,j,k))                            + &
              gupyz(i,j,k) *(Axy(i,j,k) * Azz(i,j,k) + Axz(i,j,k) * Ayz(i,j,k))
  fyz(i,j,k) =       gupxx(i,j,k) * Axy(i,j,k) * Axz(i,j,k) + gupyy(i,j,k) * Ayy(i,j,k) * Ayz(i,j,k) + &
     gupzz(i,j,k) * Ayz(i,j,k) * Azz(i,j,k) + &
              gupxy(i,j,k) *(Axy(i,j,k) * Ayz(i,j,k) + Ayy(i,j,k) * Axz(i,j,k))                            + &
              gupxz(i,j,k) *(Axy(i,j,k) * Azz(i,j,k) + Ayz(i,j,k) * Axz(i,j,k))                            + &
              gupyz(i,j,k) *(Ayy(i,j,k) * Azz(i,j,k) + Ayz(i,j,k) * Ayz(i,j,k))

  f(i,j,k) = chin1(i,j,k)
! store D^i D_i Lap in trK_rhs
  trK_rhs(i,j,k) = f(i,j,k)*trK_rhs(i,j,k)

  Axx_rhs(i,j,k) =           f(i,j,k) * Axx_rhs(i,j,k)+ alpn1(i,j,k) * (trK(i,j,k) * Axx(i,j,k) - TWO * fxx(i,j,k))  + &
           TWO * (  Axx(i,j,k) * betaxx(i,j,k) +   Axy(i,j,k) * betayx(i,j,k) +   Axz(i,j,k) * betazx(i,j,k) )- &
             F2o3 * Axx(i,j,k) * (betaxx(i,j,k) + betayy(i,j,k) + betazz(i,j,k))

  Ayy_rhs(i,j,k) =           f(i,j,k) * Ayy_rhs(i,j,k)+ alpn1(i,j,k) * (trK(i,j,k) * Ayy(i,j,k) - TWO * fyy(i,j,k))  + &
           TWO * (  Axy(i,j,k) * betaxy(i,j,k) +   Ayy(i,j,k) * betayy(i,j,k) +   Ayz(i,j,k) * betazy(i,j,k) )- &
             F2o3 * Ayy(i,j,k) * (betaxx(i,j,k) + betayy(i,j,k) + betazz(i,j,k))

  Azz_rhs(i,j,k) =           f(i,j,k) * Azz_rhs(i,j,k)+ alpn1(i,j,k) * (trK(i,j,k) * Azz(i,j,k) - TWO * fzz(i,j,k))  + &
           TWO * (  Axz(i,j,k) * betaxz(i,j,k) +   Ayz(i,j,k) * betayz(i,j,k) +   Azz(i,j,k) * betazz(i,j,k) )- &
             F2o3 * Azz(i,j,k) * (betaxx(i,j,k) + betayy(i,j,k) + betazz(i,j,k))

  Axy_rhs(i,j,k) =           f(i,j,k) * Axy_rhs(i,j,k)+ alpn1(i,j,k) *( trK(i,j,k) * Axy(i,j,k)  - TWO * fxy(i,j,k) )+ &
                    Axx(i,j,k) * betaxy(i,j,k)                  +   Axz(i,j,k) * betazy(i,j,k)  + &
                                     Ayy(i,j,k) * betayx(i,j,k) +   Ayz(i,j,k) * betazx(i,j,k)  + &
             F1o3 * Axy(i,j,k) * (betaxx(i,j,k) + betayy(i,j,k) + betazz(i,j,k))                -   Axy(i,j,k) * betazz(i,j,k)

  Ayz_rhs(i,j,k) =           f(i,j,k) * Ayz_rhs(i,j,k)+ alpn1(i,j,k) *( trK(i,j,k) * Ayz(i,j,k)  - TWO * fyz(i,j,k) )+ &
                    Axy(i,j,k) * betaxz(i,j,k) +   Ayy(i,j,k) * betayz(i,j,k)                   + &
                    Axz(i,j,k) * betaxy(i,j,k)                  +   Azz(i,j,k) * betazy(i,j,k)  + &
             F1o3 * Ayz(i,j,k) * (betaxx(i,j,k) + betayy(i,j,k) + betazz(i,j,k))                -   Ayz(i,j,k) * betaxx(i,j,k)

  Axz_rhs(i,j,k) =           f(i,j,k) * Axz_rhs(i,j,k)+ alpn1(i,j,k) *( trK(i,j,k) * Axz(i,j,k)  - TWO * fxz(i,j,k) )+ &
                    Axx(i,j,k) * betaxz(i,j,k) +   Axy(i,j,k) * betayz(i,j,k)                   + &
                                     Ayz(i,j,k) * betayx(i,j,k) +   Azz(i,j,k) * betazx(i,j,k)  + &
             F1o3 * Axz(i,j,k) * (betaxx(i,j,k) + betayy(i,j,k) + betazz(i,j,k))                - &
     Axz(i,j,k) * betayy(i,j,k)      !rhs for Aij

! Compute trace of S_ij

  S(i,j,k) =  f(i,j,k) * ( gupxx(i,j,k) * Sxx(i,j,k) + gupyy(i,j,k) * Syy(i,j,k) + gupzz(i,j,k) * Szz(i,j,k) + &
     TWO * ( gupxy(i,j,k) * Sxy(i,j,k) + gupxz(i,j,k) * Sxz(i,j,k) + gupyz(i,j,k) * Syz(i,j,k) ) )

  trK_rhs(i,j,k) = - trK_rhs(i,j,k) + alpn1(i,j,k) *( F1o3 * trK(i,j,k) * trK(i,j,k)         + &
                gupxx(i,j,k) * fxx(i,j,k) + gupyy(i,j,k) * fyy(i,j,k) + gupzz(i,j,k) * fzz(i,j,k)   + &
        TWO * ( gupxy(i,j,k) * fxy(i,j,k) + gupxz(i,j,k) * fxz(i,j,k) + gupyz(i,j,k) * fyz(i,j,k) ) + &
       FOUR * PI * ( rho(i,j,k) + S(i,j,k) ))                                !rhs for trK(i,j,k)

!!!! gauge variable part

  Lap_rhs(i,j,k) = -TWO*alpn1(i,j,k)*trK(i,j,k)
  betax_rhs(i,j,k) = FF*dtSfx(i,j,k)
  betay_rhs(i,j,k) = FF*dtSfy(i,j,k)
  betaz_rhs(i,j,k) = FF*dtSfz(i,j,k)

  dtSfx_rhs(i,j,k) = Gamx_rhs(i,j,k) - eta*dtSfx(i,j,k)
  dtSfy_rhs(i,j,k) = Gamy_rhs(i,j,k) - eta*dtSfy(i,j,k)
  dtSfz_rhs(i,j,k) = Gamz_rhs(i,j,k) - eta*dtSfz(i,j,k)

  enddo
  enddo
  enddo
  SSS(1)=SYM
  SSS(2)=SYM
  SSS(3)=SYM

  AAS(1)=ANTI
  AAS(2)=ANTI
  AAS(3)=SYM

  ASA(1)=ANTI
  ASA(2)=SYM
  ASA(3)=ANTI

  SAA(1)=SYM
  SAA(2)=ANTI
  SAA(3)=ANTI

  ASS(1)=ANTI
  ASS(2)=SYM
  ASS(3)=SYM

  SAS(1)=SYM
  SAS(2)=ANTI
  SAS(3)=SYM

  SSA(1)=SYM
  SSA(2)=SYM
  SSA(3)=ANTI


!!!!!!!!!advection term part

  call lopsided(ex,X,Y,Z,gxx,gxx_rhs,betax,betay,betaz,Symmetry,SSS)
  call lopsided(ex,X,Y,Z,gxy,gxy_rhs,betax,betay,betaz,Symmetry,AAS)
  call lopsided(ex,X,Y,Z,gxz,gxz_rhs,betax,betay,betaz,Symmetry,ASA)
  call lopsided(ex,X,Y,Z,gyy,gyy_rhs,betax,betay,betaz,Symmetry,SSS)
  call lopsided(ex,X,Y,Z,gyz,gyz_rhs,betax,betay,betaz,Symmetry,SAA)
  call lopsided(ex,X,Y,Z,gzz,gzz_rhs,betax,betay,betaz,Symmetry,SSS)

  call lopsided(ex,X,Y,Z,Axx,Axx_rhs,betax,betay,betaz,Symmetry,SSS)
  call lopsided(ex,X,Y,Z,Axy,Axy_rhs,betax,betay,betaz,Symmetry,AAS)
  call lopsided(ex,X,Y,Z,Axz,Axz_rhs,betax,betay,betaz,Symmetry,ASA)
  call lopsided(ex,X,Y,Z,Ayy,Ayy_rhs,betax,betay,betaz,Symmetry,SSS)
  call lopsided(ex,X,Y,Z,Ayz,Ayz_rhs,betax,betay,betaz,Symmetry,SAA)
  call lopsided(ex,X,Y,Z,Azz,Azz_rhs,betax,betay,betaz,Symmetry,SSS)

  call lopsided(ex,X,Y,Z,chi,chi_rhs,betax,betay,betaz,Symmetry,SSS)
  call lopsided(ex,X,Y,Z,trK,trK_rhs,betax,betay,betaz,Symmetry,SSS)

  call lopsided(ex,X,Y,Z,Gamx,Gamx_rhs,betax,betay,betaz,Symmetry,ASS)
  call lopsided(ex,X,Y,Z,Gamy,Gamy_rhs,betax,betay,betaz,Symmetry,SAS)
  call lopsided(ex,X,Y,Z,Gamz,Gamz_rhs,betax,betay,betaz,Symmetry,SSA)
!!
  call lopsided(ex,X,Y,Z,Lap,Lap_rhs,betax,betay,betaz,Symmetry,SSS)

  call lopsided(ex,X,Y,Z,betax,betax_rhs,betax,betay,betaz,Symmetry,ASS)
  call lopsided(ex,X,Y,Z,betay,betay_rhs,betax,betay,betaz,Symmetry,SAS)
  call lopsided(ex,X,Y,Z,betaz,betaz_rhs,betax,betay,betaz,Symmetry,SSA)

  call lopsided(ex,X,Y,Z,dtSfx,dtSfx_rhs,betax,betay,betaz,Symmetry,ASS)
  call lopsided(ex,X,Y,Z,dtSfy,dtSfy_rhs,betax,betay,betaz,Symmetry,SAS)
  call lopsided(ex,X,Y,Z,dtSfz,dtSfz_rhs,betax,betay,betaz,Symmetry,SSA)


  if(eps>0)then 
! usual Kreiss-Oliger dissipation      
  call kodis(ex,X,Y,Z,chi,chi_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,trK,trK_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,dxx,gxx_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,gxy,gxy_rhs,AAS,Symmetry,eps)
  call kodis(ex,X,Y,Z,gxz,gxz_rhs,ASA,Symmetry,eps)
  call kodis(ex,X,Y,Z,dyy,gyy_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,gyz,gyz_rhs,SAA,Symmetry,eps)
  call kodis(ex,X,Y,Z,dzz,gzz_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Axx,Axx_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Axy,Axy_rhs,AAS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Axz,Axz_rhs,ASA,Symmetry,eps)
  call kodis(ex,X,Y,Z,Ayy,Ayy_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Ayz,Ayz_rhs,SAA,Symmetry,eps)
  call kodis(ex,X,Y,Z,Azz,Azz_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Gamx,Gamx_rhs,ASS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Gamy,Gamy_rhs,SAS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Gamz,Gamz_rhs,SSA,Symmetry,eps)

#if 1 
!! bam does not apply dissipation on gauge variables
  call kodis(ex,X,Y,Z,Lap,Lap_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,betax,betax_rhs,ASS,Symmetry,eps)
  call kodis(ex,X,Y,Z,betay,betay_rhs,SAS,Symmetry,eps)
  call kodis(ex,X,Y,Z,betaz,betaz_rhs,SSA,Symmetry,eps)
  call kodis(ex,X,Y,Z,dtSfx,dtSfx_rhs,ASS,Symmetry,eps)
  call kodis(ex,X,Y,Z,dtSfy,dtSfy_rhs,SAS,Symmetry,eps)
  call kodis(ex,X,Y,Z,dtSfz,dtSfz_rhs,SSA,Symmetry,eps)
#endif


  endif

  if(co == 0)then
! ham_Res = trR + 2/3 * K^2 - A_ij * A^ij - 16 * PI * rho
! here trR is respect to physical metric
  ham_Res =   gupxx * Rxx + gupyy * Ryy + gupzz * Rzz + &
        TWO* ( gupxy * Rxy + gupxz * Rxz + gupyz * Ryz )

  ham_Res = chin1*ham_Res + F2o3 * trK * trK -(&
       gupxx * ( &
       gupxx * Axx * Axx + gupyy * Axy * Axy + gupzz * Axz * Axz + &
       TWO * (gupxy * Axx * Axy + gupxz * Axx * Axz + gupyz * Axy * Axz) ) + &
       gupyy * ( &
       gupxx * Axy * Axy + gupyy * Ayy * Ayy + gupzz * Ayz * Ayz + &
       TWO * (gupxy * Axy * Ayy + gupxz * Axy * Ayz + gupyz * Ayy * Ayz) ) + &
       gupzz * ( &
       gupxx * Axz * Axz + gupyy * Ayz * Ayz + gupzz * Azz * Azz + &
       TWO * (gupxy * Axz * Ayz + gupxz * Axz * Azz + gupyz * Ayz * Azz) ) + &
       TWO * ( &
       gupxy * ( &
       gupxx * Axx * Axy + gupyy * Axy * Ayy + gupzz * Axz * Ayz + &
       gupxy * (Axx * Ayy + Axy * Axy) + &
       gupxz * (Axx * Ayz + Axz * Axy) + &
       gupyz * (Axy * Ayz + Axz * Ayy) ) + &
       gupxz * ( &
       gupxx * Axx * Axz + gupyy * Axy * Ayz + gupzz * Axz * Azz + &
       gupxy * (Axx * Ayz + Axy * Axz) + &
       gupxz * (Axx * Azz + Axz * Axz) + &
       gupyz * (Axy * Azz + Axz * Ayz) ) + &
       gupyz * ( &
       gupxx * Axy * Axz + gupyy * Ayy * Ayz + gupzz * Ayz * Azz + &
       gupxy * (Axy * Ayz + Ayy * Axz) + &
       gupxz * (Axy * Azz + Ayz * Axz) + &
       gupyz * (Ayy * Azz + Ayz * Ayz) ) ))- F16 * PI * rho

! mov_Res_j = gupkj*(-1/chi d_k chi*A_ij + D_k A_ij) - 2/3 d_j trK - 8 PI s_j where D respect to physical metric
! store D_i A_jk - 1/chi d_i chi*A_jk in gjk_i
  call fderivs(ex,Axx,gxxx,gxxy,gxxz,X,Y,Z,SYM ,SYM ,SYM ,Symmetry,0)
  call fderivs(ex,Axy,gxyx,gxyy,gxyz,X,Y,Z,ANTI,ANTI,SYM ,Symmetry,0)
  call fderivs(ex,Axz,gxzx,gxzy,gxzz,X,Y,Z,ANTI,SYM ,ANTI,Symmetry,0)
  call fderivs(ex,Ayy,gyyx,gyyy,gyyz,X,Y,Z,SYM ,SYM ,SYM ,Symmetry,0)
  call fderivs(ex,Ayz,gyzx,gyzy,gyzz,X,Y,Z,SYM ,ANTI,ANTI,Symmetry,0)
  call fderivs(ex,Azz,gzzx,gzzy,gzzz,X,Y,Z,SYM ,SYM ,SYM ,Symmetry,0)

  gxxx = gxxx - (  Gamxxx * Axx + Gamyxx * Axy + Gamzxx * Axz &
                 + Gamxxx * Axx + Gamyxx * Axy + Gamzxx * Axz) - chix*Axx/chin1
  gxyx = gxyx - (  Gamxxy * Axx + Gamyxy * Axy + Gamzxy * Axz &
                 + Gamxxx * Axy + Gamyxx * Ayy + Gamzxx * Ayz) - chix*Axy/chin1
  gxzx = gxzx - (  Gamxxz * Axx + Gamyxz * Axy + Gamzxz * Axz &
                 + Gamxxx * Axz + Gamyxx * Ayz + Gamzxx * Azz) - chix*Axz/chin1
  gyyx = gyyx - (  Gamxxy * Axy + Gamyxy * Ayy + Gamzxy * Ayz &
                 + Gamxxy * Axy + Gamyxy * Ayy + Gamzxy * Ayz) - chix*Ayy/chin1
  gyzx = gyzx - (  Gamxxz * Axy + Gamyxz * Ayy + Gamzxz * Ayz &
                 + Gamxxy * Axz + Gamyxy * Ayz + Gamzxy * Azz) - chix*Ayz/chin1
  gzzx = gzzx - (  Gamxxz * Axz + Gamyxz * Ayz + Gamzxz * Azz &
                 + Gamxxz * Axz + Gamyxz * Ayz + Gamzxz * Azz) - chix*Azz/chin1
  gxxy = gxxy - (  Gamxxy * Axx + Gamyxy * Axy + Gamzxy * Axz &
                 + Gamxxy * Axx + Gamyxy * Axy + Gamzxy * Axz) - chiy*Axx/chin1
  gxyy = gxyy - (  Gamxyy * Axx + Gamyyy * Axy + Gamzyy * Axz &
                 + Gamxxy * Axy + Gamyxy * Ayy + Gamzxy * Ayz) - chiy*Axy/chin1
  gxzy = gxzy - (  Gamxyz * Axx + Gamyyz * Axy + Gamzyz * Axz &
                 + Gamxxy * Axz + Gamyxy * Ayz + Gamzxy * Azz) - chiy*Axz/chin1
  gyyy = gyyy - (  Gamxyy * Axy + Gamyyy * Ayy + Gamzyy * Ayz &
                 + Gamxyy * Axy + Gamyyy * Ayy + Gamzyy * Ayz) - chiy*Ayy/chin1
  gyzy = gyzy - (  Gamxyz * Axy + Gamyyz * Ayy + Gamzyz * Ayz &
                 + Gamxyy * Axz + Gamyyy * Ayz + Gamzyy * Azz) - chiy*Ayz/chin1
  gzzy = gzzy - (  Gamxyz * Axz + Gamyyz * Ayz + Gamzyz * Azz &
                 + Gamxyz * Axz + Gamyyz * Ayz + Gamzyz * Azz) - chiy*Azz/chin1
  gxxz = gxxz - (  Gamxxz * Axx + Gamyxz * Axy + Gamzxz * Axz &
                 + Gamxxz * Axx + Gamyxz * Axy + Gamzxz * Axz) - chiz*Axx/chin1
  gxyz = gxyz - (  Gamxyz * Axx + Gamyyz * Axy + Gamzyz * Axz &
                 + Gamxxz * Axy + Gamyxz * Ayy + Gamzxz * Ayz) - chiz*Axy/chin1
  gxzz = gxzz - (  Gamxzz * Axx + Gamyzz * Axy + Gamzzz * Axz &
                 + Gamxxz * Axz + Gamyxz * Ayz + Gamzxz * Azz) - chiz*Axz/chin1
  gyyz = gyyz - (  Gamxyz * Axy + Gamyyz * Ayy + Gamzyz * Ayz &
                 + Gamxyz * Axy + Gamyyz * Ayy + Gamzyz * Ayz) - chiz*Ayy/chin1
  gyzz = gyzz - (  Gamxzz * Axy + Gamyzz * Ayy + Gamzzz * Ayz &
                 + Gamxyz * Axz + Gamyyz * Ayz + Gamzyz * Azz) - chiz*Ayz/chin1
  gzzz = gzzz - (  Gamxzz * Axz + Gamyzz * Ayz + Gamzzz * Azz &
                 + Gamxzz * Axz + Gamyzz * Ayz + Gamzzz * Azz) - chiz*Azz/chin1
movx_Res = gupxx*gxxx + gupyy*gxyy + gupzz*gxzz &
          +gupxy*gxyx + gupxz*gxzx + gupyz*gxzy &
          +gupxy*gxxy + gupxz*gxxz + gupyz*gxyz
movy_Res = gupxx*gxyx + gupyy*gyyy + gupzz*gyzz &
          +gupxy*gyyx + gupxz*gyzx + gupyz*gyzy &
          +gupxy*gxyy + gupxz*gxyz + gupyz*gyyz
movz_Res = gupxx*gxzx + gupyy*gyzy + gupzz*gzzz &
          +gupxy*gyzx + gupxz*gzzx + gupyz*gzzy &
          +gupxy*gxzy + gupxz*gxzz + gupyz*gyzz

movx_Res = movx_Res - F2o3*Kx - F8*PI*sx
movy_Res = movy_Res - F2o3*Ky - F8*PI*sy
movz_Res = movz_Res - F2o3*Kz - F8*PI*sz
  endif



  gont = 0

  return

  end function compute_rhs_bssn
