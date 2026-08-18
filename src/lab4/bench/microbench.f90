program microbench
  use omp_lib
  implicit none
  real*8, allocatable :: f(:,:,:), fx(:,:,:), fy(:,:,:), fz(:,:,:)
  real*8, allocatable :: fxx(:,:,:), fxy(:,:,:), fxz(:,:,:), fyy(:,:,:), fyz(:,:,:), fzz(:,:,:), frhs(:,:,:)
  real*8, allocatable :: X(:), Y(:), Z(:)
  real*8, dimension(3) :: SoA
  real*8 :: t0, t1
  integer :: it, nrep, NX, NY, NZ, nt, i
  integer :: ex(3)
  integer, parameter :: Symmetry = 1
  character(len=32) :: arg
  integer, parameter :: sz_nx(5) = (/12, 24, 48, 96, 24/)
  integer, parameter :: sz_ny(5) = (/60, 60, 60, 120, 240/)
  integer, parameter :: sz_nz(5) = (/24, 24, 24, 48, 96/)

  SoA = (/1.d0,1.d0,1.d0/)

  do i = 1, 5
    NX = sz_nx(i); NY = sz_ny(i); NZ = sz_nz(i)
    ex = (/NX,NY,NZ/)
    allocate(f(NX,NY,NZ), fx(NX,NY,NZ), fy(NX,NY,NZ), fz(NX,NY,NZ))
    allocate(fxx(NX,NY,NZ), fxy(NX,NY,NZ), fxz(NX,NY,NZ))
    allocate(fyy(NX,NY,NZ), fyz(NX,NY,NZ), fzz(NX,NY,NZ), frhs(NX,NY,NZ))
    allocate(X(NX), Y(NY), Z(NZ))
    do it = 1, NX; X(it) = (it-1)*1.d0; end do
    do it = 1, NY; Y(it) = (it-1)*1.d0; end do
    do it = 1, NZ; Z(it) = (it-1)*1.d0; end do
    f = 1.d0; frhs = 0.d0

    nrep = max(50, 4000000/(NX*NY*NZ))
    write(*,'(A,3(I5,A))') "SIZE", NX, "x", NY, "x", NZ, "  (", NX*NY*NZ, " pts)"

    do nt = 1, 8
      call omp_set_num_threads(nt)
      t0 = omp_get_wtime()
      do it = 1, nrep
        call fderivs(ex,f,fx,fy,fz,X,Y,Z,1.d0,1.d0,1.d0,Symmetry,0)
      end do
      t1 = omp_get_wtime()
      write(*,'(A,I2,A,F10.6,A)') "  fderivs T=", nt, ": ", t1-t0, " s"
    end do

    do nt = 1, 8
      call omp_set_num_threads(nt)
      t0 = omp_get_wtime()
      do it = 1, nrep
        call kodis(ex,X,Y,Z,f,frhs,SoA,Symmetry,0.15d0)
      end do
      t1 = omp_get_wtime()
      write(*,'(A,I2,A,F10.6,A)') "  kodis   T=", nt, ": ", t1-t0, " s"
    end do

    do nt = 1, 8
      call omp_set_num_threads(nt)
      t0 = omp_get_wtime()
      do it = 1, nrep
        call lopsided(ex,X,Y,Z,f,frhs,fx,fy,fz,Symmetry,SoA)
      end do
      t1 = omp_get_wtime()
      write(*,'(A,I2,A,F10.6,A)') "  lopsided T=", nt, ": ", t1-t0, " s"
    end do

    deallocate(f,fx,fy,fz,fxx,fxy,fxz,fyy,fyz,fzz,frhs,X,Y,Z)
  end do
end program microbench
