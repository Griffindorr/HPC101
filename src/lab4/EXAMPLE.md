<center><span style="font-weight: 500"><font size="20"><bold>Lab4 实验报告</bold></font></span></center>

Lab4 的优化目标是 **数值相对论程序**，该程序本身是一个规模较大、结构复杂的科学计算程序。但是究其根本可以归结为若干个独立的优化任务：在 x86 架构平台上优化 **TwoPunctureABE** 程序，在 A100 平台上优化 **ABEGPU** 程序，在 arm64 架构平台（鲲鹏处理器）上优化 **TwoPunctureABE** 程序，在 arm64 架构平台（鲲鹏处理器）上优化 **ABE** 程序。

## 1. x86 平台上优化 TwoPuncturesABE

x86 平台上端到端的执行流包含 TwoPuncturesABE 初值求解程序以及 ABE 演化程序两个独立的部分，单独执行初值求解程序并利用 vtune（x86 平台上可以用，但是到了 arm64 平台上就很不幸了）来做热点采样，得到如下统计结果（此时只开了一个进程，MPI rank = 1 而且也只有一个主线程，所以统计结果非常清晰）：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/Thomas.png)

当前 ThomasAlgorithm 算法占用约 $25\%$ 的时间开销，该函数是一个三对角矩阵求解器（本身已经是 $O(n)$ 的复杂度不能再降）。观察到程序单词调用就会做 $4$ 次动态内存分配和 $4$ 次动态内存释放，所以我加了计时器去关注一下这部分动态内存分配时间开销：

```cpp
void TwoPunctures::ThomasAlgorithm( ){
		double *l, *u, *d, *y;
    double thomas_total_t0 = g_thomas_alloc_profile.enabled ? twop_now_sec( ) : 0.0;
    double thomas_alloc_t0 = g_thomas_alloc_profile.enabled ? twop_now_sec( ) : 0.0;
    l = new double[ N - 1 ];	u = new double[ N - 1 ];
    d = new double[ N ];			y = new double[ N ];
    g_thomas_alloc_profile.alloc_sec += twop_now_sec( ) - thomas_alloc_t0;

    double thomas_free_t0 = g_thomas_alloc_profile.enabled ? twop_now_sec( ) : 0.0;
    delete[	] l;	delete[ ] u;	delete[ ] d;	delete[ ] y;
    g_thomas_alloc_profile.free_sec += twop_now_sec( ) - thomas_free_t0;
    g_thomas_alloc_profile.total_sec += twop_now_sec( ) - thomas_total_t0;
    g_thomas_alloc_profile.calls += 1;
}
```

测试结果如下：

```bash
ThomasAlgorithm allocation profile
  calls              = 86923200				# Thomas 算法调用超过 8 千万次
  total time         = 88.5568 s			# Thomas 算法总执行时间约为 88s
  alloc+free time    = 8.11378 s
  alloc+free / total = 9.16223 %
```

此外，Thomas 算法的外层还有 LineRelax_be 和 LineRelax_al 两个算法，后面两个算法每次被调用时又会各自做 $5$ 次动态内存分配和释放，这样就会产生更多的无价值 alloc 开销和 free 开销。使用栈数组代替这些大量的动态内存分配和释放可以降低这些开销。这一步优化之后的结果如下，Elapsed Time 降低大约 $8$ 个单位时间（$292.7 \longrightarrow 284.7$）：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/Thomas_v1.png)

然后由于 Thomas 算法的调用次数过于惊人，沿着调用链向上回溯发现大量的无意义的重复调用，发现优化前的调用关系：

```shell
TwoPunctures::Solve( )
 └─ Newton( )                        			# 外层迭代，求 v 使 F( v ) = 0
     └─ bicgstab()                  			# 内层解线性系统 J·dv = F（ BiCGSTAB ）
         ├─ SetMatrix_JFD( )         			# 组装稀疏 Jacobian J
         └─ relax()                				# 行松弛预条件器 M 逆（ 每迭代调用 NRELAX=200 次 ）
             └─ LineRelax_be( ) / LineRelax_al( )   	# 解单条线的三对角系统
                 └─ ThomasAlgorithm( )              	# 三对角直接求解
                     ├─ 组装系数 diag/e/f          		 # 每线解重新扫描
                     ├─ LU 分解							       		# 每线解重新分解
                     └─ 前代/回代
```

每 bicgstab 迭代内 `relax` 被调用 400 次（ph、sh 各 200 次），但 `SetMatrix_JFD` 只组装一次 Jacobian。因此每条线的三对角系数在整个迭代内不变，而初始代码每次线性求解（单次完整求解共 $8692$ 万次）都重新组装、重新 LU 分解、重新分配内存——这是在做大量的无意义的计算。求解线性系统 $A\,x = b$ 时，系数矩阵  `diag/e/f`（三对角矩阵），描述这条线内部的点怎么互相影响，它只由 Jacobian `JFD` 决定，所以每次 `SetMatrix_JFD` 之后只需要做一次 LU 分解即可；而右端项 $b$ 是等号右边已知的量，即残差 $rhs$ 减去线外邻居（当前已知）的贡献 $Σ JFD·dv[col]$，这个是每次动态变化的，所以每次必须要算。因此，我在单词 `bicgstab( )` 迭代内提前通过 `BuildLineFactors( )` 提前做好 LU 分解并存住，后面大量的 `relax( )` 可以直接使用。 

优化后的调用关系：

```shell
TwoPunctures::Solve( )
 └─ Newton( )                        # 外层迭代，求 v 使 F(v)=0
     └─ bicgstab( )                  # 内层解线性系统 J·dv = F（BiCGSTAB）
         ├─ SetMatrix_JFD( )         # 组装稀疏 Jacobian J（CSR，每行≤19列）
         ├─ BuildLineFactors( )      #【新增】预分解全部线的 LU 因子，存 lu_be_*/lu_al_*
         └─ relax()                	 # 行松弛预条件器 M 逆（ 每迭代调用 NRELAX=200 次 ）
             └─ LineRelax_be( ) / LineRelax_al( )   	# 解单条线的三对角系统
                 ├─ 残差组装 b（只累加非三对角项）
                 └─ 前代/回代（用缓存的 LU 因子，只包含乘法）
```

其中，`BuildLineFactors( )` 的计算过程和先前的 Thomas 算法中 LU 分解的部分是几乎一致的，在此不赘述，测试结果：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/Thomas_v2.png)

加速了大约 $40$ 个单位时间（$284.7 \longrightarrow 241.5$）。此时热点函数已经转变为 `LineRelax_be( )` 以及 `LineRelax_al` 两个函数，阅读源代码发现两个函数的调用者都是 `relax( )` 函数，而且调用过程高度串行化（通过大量的 `for` 循环来做）：

```cpp
void TwoPunctures::relax( )
  for (k = 0; k < n3; k = k + 2)																/* k 偶数遍历 */
    for (n = 0; n < N_PlaneRelax; n++)								
      for (i = 2; i < n1; i = i + 2)	LineRelax_be( );					/* i 偶数遍历 loop0 */
      for (i = 1; i < n1; i = i + 2) 	LineRelax_be( );					/* i 奇数遍历 loop1 */
      for (j = 1; j < n2; j = j + 2)	LineRelax_al( );					/* j 奇数遍历 */	
      for (j = 0; j < n2; j = j + 2)	LineRelax_al( );					/* j 偶数遍历 */
  for (k = 1; k < n3; k = k + 2)																/* k 奇数遍历 */
    for (n = 0; n < N_PlaneRelax; n++)
      for (i = 0; i < n1; i = i + 2)	LineRelax_be( );					/* i 偶数遍历 */
      for (i = 1; i < n1; i = i + 2)	LineRelax_be( );					/* i 奇数遍历 */
      for (j = 1; j < n2; j = j + 2)	LineRelax_al( );					/* j 奇数遍历 */
      for (j = 0; j < n2; j = j + 2)	LineRelax_al( );					/* j 偶数遍历 */
```

这里外层循环是 k 维，内层循环是 i 维和 j 维。由于 `LineRelax_be( )` 和 `LineRelax_al( )` 是求解三对角矩阵，所以固定 $k=k_{0}$ 之后，对于 $i = i_{0}$ 所表示的直线，其只需要读取 $i=i_{0}\pm1$ 这两条直线的数据即可，即偶数线作为主对角线的求解器只需要依赖相邻的两条奇数线的数据。所以 loop0 依赖上一轮 `relax( )` 的 loop1，而 loop1 依赖本轮 `relax( )` 的 loop0，loop0 和 loop1 内部本身不存在数据依赖，本身 `LineRelax_be( )` 和 `LineRelax_al( )` 的计算量也足够大，所以适合设置 OPENMP 并行区做线程级并行：

```cpp
#pragma omp parallel default(none)
	shared(dv, nvar, n1, n2, n3, rhs, ncols, cols, JFD)
  {		for (int n = 0; n < N_PlaneRelax; n++)
      {
#pragma omp for collapse(2) schedule(static)
        for (int k = 0; k < n3; k += 2)	for (int i = 2; i < n1; i += 2)	LineRelax_be( );
#pragma omp for collapse(2) schedule(static)
        for (int k = 0; k < n3; k += 2)	for (int i = 1; i < n1; i += 2)	LineRelax_be( );
#pragma omp for collapse(2) schedule(static)
        for (int k = 0; k < n3; k += 2)	for (int j = 1; j < n2; j += 2)	LineRelax_al( );
#pragma omp for collapse(2) schedule(static)
        for (int k = 0; k < n3; k += 2)	for (int j = 0; j < n2; j += 2)	LineRelax_al( );
      }
  }
```

由于集群 lab4g10 分区最多开 $16$ 个线程，这里我测试 OPENMP 分别开 $\{1,2,4,8,16\}$ 线程时的时间开销，基本数据如下（测试正确性都得到满足，不存在精度妥协）：

```bash
== wall time summary
OMP_NUM_THREADS=1 : 261.686 s				speedup: 1.00x
OMP_NUM_THREADS=2 : 183.605 s				speedup: 1.43x
OMP_NUM_THREADS=4 : 145.036 s				speedup: 1.80x
OMP_NUM_THREADS=8 : 123.351 s				speedup: 2.12x
OMP_NUM_THREADS=16 : 122.670 s			speedup: 2.13x}
```

这里大约在 $8$ 或者 $16$ 个线程时达到上限，加速比上限大约为 $2.12$ 左右。从 Amdahl's Law 的角度来说，这里并行化的提升已经非常好，因为根据 vtune 的 Top-down Tree 的记录，`relax( )` 函数函数的总开销占 TwoPunctureABE  时间的约 $60\%$：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/relax_v1.png)

所以理想的加速比为 $s=\frac{1}{0.4+\frac{0.6}{16}}=2.29$，这里达到的 $2.13$ 左右的加速比已经接近上限。重新测量 TwoPunctureABE 的时间，产生较大幅度的加速（$240.9 \longrightarrow 120.4$）：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/relax_v1.png)

当前 `LineRelax_al` 函数以及 `LineRelax_be` 函数如果进一步优化，由于并行的线程数为 $16$，实际的加速效果不会很显著。因此我试着关注一下其他的热点函数，即 `__cos_fma` 函数，该函数的调用者如下，包含切比雪夫变换函数 `chebft_Zeros( )` 以及傅立叶变换函数 `fourft( )`，分析一下在这两个变换中 `__cos_fma` 的调用情况：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/cos_fma_v0.png)

`chebft_Zeros`（Chebyshev 变换，规模 n=50）和 `fourft`（傅里叶变换，规模 N=26）都是 $O(n^2)$ 复杂度的谱变换，内层循环反复调 libm `cos( )`/`sin( )`。但变换矩阵本身数值只依赖索引和固定 n，与数据无关——每次迭代重新调三角函数是纯浪费。

```cpp
void TwoPunctures::chebft_Zeros(double u[], int n, int inv ){
  	for( j = 0; j < n; j++ ){
        sum = 0.0;
        for (k = 0; k < n; k++)
          	sum += u[k] * cos(Pion * j * (k + 0.5));   				/* 每次迭代都调 cos( ) */
        c[j] = fac * sum * isignum;
        isignum = -isignum;
    }
}
```

```cpp
void TwoPunctures::fourft(double *u, int N, int inv){
    for (l = 0; l <= M; l++) 
    	for (k = 0; k < N; k++){
            x = x1 * k;
            a[l] += fac * u[k] * cos(x);        	/* 每次迭代都调 cos( ) */
            if (l > 0 && l < M)
              	b[l] += fac * u[k] * sin(x);      	/* 每次迭代都调 sin( ) */
        }
}
```

但是每次调用的 `cos( )` 函数或者 `sin( )` 函数的自变量只与矩阵规模 N 以及坐标有关，完全可以在第一次计算并且缓存好，后面再使用时直接读取即可，避免每次对相同的数值计算 `cos( )` 或者 `sin( )` 的数值，降低大量的重复计算。这里使用函数内静态变量（但是是动态申请的内存作为存储区），如果当前输入 N 与先前缓存的 cache_N 一致，就不需要重新计算，直接读数据：

```cpp
static double *cos_mat = 0; /* cos_mat[ l*N + k ] = cos( Pi_fac*l*k ), l in [ 0, M ] */
static double *sin_mat = 0; /* sin_mat[ l*N + k ] = sin( Pi_fac*l*k ), l in [ 0, M ] */
if (N != cached_N){									/* 需要重新计算 */
    for( l = 0; l <= M; l++ ){
        double x1 = Pi_fac * l;
        for( k = 0; k < N; k++ ){
            double x = x1 * k;
            cos_mat[ l * N + k ] = cos( x );		/* 缓存 */
            sin_mat[ l * N + k ] = sin( x );		/* 缓存 */
        }
    }
	cached_N = N;
}
```

经过优化后重新测试 TwoPunctureABE 的计算时间，收益很明显（$120.4 \rightarrow 41.0$），三角函数目前已经不是热点：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/cos_fma_v1.png)

现在由于 `relax( )` 函数中多线程的使用，vtune 的热点图已经不太清晰。当前程序唯一的并行区域是 `relax( )`，我通过手动插桩测试其他串行区域的时间开销，目前 TwoPunctureABE 总的时间约为 $41.0$ 个单位时间，主要的三大热点如下：

|           阶段            |        时间        |         说明         |
| :-----------------------: | :----------------: | :------------------: |
| `relax( )` 函数（并行区） | $19.43$ 个单位时间 |   已经做线程级并行   |
|   `J_times_dv( )` 函数    | $9.70$ 个单位时间  | 包含谱变换和逐点求值 |
|  `SetMatrix_JFD( )` 函数  | $8.90$ 个单位时间  |  构建 Jacobian 矩阵  |

其中 `J_times_dv( )` 函数还没有并行化，原始的基本结构如下：

```cpp
void TwoPunctures::J_times_dv( ){
  Derivatives_AB3( );									/* 谱变换 */
  for (i = 0; i < n1; i++)								/* 逐点求值 */
    for (j = 0; j < n2; j++)
      for (k = 0; k < n3; k++)
        for (ivar = 0; ivar < nvar; ivar++){ ... }
}
```

第二阶段逐点求值各个点其实相互独立，因此可以较为简洁地直接做线程级并行化：

```cpp
void TwoPunctures::J_times_dv( ){
  Derivatives_AB3(nvar, n1, n2, n3, dv);
#pragma omp parallel for schedule(static) private( i, j, k, ivar, indx )		/* 线程级并行化 */
  for (i = 0; i < n1; i++)
    for (j = 0; j < n2; j++)
      for (k = 0; k < n3; k++)
        for (ivar = 0; ivar < nvar; ivar++)
            indx = Index(ivar, i, j, k, nvar, n1, n2, n3);
}
```

但是只做逐点求值的并行化只会带来不大的收益（$9.7 \rightarrow 8.0$），所以 `J_times_dv( )` 里的谱变换（`Derivatives_AB3( )`），占 `J_times_dv( )` 的大部分时间。 `Derivative_AB3( )` 函数在 $i$，$j$，$k$​ 三个空间方向上做求导计算，把数据用切比雪夫 Chebyshev 基或者傅立叶 Fourier 基展开求导，`Derivatives_AB3` 是三个方向独立的 pass：

$1.$ A 方向：for k { for j { 对 i 做 Chebyshev 变换 } }——每根 ( j, k ) 行独立

$2.$ B 方向：for k { for i { 对 j 做 Chebyshev 变换 } }——每根 ( i, k ) 行独立，但会依赖前一轮 A 方向 pass 的结果

$3.$ phi 方向：for i { for j { 对 k 做 Fourier 变换 } }——每根 ( i, j ) 行独立，但会依赖前一轮 B 方向 pass 的结果

每个 pass 内部各点是完全独立的，与 `relax( )` 的阶段划分一致，可以用阶段内 openmp 并行 + 阶段间 barrier 的方式来组织：

```cpp
void TwoPunctures::Derivatives_AB3( ){
#pragma omp parallel shared(nvar, n1, n2, n3, N, v) private(i, j, k)
    {
#pragma omp for collapse(2) schedule(static)			/* A 方向 pass */
    for (ivar = 0; ivar < nvar; ivar++)
      for (k = 0; k < n3; k++)
        for (j = 0; j < n2; j++)
          for (i = 0; i < n1; i++)
#pragma omp for collapse(2) schedule(static)			/* B 方向 pass */
    for (ivar = 0; ivar < nvar; ivar++)
      for (k = 0; k < n3; k++)
        for (i = 0; i < n1; i++)
          for (j = 0; j < n2; j++)
#pragma omp for collapse(2) schedule(static)			/* C 方向 pass */
    for (ivar = 0; ivar < nvar; ivar++)
      for (i = 0; i < n1; i++)
        for (j = 0; j < n2; j++)
          for (k = 0; k < n3; k++)
	}
}
```

对 `Derivative_AB3( )` 做并行化之后来测试 TwoPunctureABE 的时间，产生较为明显的收益（$41.0 \rightarrow 30.1$），而且通过打桩得到 `J_times_dv( )` 的加速（$7.98 \longrightarrow 2.41$）也比较明显：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/J_times_dv_v1.png)

下一个占比较大的串行瓶颈就是 `SetMatrix_JFD( )` 函数，用于构建 bicgstab 的 Jacobian 矩阵 **JFD**. **JFD** 是 $65000 \times 19$ 的稀疏矩阵，每行对应一个网格点的残差方程，非零列是该点的 3×3×3 邻居（27 个候选中 19 个非零）。构建该矩阵的方式是，对每个 column 设 `dv.d0[ column ]=1`，调用 `JFD_times_dv( )` 数值微分求偏导：

```cpp
void TwoPunctures::SetMatrix_JFD( ){
  for( i = 0; i < n1; i++ )									/* n1 = 50 */
    for( j = 0; j < n2; j++ )								/* n2 = 50 */
      for( k = 0; k < n3; k++ )								/* n3 = 26 */
        for( ivar = 0; ivar < nvar; ivar++ )				/* nvar = 1，唯一未知函数 */
          for( i1 = i - 1; i1 <= i + 1; i1++ )				/* 相邻的三个邻居 */
            for( j1 = j - 1; j1 <= j + 1; j1++ )			/* 相邻的三个邻居 */
              for( k1 = k - 1; k1 <= k + 1; k1++ )			/* 相邻的三个邻居 */
                JFD_times_dv( );
}
```

但是这个不能无脑地做并行化，因为每一个点需要计算其周围 $27$ 个邻居的信息，然后写入对应邻居的 slot. 假设 C 同时是点 A 和点 B 的邻居，那么点 A 和点 B 求解 `JFD_times_dv( )` 之后可能同时向 C 的 slot 中写入，产生写冲突。我使用两阶段处理。**第一阶段是并行的**，每个 column 独立评估周围 27 个 stencil 的贡献，每线程私有 `dv.d0` 副本（`JFD_times_dv` 只读 `dv.d0`）和私有 `values`，结果存临时数组 `tmp_row` / `tmp_val`（每 column 最多 27 项），不写到共享的矩阵中。**第二阶段是串行的**，按原 $(i,j,k)$ 的 column 字典序遍历，把临时值追加进 `cols[row]` / `Matrix[row]` 并且递增 `ncols[row]++`。由于阶段 $2$ 严格保持串行计算、串行访问，可避免写冲突：

```cpp
void TwoPunctures::SetMatrix_JFD( ){
#pragma omp parallel default(none)                            \
    shared( nvar, n1, n2, n3, u, ntotal, tmp_row, tmp_val, tmp_cnt, \
            N1, N2, N3) private(i, j, k, ivar, column, row, i1, j1, k1, \
            i_0, i_1, j_0, j_1, k_0, k_1, ivar1, mcol )
  	{
    /* Phase 1: evaluate all stencil contributions in parallel. */
#pragma omp for schedule(static)
        for (i = 0; i < n1; i++)
            for (j = 0; j < n2; j++)
                for (k = 0; k < n3; k++)
                    for (i1 = i - 1; i1 <= i + 1; i1++)				/* i-1, i, i+1 */
                        for (j1 = j - 1; j1 <= j + 1; j1++)			/* j-1, j, j+1 */
                            for (k1 = k - 1; k1 <= k + 1; k1++)		/* k-1, k, k+1 */
                                JFD_times_dv( );
    }
	/* Phase 2: serial fill preserving the original (i,j,k) column order. */
    for (i = 0; i < n1; i++)
   		for (j = 0; j < n2; j++)
        	for (k = 0; k < n3; k++)
            	for (ivar = 0; ivar < nvar; ivar++)
                	for (mcol = 0; mcol < tmp_cnt[column]; mcol++)
}
```

完成 `SetMatrix_JFD( )` 函数并行化之后来测试 TwoPunctureABE 的时间，产生较为明显的收益（$30.1 \longrightarrow 22.2$），而且通过打桩得到 `SetMatrix_JFD( )` 的加速也很显著（$9.31 \longrightarrow 1.50$）：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/SetMatrix_JFD_v1.png)

做完这次优化之后，我重新看一下 TwoPunctureABE 各阶段的 wall time，判断新的热点。这个时候 vtune 里面统计的 CPU time 具有误导性，很难判断端到端时间的组成和各部分占比。我在程序中打了一些桩，测试得到如下的结果：

|           阶段            | 时间（总计 $21.76$） |  占比  |                 说明                  |
| :-----------------------: | :------------------: | :----: | :-----------------------------------: |
|     `relax( )` 并行区     |  $16.88$ 个单位时间  | $77\%$ |        线松弛预条件子（主导）         |
| `SetMatrix_JFD( )` 并行区 |  $1.51$ 个单位时间   | $7\%$  | 已经并行（$9.3 \longrightarrow 1.5$） |
|  `J_times_dv( )` 并行区   |  $2.26$ 个单位时间   | $10\%$ | 已经并行（$9.7 \longrightarrow 2.3$） |
|    `F_of_v( )` 并行区     |  $0.24$ 个单位时间   | $1\%$  |       仍然串行（但是占比很小）        |

现在 `relax( )` 经过并行化之后仍然是热点区域，查看一下其调用者 `bicgstab( )` 函数：

```cpp
int TwoPunctures::bicgstab( ){
	for (int j = 0; j < NRELAX; j++) 	relax( );	/* solves JFD*ph = p by relaxation*/
    for (int j = 0; j < NRELAX; j++) 	relax( );	/* solves JFD*sh = s by relaxation*/
```

每次 `bicgstab( )` 会调用 `relax( )` 函数 $2 \times \mathrm{NRELAX}$ 次，NRELAX 作为预条件子的内层迭代数，当前的数值为 200. 理论上可以降低 NRELAX 从而减小内层迭代的次数，但是为了保证最终收敛，`bicgstab( )` 函数的调用次数会相应地增加。这意味着单方面降低 BRELAX 不一定会减少 TwoPunctureABE 整体的时间，而且精度和准确性也需要做检查，通过测试脚本测试结果如下：

| NRELAX |     wall time      | bicgstab 迭代次数 |    残差范数 $|F|$    | 收敛？ |
| :----: | :----------------: | :---------------: | :------------------: | :----: |
|  200   | $21.25$ 个单位时间 |        96         | $2.87\times10^{-13}$ |  收敛  |
|  100   | $18.60$ 个单位时间 |        141        | $3.85\times10^{-13}$ |  收敛  |
|   50   | $16.40$ 个单位时间 |        194        | $4.16\times10^{-13}$ |  收敛  |
|   25   | $15.02$ 个单位时间 |        255        | $4.16\times10^{-13}$ |  收敛  |
|   10   | $16.95$ 个单位时间 |        400        | $4.33\times10^{-13}$ |  收敛  |
|   5    | $21.30$ 个单位时间 |        599        | $2.23\times10^{-12}$ |  收敛  |

可以看到 sweet point 是 $\mathrm{NRELAX}=25$ 的时候，这个取值下，TwoPunctureABE 的端到端时间最小，而且精度和正确性可保证。修改之后，可以看到 TwoPunctureABE 的端到端时间得到提升（$22.2 \longrightarrow 16.1$）。打桩之后可以确认，`relax( )` 的 wall time 从 $16.9$ 个单位时间降到 $6.1$ 个单位时间，但是由于 `bicgstab( )` 迭代次数增加（$96 \longrightarrow 255$），`J_times_dv( )` 被调用次数增加（被调用次数大约翻了三倍），wal time 从 $2.3$ 个单位时间翻了约三倍增至 $6.4$ 个单位时间。二者综合来看减少大约 $6.0$ 个单位时间：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/NRELAX_v1.png)

现在观察一下 TwoPunctureABE 的端到端的 wall time 组成，由于 `bicgstab( )` 迭代次数翻了三倍，`J_times_dv( )` 重新成为热点函数。`J_times_dv( )` 函数本身是由谱变换 `Derivatives_AB3( )` 和逐点计算两部分组成，前者开销为 $1.94$ 个单位时间，后者开销为 $4.38$ 个单位时间：

| 组成阶段                | wall time        |
| ----------------------- | ---------------- |
| `relax( )` 并行区       | $6.1$ 个单位时间 |
| `J_times_dv( )` 函数    | $6.4$ 个单位时间 |
| `SetMatrix_JFD( )` 函数 | $1.5$ 个单位时间 |
| `F_of_v( )` 函数        | $0.23$ 个单位间  |

尝试在逐点计算中发现可优化的点，`J_times_dv( )` 每次调用里逐点循环对每个格点（$65000$ 个）都执行 $21$ 次动态分配/释放，每格点 $21$ 次 malloc/free，浪费严重（而且前面 vtune 的 profile 中 malloc/free 总计占 $19.0\%$ 的 CPU time）。这里把其中重复做的 `values`/`dU`/`U` 的分配提升到 OpenMP 线程级，每个线程分配一次，在线程访问的所有格点间复用：

```cpp
#pragma omp parallel shared(dv, Jdv, u, nvar, n1, n2, n3) private(i, j, k, ivar, indx)
	{
        double *values = dvector(0, nvar - 1);   		/* 每个线程分配一次 */
        derivs dU, U;
        allocate_derivs(&dU, nvar);
        allocate_derivs(&U, nvar);
#pragma omp for schedule(static)
        for (i = 0; i < n1; i++)
        	for (j = 0; j < n2; j++)
        		for (k = 0; k < n3; k++)	... 			/* 计算 */

        free_dvector(values, 0, nvar - 1);       		/* 线程结束释放一次 */
        free_derivs(&dU, nvar);
        free_derivs(&U, nvar);
    }
```

这样，原来需要做 $65000 \times 510$ 约为 $3300$ 万次动态内存分配和释放，现在只需要做 $65000 \times 16$ 约 $8000$ 次动态内存分配和释放即可。优化后通过打桩可以得到 `J_times_dv( )` 从 $6.32$ 个单位时间降至 $3.51$ 个单位时间，其中谱变换阶段几乎不变，逐点计算阶段从 $4.38$ 个单位时间降至 $1.64$ 个单位时间。而 TwoPunctureABE 整体的端到端时间也有改善（$16.1 \longrightarrow 13.7$）：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/J_times_dv_v2.png)

目前的 TwoPunctureABE 已经从 $292.7$ 个单位时间优化到 $13.7$ 个单位时间，约为 $21.4$ 倍的加速比。

---------



## 2. A100 平台上优化 ABEGPU 程序

首先跑一下 baseline（演化的时间设置为 20.0），通过 Nsight Systems 发现 **绝对热点** 就是 `res_kernel( )`. 当前为了提高测试效率，将演化的时间设置为 $2.0$，在 ABEGPU 完全没有优化的情况下，测试得到单步演化的时间开销大约为 $17.53$ 秒，通过 Nsight Systems 的报告可以再次验证一下当前的绝对热点是 `rhs_kernel( )`，在单步演化里的各个函数的时间占比为：

|            核函数            | 两步演化的时间开销 |   占比   | 被调用次数 |      单次开销      |
| :--------------------------: | :----------------: | :------: | :--------: | :----------------: |
|       `rhs_kernel( )`        | $13.47~\mathrm{s}$ | $77.5\%$ |   $401$    | $33.6~\mathrm{ms}$ |
|     `prolong3_kernel( )`     | $2.09~\mathrm{s}$  | $12.0\%$ |  $11640$   | $180~\mathrm{us}$  |
|    `restirct3_kernel( )`     | $0.64~\mathrm{s}$  | $3.7\%$  |   $2328$   | $273~\mathrm{us}$  |
|  `global_interp_kernel( )`   | $0.49~\mathrm{s}$  | $2.8\%$  |   $344$    | $1.41~\mathrm{ms}$ |
| `Sommerfeld_rout_kernel( )`  | $0.35~\mathrm{s}$  | $2.0\%$  |   $9312$   |  $37~\mathrm{us}$  |
| `rungekutta4_rout_kernel( )` | $0.21~\mathrm{s}$  | $1.2\%$  |   $9408$   |  $22~\mathrm{us}$  |

可以看到 `rhs_kernel( )` 确实是绝对热点。该核函数虽然不是被调用次数最多的，但是单次开销非常高，属于高负载的核函数，于是又定点通过 Nsight Compute 来单独对 `rhs_kernel( )` 来做 profiling，得到若干数据。首先应该对 NVIDIA A100 架构做一些分析：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/A100_topo.png)

这里的架构图中一张 A100 卡共有 128 个流式多处理器（Stream Multi-processor），单个 SM 内部会有 L1 Cache、Shared Memory 等高级存储，64 个流式多处理器会共享一个 L2 Cache，目前不存在全局共享的 L3 Cache，通过 Memory Contorller 与 HBM 做数据传输。这个是基本的宏观硬件拓扑。单个 SM 内部存在一些微架构数据，单个 SM 可以同时驻留 64 个 warp，存在 4 个 warp scheduler，因此单个 warp sheduler 可以做最多 16 个 warp 的并发执行。下面是 Nsight Computer 对 `rhs_kernel( )` 的 profiling：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/launch_statistics.png)

当前 launch 的 kernel 有 125 个 blocks，单个 block 有 256 个 thread（也就是 8 个 warp，因为单个 warp 固定由 32 个 thread 组成），因此该 kernel 总计 32000 个线程。由于这个 kernel 非常大，单个线程需要 250 个寄存器，没有申请静态共享内存或者动态共享内存。而且临时数据也已经溢出寄存器的空间，spill 到 local memory 中，额外需要 1120 字节 stack size 的内存空间。然后，看一下 occupancy 的数据：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/occupancy_statistics.png)

由于单个线程需要 250 个寄存器，则整个 block 的所需要的寄存器数量为 $250 \times 256 = 64000$，而单个 SM 的寄存器数量上限是 65536 个，因此单个 SM 的寄存器数量作为瓶颈会限制只有 1 个 block 驻留。单个 block 有 8 个 warp，单个 SM 的 warp 数上限为 64，所以理论的 occupancy 占比为 $12.5\%$，而实际的利用率只会更低。可以看到与此同时，Shared Memory 或者 warps 数对单个 SM 的 blocks 数的限制为 8，所以寄存器数量是 `rhs_kernel( )` 当前的一个非常严重的瓶颈。由于一共有 125 个 blocks，单个 A100 MIG 实例有 14 个流式多处理器，某一个时刻单个 SM 上只能驻留一个 block，所以 Waves Per SM 为 $\lceil \frac{125}{14} \rceil = 9$，也就是目前的低效的执行一共会有 9 波。最后看一下吞吐量的统计数据：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/throughput_statistics.png)

可以看到单个 SM 的计算利用率仅为 $22.30\%$，内存的吞吐量占极限带宽仅为 $28.21\%$（由于 DRAM 的吞吐比率是 $4.04\%$，所以可以推知这里的内存吞吐主要是由 L1 Cache 和 L2 Cache 贡献的）。L1 Cache 的吞吐占极限带宽的 $17.91\%$，L2 Cache 的吞吐占极限带宽的 $28.21\%$. 因此，可以得到如下的逻辑链：kernel 体量过于庞大 $\longrightarrow$ 单个线程占用的逻辑寄存器过多 $\longrightarrow$ 单个 SM 的 warp occupancy 仅为 $12.5\%$  $\longrightarrow$ 访存延迟无法 hide $\longrightarrow$ SM 的计算硬件并未充分运用 $\longrightarrow$ 性能太差。因此，首先关键是通过拆解 `rhs_kernel( )` 单个核函数，拆分为若干个较小的核函数来 launch，尽管会引入多次 launch 的开销，但是单个 kernel 的效率可能会大幅提升。

首先当前单个线程占用的寄存器数量是 $250$ 个，但是这个对编译器不加约束的结果，寄存器瓶颈会导致单个 SM 上只会驻留一个 block，造成仅仅 $12.5\%$ 的 occupancy. 其实可以通过 `__launch_bounds__( 256, 2 )` 可以明确限制编译器，强制单个线程的寄存器占用量 $\le 128$，使得单个 SM 上可以驻留 $2$ 个 block. 通过 Nsight Compute 看一下结果：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/rhs_kernel_v1_througput.png)

看 Nsight Compute 的 profiling 数据，计算吞吐大幅增长（$22.30\% \longrightarrow 33.66\%$），存储吞吐也大幅增长（$28.21\% \longrightarrow 44.35\%$）（其他如 L1 Cache 以及 L2 Cache 的吞吐也都增长），内存吞吐甚至增长至原来的五倍（$4.04\% \longrightarrow 20.05\%$）。目前 `rhs_kernel( )` 核函数的执行时间从 $7.35~\mathrm{ms}$ 降低到 $4.61~\mathrm{ms}$. 其实根本原因就是通过压缩寄存器增大一个 SM 上驻留的 warp，提高并发度。下面可以看一下改后 Nsight Compute occupancy 的数据：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/rhs_kernel_v1_occupancy.png)

可见，确实 Block Limit Registers 从 $1$ 个提升到了 $2$ 个。当前实际实现的 occupancy 相较于原来的 $11.05\%$ 提升到了 $21.57\%$，非常显著，可见通过 `__launch_bounds( 256, 2 )` 确实可以硬性压缩寄存器占用来提高并发度。但是这是有限度的，假设我强行设置 `__launch_bounds( 256, 3 )` 会直接编译报错：

```shell
nvlink error   : entry function 'rhs_kernel' with max regcount of 80
                 calls function 'd_lopsided_point' with regcount of 85
nvlink error   : entry function 'rhs_kernel' with max regcount of 80
                 calls function 'd_fdderivs_point' with regcount of 104
gmake[2]: *** [CMakeFiles/ABEGPU.dir/build.make:799: CMakeFiles/ABEGPU.dir/cmake_device_link.o] Error 255
gmake[1]: *** [CMakeFiles/Makefile2:157: CMakeFiles/ABEGPU.dir/all] Error 2
gmake: *** [Makefile:91: all] Error 2
```

报错显示虽然我们要求 `rhs_kernel( )` 的寄存器数量上限是 $80$ 个，但是其调用的 `d_lopsided_point( )` 函数需要 $85$ 个寄存器，其调用的 `d_fdderivs_point( )` 函数需要 $104$ 个寄存器，因此无法压缩，直接编译报错。 因此停止继续压缩。

由于我没有修改核函数的逻辑，所以单个线程对寄存器的需求其实不变，压缩寄存器的数量会导致溢出的数据 spill 到 local memory 里，也就是体现在 Nsight Compute 的 stack size 的增长。但是由于 occupancy 的提升，单个 SM 由于同时驻留 $2$ 个 block，也就是 $16$ 个 warp，这样单个 warp schuduler 就可以为 $4$ 个 warp 发射指令，warp 的并发度更高，hide latency 的效果更好。两步演化的效果确实变得更好，提升比较大（单步 $17.48 \longrightarrow 12.60$）：

|    配置    | REG 占用 | STACK 占用  |      单步时间      |     演化总时间     |
| :--------: | :------: | :---------: | :----------------: | :----------------: |
| $(256，1)$ |  $250$   | $1112$ 字节 | $17.48~\mathrm{s}$ | $34.96~\mathrm{s}$ |
| $(256，2)$ |  $128$   | $1480$ 字节 | $12.60~\mathrm{s}$ | $25.19~\mathrm{s}$ |

然后我认为，`rhs_kernel( )` 整体由于逻辑过于复杂，寄存器占用始终作为其瓶颈而存在，大大降低其计算吞吐和内存吞吐。因此，很重要的一点是通过拆分 `rhs_kernel( )` 这个很大的核函数来降低各自的寄存器占用。由于核函数的拆分会造成更多的全局内存的读写，所以也是有代价的。综合来看，拆分力度不可过细，要遵循实际的数学计算语义进行分块。平流（lopsided upwind）和耗散（Kreiss-Oliger）只是对 RHS 做后处理修正，它们不依赖 Ricci/导数阶段的中间量，却挤占了同一个 kernel 的寄存器预算；约束计算（co==0 才做）也只依赖 RHS 结果。把这些拆成独立 kernel，每个 kernel 的活少、寄存器需求低，编译器能更自由地优化各自的循环和 occupancy。这里拆分的逻辑如下：

|               核函数                |                           计算内容                           |                             输入                             |                         输出                         |
| :---------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------------------------------------------: |
|           `rhs_kernel( )`           | 一阶导数、二阶导数、逆度规 、Christoffel、Ricci、24 个 RHS 组装、规范量 RHS |          chi/trK/gij/Aij/Gam/Lap/beta/dtSf、Sij 等           | _rhs 24 个；co==0 时另写 Gamxxx..Gamzzz、Rxx ... Rzz |
|      `rhs_advection_kernel( )`      |                     lopsided 上风格平流                      |         gxx ... gzz 状态、gxx_rhs ... gzz_rhs、beta          |               gxx_rhs..gzz_rhs 读改写                |
| `rhs_advection_curvature_kernel( )` |                  lopsided 平流（曲率+标量）                  |            Axx ... Azz/chi/trK、对应 *_rhs、beta             |     Axx_rhs ... Azz_rhs, chi_rhs, trK_rhs 读改写     |
|   `rhs_advection_gauge_kernel( )`   |                   lopsided 平流（规范量）                    |             Gam/Lap/beta/dtSf、对应 *_rhs、beta              |     Gam_rhs, Lap_rhs, beta_rhs, dtSf_rhs 读改写      |
|     `rhs_dissipation_kernel( )`     |                Kreiss-Oliger 耗散（24 变量）                 |                      24 状态、24 *_rhs                       |                  24 个 *_rhs 读改写                  |
|     `rhs_constraints_kernel( )`     |                           约束残差                           | chi/trK/gij/Aij/rho/S、Rxx ... Rzz、Gamxxx..Gamzzz（全局读回） |                ham_Res, movx/y/z_Res                 |

所以整体的执行流在 `gpu_compute_rhs_bssn_launch( )` 中得以体现：

```cpp
void gpu_compute_rhs_bssn_launch( ){
    rhs_kernel<<< ... >>>( );						/* 组装 RHS 基础项 */
    rhs_advection_kernel<<< ... >>>( );				/* metric 6 变量 */
    rhs_advection_curvature_kernel<<< ... >>>( );	/* Aij 6 + chi/trK */
    rhs_advection_gauge_kernel<<< ... >>>( );		/* Gam/Lap/beta/dtSf */
    rhs_dissipation_kernel<<< ... >>>( );			/* 耗散项累加（+=）*/
    if( co == 0 ){
        rhs_constraints_kernel<<< ... >>>( );		/* 约束残差（仅 predictor）*/
    }
}
```

拆分后测试性能（这里包含前面 `__launch_bounds__( 256, 2 )` 的约束），单步迭代从 $12.60 ~\mathrm{s}$ 提升到 $12.06 ~\mathrm{s}$，提升不算明显，但是只要拆分没有出现明显的性能下降就是成功的，这表明拆分带来的并发度提升可以把 kernel 之间的全局内存读写的开销掩盖掉。后面可以针对这些核函数分别做优化，先用 Nsight Compute 对这些函数做一些 Profiling，数据如下：

|               核函数                |      Duration      | 寄存器占用 | 理论 Occupancy | 实际 Occupancy | Compute% | Memory%  |  DRAM%   |
| :---------------------------------: | :----------------: | :--------: | :------------: | :------------: | :------: | :------: | :------: |
|           `rhs_kernel( )`           | $6.30~\mathrm{ms}$ |   $128$    |     $25\%$     |    $21.5\%$    | $35.9\%$ | $37.4\%$ | $26.4\%$ |
|      `rhs_advection_kernel( )`      | $0.83~\mathrm{ms}$ |    $85$    |     $25\%$     |    $20.6\%$    | $33.1\%$ | $39.8\%$ | $8.0\%$  |
| `rhs_advection_curvature_kernel( )` | $1.11~\mathrm{ms}$ |    $85$    |     $25\%$     |    $20.7\%$    | $33.4\%$ | $39.7\%$ | $7.7\%$  |
|   `rhs_advection_gauge_kernel( )`   | $1.34~\mathrm{ms}$ |    $85$    |     $25\%$     |    $20.7\%$    | $34.2\%$ | $40.5\%$ | $7.5\%$  |
|     `rhs_dissipation_kernel( )`     | $2.08~\mathrm{ms}$ |    $61$    |     $50\%$     |    $42.7\%$    | $51.4\%$ | $56.0\%$ | $36.3\%$ |
|     `rhs_constraints_kernel( )`     | $2.34~\mathrm{ms}$ |   $154$    |    $12.5\%$    |    $10.8\%$    | $20.9\%$ | $35.5\%$ | $8.8\%$  |

可以看到，拆分出来的 `rhs_advection_kernel( )`，`rhs_advection_curvature_kernel( )` 以及 `rhs_advection_gauge_kernel( )` 的寄存器最低只能是 $85$，因为其调用的 `d_lopsided_point( )` 函数至少需要做 $85$ 个寄存器。目前不能做寄存器压缩了，试着做一下其他的无意义计算的削减优化。

三个 `advection( )` 核函数，DRAM 只有 ~8%，说明数据基本在 L1 内复用；compute $33\% - 34\%$ 意味着指令吞吐也不低，而每点几十次边界/对称比较的冗余原因可能就在于此。删掉它们应该可以直接降低指令开销。模板的计算要取邻居值，原代码每次取邻居都经 `d_symmetry_bd_1b` 函数，包含 $3$ 次越界检查，若在对称面外需要翻转索引并乘 `±factor`，再 $3$ 次检查翻转后是否越界。一个网格点大约要执行几十次 `d_symmetry_bd_1b`. 距所有边界/对称面至少 3 格（即 `i∈[3, imax-3]` 那些点）可以称为内部点，网格越大内部点占比越高，比如 `40×40×20` 网格内部占 $\frac{34×34×14}{40×40×20} ≈ 50\%$​，边界检查对于这些内部点而言完全是无意义的开销。一次内部点判定后，直接按列主序索引取邻居、无分支直线计算，但是边界点仍走原逻辑（正确性不变）。

以导数 `d_fderivs_point( )` 计算为例，优化前 —— 每个邻居经 `fh( )`（即 `d_symmetry_bd_1b`，6 次比较 + 可能翻转）取数：

```cpp
const auto fh = [ & ]( int ii, int jj, int kk ){
	return d_symmetry_bd_1b( 2, ex, f, ii+1, jj+1, kk+1, SoA );  	/* 边界检查+翻转 */
};
if( i+2 <= imax && i-2 >= imin && ... ){          					/* 4 阶档 */
	*fx = d12dx * (fh(i-2,j,k) - 8*fh(i-1,j,k) + 8*fh(i+1,j,k) - fh(i+2,j,k));		...
} else if (i+1 <= imax && i-1 >= imin && ...) {   					/* 2 阶档 */
	*fx = d2dx * (-fh(i-1,j,k) + fh(i+1,j,k));										...
}
```

优化之后，在一开始做内部点判定，如果满足条件则直接直接 `f[ idx ± off ]`、固定 4 阶、提前返回。

```cpp
if (i >= 2 && i <= imax-2 && j >= 2 && j <= jmax-2 && k >= 2 && k <= kmax-2) {
    *fx = d12dx * (f[i-2] - 8*f[i-1] + 8*f[i+1] - f[i+2]);  		/* 直接 f[ idx±off ] */
    *fy = d12dy * (...);
    *fz = d12dz * (...);
    return;                                               			/* 算完提前返回 */
}
```

基于上面的思路我修改了如下函数，`d_fderivs_point`, `d_fdderivs_point`, `d_kodis_point` 的内部逻辑，增加了 `d_lopsided_point_cached` 函数，然后测试修改之后的函数，发现单步演化的时间从 $12.06~\mathrm{s}$ 降低到了 $8.60~\mathrm{s}$ 左右，性能提升比较明显。通过 Nsight Compute 看一下类 `advection( )` 的核函数的分析数据：

| Kernel | Duration | Regs/Thread | Theor Occupancy | Achieved Occupancy | Memory% | Compute% |
|:-:|---|---|---|---|---|---|
| `rhs_advection_kernel( )` | $830 → 418~\mathrm{µs}$ | $85 → 74$ | $25\% → 37.5\%$ | $20.6\% → 26.9\%$ | $39.8\% → 26.5\%$ | $33.1\% → 37.7\%$ |
| `rhs_advection_curvature_kernel( )` | $1110 → 554~\mathrm{µs}$ | $85 → 74$ | $25\% → 37.5\%$ | $20.7\% → 27.2\%$ | $39.7\% → 26.0\%$ | $33.4\% → 37.2\%$ |
| `rhs_advection_gauge_kernel( )` | $1340 → 702~\mathrm{µs}$ | $85 → 74$ | $25\% → 37.5\%$ | $20.7\% → 26.7\%$ | $40.5\% → 24.8\%$ | $34.2\% → 35.6\%$ |

目前可以看到通过内部点快速优化，类 `advection( )` 核函数时间开销显著降低，寄存器占用也从 $85$ 降低到 $74$，这意味着单个 SM 上可以同时驻留 $3$ 个 block 了。由此代码的内存部件和计算部件都得到进一步的充分利用。然后我试着进一步压缩寄存器的占用数，即 `__launch_bounds__( 256, 4 )` 来做测试，发现性能可以得到一点提升，occupancy 进一步提高，单步演化时间从 $8.60~\mathrm{s}$ 降低到 $8.44~\mathrm{s}$. 

下面尝试了一下内联函数。虽然内联函数看似没有做实际的代码逻辑的改动，但是这一步优化是作用在寄存器上的。我了解到一个很重要的概念是 **公共子表达式消除**（Common Subexpression Elimination）：如果同一个子表达式在代码里出现多次而且数值不发生改变，那么编译器只会加载并计算一次，之后会把数值放在寄存器里复用，不需要重复计算或者加载。比如：

不内联时每次函数调用都需要：

```cpp
value  = dxx[idx];            							/* 读 dxx[idx] 一次 */
fx     = (dxx[idx-1] - 8*dxx[idx] + ...)/12;   			/* 又读 dxx[idx-1] 一次 */
```

函数内联之后：

```cpp
r1 = dxx[idx-1];   										/* 各邻居只加载一次 */
r0 = dxx[idx];											/* 第一次时会加载 */
r2 = dxx[idx+1];										/* 第一次时会加载 */
value = r0;												/* r0 寄存器数值复用 */
fx    = (r1 - 8*r0 + ...)/12;							/* r1 寄存器数值复用 */
```

如果不内联的话：函数在另一编译单元，编译器看不到内部读了什么，无法合并，同一份数据被反复加载进 L1 Cache. 内联之后：调用展开成直线代码，编译器看到重复邻居访问，合并成一次加载，中间结果存寄存器复用。此外，从统一寄存器分配角度分析，如果不内联：每个被调函数单独占寄存器（`d_fdderivs_point` 需 104 regs）， `__launch_bounds__` 压不动。如果内联：编译器为整个展开体统一分配，可跨调用复用临时值，也就是单个函数的寄存器占用不会作为调用者的寄存器瓶颈，可以配合`__launch_bounds__` 压整个调用图。

对两个函数做了内联尝试：

| 函数                      | 源文件            | 改后文件        | 标记                            |
| ------------------------- | ----------------- | --------------- | ------------------------------- |
| `d_fderivs_point( )` 函数 | `diff_new_gpu.cu` | `derivatives.h` | __device__ **\__forceinline__** |
| `d_fdderivs( )` 函数      | `diff_new_gpu.cu` | `derivatives.h` | __device__ **\__forceinline__** |

然后进行性能测试，发现单步演化时长从 $8.44~\mathrm{s}$ 降到 $7.19~\mathrm{s}$. 提升幅度比较明显。通过 Nsight Compute 来分析一下受内联函数影响的三个核函数前后数据的对比：

| 核函数 | 寄存器占用情况 | Memory% | Duration |
|:-:|:-:|:-:|:-:|
| `rhs_kernel( )` | $128 → 128$ | $37.4\% → 38.2\%$ | $6.30 → 2.93~\mathrm{ms}$ |
| `rhs_dissipation_kernel( )` | $61 → 48$ | $56.0\% → 48.5\%$ | $2080 → 653~\mathrm{µs}$ |
| `rhs_constraints_kernel( )` | $154 → 80$ | $35.5\% → 52.4\%$ | $2340 → 893~\mathrm{µs}$ |

原来由于被调用函数的寄存器需求导致调用者的寄存器占用降不下去，现在将两个函数内联之后，可以做进一步的 `__launch_bounds__( )` 了，目前 `rhs_dissipation_kernel( )` 核函数压到了 $( 256, 5 )$，`rhs_constraints_kernel( )` 核函数压到了 $( 256, 3 )$ 的水平。然后我尝试做 `rhs_kernel( )` 函数压缩到 $(256,3)$，但是单步演化的时间从 $7.19~\mathrm{s}$ 提高到 $7.35~\mathrm{s}$. 可以推断是由于数据从寄存器 spill 到 local memory 里面导致数据加载变慢，warp 并发度提高带来的收益小于 spill 带来的损失。后面又尝试将三个 `advection( )` 函数压缩到 $(256,5)$，出现微小的提升，单步演化从 $7.19~\mathrm{s}$ 降低到 $7.10~\mathrm{s}$.

目前已经对 `rhs_kernel( )` 做过一些优化了，重新通过全程序的 profiling 来判断热点函数（还是跑两步演化）：


| 核函数 | 总时间 | 占比 | 调用次数 |
|:-:|:-:|:-:|:-:|
| **`prolong3_kernel( )`** | **$4.25~\mathrm{s}$** | $27.9\%$ | $23643$ |
| `rhs_kernel( )` | $3.98~\mathrm{s}$ | $26.1\%$ | $821$ |
| `restrict3_kernel( )` | $1.31~\mathrm{s}$ | $8.6\%$ | $4815$ |
| `global_interp_kernel( )` | $0.98~\mathrm{s}$ | $6.5\%$ | $688$ |
| `rhs_dissipation_kernel( )` | $0.95~\mathrm{s}$ | $6.2\%$ | $821$ |
| `rhs_advection_gauge_kernel( )` | $0.91~\mathrm{s}$ | $6.0\%$ | $821$ |

可以看到 `prolong3_kernel( )` 目前已经成为占比最高的核函数，尝试用 Nsight Compute 对其进行分析：

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/prolong_launch.png)

可以看到单次核函数 launch 只有 $43$ 个 block，数量比较少。由于单个线程占用的寄存器数量是 $66$，单个 SM 上同时驻留的 block 数为 3，一共 $14$ 个 SM，那么平均下来只有 $1.02~\mathrm{Waves}$. 在 Grid 的规模不变的前提下，其实没必要进一步压缩寄存器，因为确实没有足够的 warps 来支持并发了。

![](/Users/griffindor/HPC101/report_scripts/asset_lab3/prolong_throughput.png)

观察当前的吞吐数据，可以看到 L2 Cache 的吞吐只有 $6.35\%$，DRAM 的吞吐只有 $3.59\%$. 这表明 `prolong3_kernel( )` 的空间局部性很高，大部分数据都在 L1 Cache 中。所以存储方向的优化也可以放一放，这是一个计算瓶颈的问题。`prolong3_kernel( )` 只是一个壳子，内部套着实际的计算负载 `d_prolong3_device( )`. 

有了 `rhs_kernel( )` 系列核函数的优化经验，可以先做一下内部点快速路径。`d_prolong3_device( )` 的插值核心是每个目标点 36 次 `d_symmetry_bd_1b( 3, ... )` 粗网格访问，每次做 6 次边界比较、及可能的对称翻转。但大多数目标点的粗邻居 $±3$ 全部落在粗网格域内，这些检查永不触发，属于无意义的计算和分支。通过统一的 `interior` 计算优化掉 `d_symmetry_bd_1b( )` 中大量的分支语句。 

```cpp
const bool interior =												/* 内部点判定 */
    (cxI_i - 2 >= 1 && cxI_i + 3 <= extc[0] &&
     cxI_j - 2 >= 1 && cxI_j + 3 <= extc[1] &&
     cxI_k - 2 >= 1 && cxI_k + 3 <= extc[2]);

auto gv = [&](int ic, int jc, int kc) -> double {
    if (interior) {
        return func[get_col_major_idx(
            ic - 1, jc - 1, kc - 1, extc[0], extc[1], extc[2])]; 	/*  直接索引 */
    }
    return d_symmetry_bd_1b(3, extc, func, ic, jc, kc, SoA);        /* 边界点回退 */
};
```

测试添加快速路径后的性能，出现一定的提升，单步演化时间从 $7.10~\mathrm{s}$ 降到了 $6.65~\mathrm{s}$. 发现添加内部点快速路径对于性能的提升具有通用性，因此下一步我也在 `restrict3_kernel( )` 这边加上了内部点快速路径，思路是一样的。经过测试，`restrict3_kernel( )` 的内部点快速路径添加之后，单步演化时间从 $6.65~\mathrm{s}$ 降到了 $6.31~\mathrm{s}$，确实是有效的。

目前 `prolong3_kernel( )` 也已经做了一些优化了，下一步转到 `rhs_kernel( )`，`prolong3_kernel( )` 以及 `restrict3_kernel( )` 后面的一个核函数，即 `global_interp_kernel( )` 核函数，其目前是没有做任何优化的。这个函数的核心 `d_polin3_1b( )`（`fmisc_gpu.cu`）仍用 Numerical Recipes 的 `polint( )`（采用 **Neville** 递推算法），基本逻辑是：

给定节点 $x_1<...<x_n$ 与对应的数值 $y_i$，递推生成 $n-1$ 次插值多项式：
$$
P_{i}(x)=y_i,\qquad

P_{i,j}(x)=\frac{(x-x_j)\,P_{i,j-1}(x)\;-\;(x-x_i)\,P_{i+1,j}(x)}{x_i-x_j}
$$
最终 $P_{1,n}(x)$ 即插值结果（每步附带误差估计），计算量 $O(n^2)$。数学上等价于 Lagrange 算法：
$$
P(x)=\sum_{j=1}^{n} y_j\, L_j(x),\qquad
L_j(x)=\prod_{\substack{k=1\\k\neq j}}^{n}\frac{x-x_k}{x_j-x_k}
$$
即基函数加权和，因为每一个基函数的复杂度为 $O(n)$，所以总计算量同为 $O(n^2)$，数学上等于 Neville。但是在本程序中计算的都是等距插值，所以基函数其实存在闭式解，可以提前全部预计算好，将计算复杂度从 $O(n^2)$ 降低到 $O(n)$. 闭式解公式为：
$$
P(x)=\frac{\displaystyle\sum_{j=1}^{n} \frac{w_j}{x-x_j}\,y_j}
{\displaystyle\sum_{j=1}^{n} \frac{w_j}{x-x_j}},
\qquad
w_j=\frac{1}{\prod_{k\neq j}(x_j-x_k)}
$$
对等距节点 $x_j=j-1\;(j=1 .. n)$，代入 $x_j-x_k=j-k$ 得：
$$
\prod_{k\neq j}(x_j-x_k)=(j-1)!\,(-1)^{n-j}\,(n-j)!
$$


 故权重闭式解（1-based）等价于（差一个公共的 $(n-1)!$ 的系数）：
$$
w_j=(-1)^{n-j}\binom{n-1}{j-1}\qquad(j=1..n)
$$
基于上面的实现方法，我将 `polint( )` 算法从 Neville 递推算法转换到 Lagrange 算法，同时也做了进一步的优化。权重与每轴倒数预计算，权重 $w_j$ 只依赖 $n$（与 $x$ 无关），每轴倒数 $1/(x-x_j)$ 只依赖该轴的插值位置。三者都是"整列共享的常量"，每点只算一次即可，如下所示：
$$
r_j^{(d)}=\frac{w_j}{x^{(d)}-x_j^{(d)}}\quad(d=z,y,x\ \text{各 6 个})
$$
之后每次做插值只是做如下的计算即可，这样是最大程度做到计算数据的复用：
$$
y^{(d)}=\frac{\sum\limits_j r_j^{(d)}\,y_j}{\sum\limits_j r_j^{(d)}}
$$
加上 `prolong3_kernel( )` 做了一些 shared memory 的利用，这两步优化之后做性能测试，单步演化时间从 $6.31~\mathrm{s}$ 优化到 $5.45~\mathrm{s}$. 效果还是比较明显的。



-----



## 3. 鲲鹏平台上优化 ABE 程序

由于 arm 架构上没有 Intel 的 Vtune 性能分析工具，所以只能通过 `perf record` 按照固定的频率采样，观察采样时正在哪一个函数里执行，以此判断热点函数是什么以及相对应的执行时间占比，该数据会写入 perf.data 中。之后通过 `perf report` 解析 perf.data 就可以看到按照占比展示的热点函数。所以首先按照默认的配置来做一下热点函数的调用，默认的并行配置是 $\mathrm{MPI~rank}=30$，不存在 OPENMP 的线程级并行。因此默认为 $30$ 个并行进程（每一个进程为单线程），发现 `libopen-pal.so` 为代表的通信开销太高，于是通过测试脚本遍历 ${1,2,4,8,16}$ 作为 rank 数时的相关数据：

| MPI rank |      单步时间      | 通信开销（mpi_overhead_pct） |
| :------: | :----------------: | :--------------------------: |
|   $1$    | $135.0$ 个单位时间 |             $0$              |
|   $2$    | $75.9$ 个单位时间  |            $6.83$            |
|   $4$    | $50.93$ 个单位时间 |            $9.26$            |
|   $8$    | $56.7$ 个单位时间  |           $20.98$            |
|   $16$   | $46.9$ 个单位时间  |           $23.55$            |

可以看到随着 MPI rank 做 $1 \longrightarrow 2 \longrightarrow 4$ 倍增时，单步时间确实在显著的降低，但是从 $4 \longrightarrow 8$ 及以后并未出现进一步的明显的降低；另外观察通信开销，$4 \longrightarrow 8$ 的通信开销也是非常显著地翻倍，后面的通信开销会使得并行化带来的速度提升被吞噬掉，并行度过高会造成单个 rank 的 block 过小，无法掩盖进程间通信开销。因此，基于上述数据，当前最优的 rank 为 4. 此外我们还需要为后面的 OPENMP 做准备，当前每个进程只有一个线程，未开线程级并行，平台最多提供 $30$ 核 $60$ 个线程，所以如果 MPI rank 数过高则会压缩 OPENMP 的并行空间。因此，后面我都将 MPI rank 设置为 $4$. 

所以首先对 ABE 来做一次 perf 的 profiling，为了加速测试，将演化时间设置为 $1.0$（即单步演化）：

|           热点函数            | 时间占比  |
| :---------------------------: | :-------: |
|  `compute_rhs_bssn( )` 函数   | $34.79\%$ |
|       `point_( )` 函数        | $9.39\%$  |
|       `kodis_( )` 函数        | $7.96\%$  |
|      `fdderivs( )_` 函数      | $6.22\%$  |
|      `lopsided_( )` 函数      | $4.48\%$  |
| `libopen-pal.so` 进程通信开销 | $4.42\%$  |
| `__memset_sve_zva64( )` 函数  | $3.16\%$  |

可以看到绝对热点是 `compute_rhs_bssn( )` 函数。



