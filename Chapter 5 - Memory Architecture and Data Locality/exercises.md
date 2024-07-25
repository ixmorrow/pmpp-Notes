1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

Example Matrix Addition Kernel
```c
__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}
```
I think if arrays A, B, and C were loaded into shared memory to begin with, then maybe that would reduce the global memory bandwidth. They are declared using `cudaMalloc`(), which allocates to global memory. However, each thread accesses a single element from those 3 arrays and thus there are no shared memory accesses between threads. Because of this using shared memory would not reduce global memory bandwidth with the example kernel I have provided.

2. 

3. What type of incorrect execution behavior if one forgot to use one or both `__syncThreads()` in the kernel in Fig 5.9?

```c
__global__ void MatrixMulKernel(float* M, float* N, float* P, Int Width)
{
__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
int bx = blockIdx.x; int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
int Row = by * blockDim.y + ty;
int Col = bx * blockDim.x + tx;
float Pvalue = 0;
// Loop over the M and N tiles required to compute the P element
for (int p = 0; p < n/TILE_WIDTH; ++p) {
	// Collaborative loading of M and N tiles into shared memory
	ds_M[ty][tx] = M[Row*Width + p*TILE_WIDTH+tx];
	ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*Width + Col];
	__syncthreads();

	for (int i = 0; i < TILE_WIDTH; ++i)Pvalue += ds_M[ty][i] * ds_N[i][tx];
	__synchthreads();
}
P[Row*Width+Col] = Pvalue;
}
```

Without the first occurence of `syncThreads()`, there is no guarantee that data would be fully loaded into the tile in shared memory. Continuing execution in this cause could result in unexpected results when accessing elements of the tile that have not been initialized with data yet.

Without the second occurence, we would not have a guarantee that each tile would finish computation before moving on to the next tile. Without this, threads may start loading new data into shared memory before other threads have finished accessing and computing the data currently stored in shared memory.

4. Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.

The main reason that I can think of is that global memory is shared and accesible among all threads and all blocks within a grid. Similarly, shared memory is accessible among all threads within a block. So, by moving data from global memory to shared memory you retain that shared accessibility aspect that is available to you in global memory, while also getting a performance improvement in terms of access latency.

5. For our tiled matrix-matrix multiplication kernel, if we use a 32x32 tile, what is the reduction of memory bandwidth usage for input matrics M and N?

6. Assume that a CUDA kernel is launched with 1000 grid blocks, each with 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?

1000 blocks * 512 threads/block = 512,000 instances of the local variable.

7. In the previous question, if the variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the kernel?

Shared memory variables are shared among threads within a block. For this reason, I believe each block would have its own instance of the shared variable. Since there are 1000 blocks, this would create 1000 instances of the variable.

8. Consider performing a matrix multiplication of two matrices of NxN. How many times is each element in the input matrices requested from global memory when:

* There is no tiling?
* Tiles of size TxT are used?

Matrix multiplication without tiling and assuming a single thread is responsible for one element in the output, means each thread would have to load an entire row from Matrix A and an entire Column from Matrix B to compute its output element. That is 2*N accesses per thread.

With Tiling, each thread acceses 2*T elements per thread.

9. A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following properties, indicate whether this kernel is memory bound or compute bound:

a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second.

Need to calculate the compute to memory ratio for the kernel and compare it to the hardware limitations.
Kernel
- 36 FLOPS
- 7 global memory accesses (32-bits, 4 bytes)

7 * 4 = 28 bytes per thread of global memory accesses.

36 flops/28 bytes per thread = 1.29 FLOPS/byte

Hardware
- 200 GFLOPS/100 GB/s = 2 FLOPS/byte

The kernel's compute to memory ratio is lower than the hardware's compute to memory ratio. This means that the kernel is performing fewer floating-point operations per byte of data accessed than the hardware can support. Since the kernel's ratio is lower than the hardware's it indicates the kernel cannot fully utilize the hardware's capabilities due to being limited by memory bandwidth. Therefore, the kernel is memory bound.

b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second.

Kernel
36 flogs/28 bytes per thread = 1.29 FLOPS/byte

Hardware
- 300 GFLOPS/250 GB/s = 1.2 FLOPS/byte

The kernel's compute to memory bandwidth is greater than the hardware's. This indicates that the kernel is compute bound. This means the kernel is performing more floating-point operations per byte of data than what the hardware is optimized for. The bottleneck is the processing speed of the computational units (e.g., ALUs, FPUs).

10. To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each dimension of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.

a. Out of the possible values BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on this device?

My concern is if the A_elements object is smaller than the possible indexing values used in the kernel.

b. If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.

We can add boundary condition checks to the few sections of the kernel that we have the potential for attempting to access memory that has not been allocated.

11. 
```c
__global__ void foo_kernel(float* a, float* b){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	float x[4];
	__shared__ float y_s;
	__shared__ float b_s[128];
	for(unsigned int j=0; j<4; ++j){
		x[j] = a[j*blockDim.x*gridDim.x + i];
	}
	if(threadIdx.x == 0){
		y_s = 7.4f;
	}
	b_s[threadIdx.x] = b[i];
	__syncThreads();
	b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3] + y_s*b_s[threadIdx.s] + b_s[(threadIdx.x + 3) % 128];
}

void foo(int* a_d, int* b_d){
	unsigned int N = 1024;
	foo_kernel<<<(128 + N - 1)/128, 128>>>(a_d, b_d);
}
```

a. How many versions of the variable i are there? -> 
128 + 1024 - 1 / 128 = 8
128 threads per block * 8 blocks = 1024 threads
1,152 , one for each thread executing the kernel

b. How many versions of the array x[] are there? -> 1,1024 , one for each thread executing the kernel

c. How many versions of the variable y_s are there? -> 8, one for each block

d. How many versions of b_s? -> 8, one for each block

e. What is the amount of shared memory used per block in bytes? -> 4 + 4*128 = 516 bytes of shared memory per block

f. What is the floating-point to global memory access ratio of the kernel? -> The only global memory accesses are from the input parameters `a` and `b`. We have the following:
4 reads from a + 1 read from b + 1 write to b => 6 global memory accesses
FLOPs per thread: 11

11/6 => ~1.83 FLOPS to global memory access ratio

12. Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65, 536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

**Full Occupancy**

To achieve full occupancy, you need to meet the constraints for:

1.	Maximum threads per SM.

2.	Maximum blocks per SM.

3.	Register usage per SM.

a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory per SM. -> 64 threads * 32 blocks = 2048 threads/SM
27 registers * 2048 threads = 55,296 registers. 
4 KB < 96 KB alloted

This configuration meets all of the requirements and can achieve full occupancy.

b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory per SM. -> 256 * 32 = 8192 threads/SM which is more than the 2048 threads/SM the hardware supports.

31 registers * 8192 threads = 253, 952 which is also more than the 64K limit on registers.
31 registers * 2048 threads = 63, 488 registers when using the max amount of threads.
8 KB < 96 KB alloted.

This configuration does not meet the requirements. The limiting factor is the number of threads.