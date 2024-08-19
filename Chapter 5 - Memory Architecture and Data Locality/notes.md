# Notes

Can calculate the effect of memory access efficiency by calculating the performance of the most executed portion of the kernel code.

Compute to global memory access ratio - ratio of floating-point operations to bytes accessed from global memory

- Floating Point Operations (FLOP)/Bytes of data
- defined as the number of FLOPs performed for each byte access from the global memory within a region of a program
- also referred to as arithmetic intensity or computational intensity in literature
- amount of work done for every byte of data loaded

Some Kernel execution is severely limited by the rate at which the data can be delivered from memory to the GPU cores.

- programs whose execution is limited by memory are called memory bound

To improve the performance of memory-bound applications, we need to reduce the amount of global memory accesses.

Memory types in GPUs:

- global - can be read/written to by both host and device code
- constant (we have not learned about this type yet)
- local - placed in global memory and has similar access latencies but is not shared between threads. each thread has a section of global memory that it uses as its own private local memory
- Registers  - allocated to individual threads, each thread can access only its own registers. This memory is stored on-chip and is extremely performant in terms of access latencies. A kernel function typically uses registers to hold frequently accessed variables.
- Shared memory - allocated to thread blocks, all threads in a block can access shared memory

We talked about how GPUs achieve zero overhead scheduling when switching between threads/warps in the previous chapter. CPUs must store all of the currently executing thread information (registers, etc) in memory before context switching, which takes time. GPUs have the ability to keep the registers of all threads that are scheduled in the processing block in the processing block’s register file. This way switching between warps is instantaneous because registers of the warp are already in the register file.

- for this reason, GPU register files need to be substantially larger than CPUs

From ChatGPT:

> Register allocation and management are critical for performance, as register pressure (demand for registers) can lead to spilling to slower memory (local or global memory).
> 

CPU registers designate a fixed number of registers per thread, whereas GPU registers need to be able to support a dynamic number of registers per thread as determined by the CUDA runtime?

Registers are stored in the register file which is stored “on-chip”, or in the processor itself. Whereas global memory is kept off-chip and must be accessed externally from the processor.

Another way to view the difference between memory accesses in registers vs global memory is to think about the number of instructions involved in each.

- Most modern processors have built-in register operands that can be called on register memory. They simply require the specific registers involved in the operation and thats it.
    - `fadd r1, r2, r3`
- Operations involving global memory variables require the chip to perform a memory load operation to make the data available to the ALU(Arithmetic Logic Unit). This means loading the data from global memory into a local register so that it can be operated on.

Additionally, in modern computers the energy that is consumed for accessing a value from the register file versus global memory is an OOM(Order of magnitude) lower!

In chapter 4, we learned that the number of registers available to each thread is limited. We also learned that there are performance degradations when we exceed those register limits. So we also need to avoid oversubscribing to this resource.

Shared memory and registers are different, but they both reside on-chip. Accessing shared memory requires a load instruction just like with global memory, but since it’s located on-chip it has much lower access latency and much higher throughput.

- however because of this load instruction, shared memory has higher latency and less bandwidth as utilizing registers
- in computer architecture terminology, shared memory is a form of scratchpad memory
- shared memory can be explicitly accessed in CUDA with the `__shared__` keyword

Data stored in shared memory is accessible by all threads in a block. This contrasts with register data which is private and local to a thread.

A CUDA device Streaming Multiprocessor (SM) typically has multiple processing units to allow multiple threads to make simultaneous progress. Threads in a block can be spread across these processing units. Therefore, the hardware implementations of shared memory are designed in a way to allow multiple processing units to simultaneously access its contents to support efficient data sharing among threads in a block.

Scalar variables - variables that are not arrays

All automatic scalar variables declared in kernel and device functions are stored in registers.

- these are all local to individual threads. When a variable of this type is declared, it will be created for every single thread that executes.
- when a thread terminates, all of its variables cease to exist

Automatic array variables are stored in a thread’s local memory

- scope of each of these is the same as automatic scalar variables, local to each thread

`__shared__` indicates a variable to be stored in shared memory

- shared variables are “shared” among all threads within a block
- A private version of each variable is created for and used by each block in an executing grid
- shared variables are an efficient means for threads within a block to collaborate with each other

“CUDA programmers often use shared variables to hold the portion of global memory data that is frequently used and reused in an execution phase of the kernel”

`__constant__` defines a constant variable in CUDA

- declarations must be outside any function body
- scope is all grids
- lifetime is entire application lifetime
- stored in global memory but are cached for efficient access
- size limited to 65, 536 bytes

A variable preceeded only by `__device__` is a global variable and is placed in global memory

No current way of syncing threads across blocks without using atomic operations or terminating the executing kernel

## 5.3 Tiling for Reduced Memory Traffic

We have learned we have an intrinsic tradeoff between types of memory in GPUs. There is global memory which is large but slow, and shared/local memory which is very small but very fast.

It is common to partition the data into subsets called tiles so that each tile fits into shared memory.

- how?
- Have the threads collaboratively load the next elements that they will operate on into shared memory before they actually use them

Explanation of Tiling: https://penny-xu.github.io/blog/tiled-matrix-multiplication

Strip mining - technique that takes a long running loop and breaks it into phases

### Tiling in my own words

Purdue/UIUC CUDA Tiling Teaching Kit: https://engineering.purdue.edu/~smidkiff/ece563/NVidiaGPUTeachingToolkit/Mod4/Lecture-4-4-tiled-matrix-multiplication-kernel.pdf

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

Let’s go over this indexing because it has taken me a while to understand what is going on here. M and N are the input matrices. 

- `for (int p = 0; p < n/TILE_WIDTH; ++p)` → calculates the number of tiles required given the number of elements in the input and loops over each tile
- `Row*Width` → returns the starting index for a given row in a block. This is because blocks are stored linearly in row-major format in the computer memory. Although these are 2D matrices, C does not support typical 2D indexing like Python, so we must index these exactly how they are stored in memory. Row major format means each row is stored one after the other in linear fashion in consecutive memory address space. So, to get the starting index of a specific row in memory, you have to take that row number and multiply it by the width of the block to get the integer value that represents where that row starts in memory.
- `p*TILE_WIDTH` → p represents the tile index/id and `TILE_WIDTH` is the width of a tile. So, with the first part of the index in the previous step we have moved to the beginning of a specific row in memory. By adding this, we can jump to the starting index of a specific tile in that row.
- `+ tx`  → adds the local thread index to the tile

Together `Row*Width + p*TILE_WIDTH + tx` returns the index for a specific element within a specific tile in matrix M. This is then stored in the `ds_M` data structure which is in the shared memory on chip because it was defined in the beginning of the kernel → `__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];` This is exactly the size of a tile and is reused throughout the process.

- `(p*TILE_WIDTH+ty)*Width` → p is the Tile index and multiplying this by the `TILE_WIDTH` will get you the starting index of this specific tile. Adding `ty` here will then get you a unique element from a different row within the current tile. So with that, you have the starting element of an item in a specific tile in a specific row. Multiplying this by the Width will return this items real index when stored in linear row major layout.
    - `ty` is the row we care about, multiplying this by `Width` will get you the actual starting index of this specific row in memory because these are stored in row major format in consecutive memory addresses.
- `+ Col` → moves the index to the specific column of this row
    - `int Col = bx * blockDim.x + tx;` → so Col is the unique Col this thread cares about in a given row. Once we have the right row for our tile (previous step), we can simply add this to that index to get the specific element we are interested in

From there, each element is in these two different arrays of size `TILE_WIDTH` x `TILE_WIDTH` in shared memory. Once the correct values have been copied over into shared memory, we execute barrier synchronization to make sure all threads copy their respective values over.

- Won’t this overwrite each value in the `ds_M` and `ds_N` matrices?
    - Answer: When using the tiling technique, we have to ensure that each block is responsible for 1 tile. This little piece of info was tripping me upas that was not clear to me. But if that is the case, then it makes sense why this works. In the case that each block is responsible for 1 tile, the values are not overwritten because `ds_M` and `ds_N` are shared within a block only.

once all the threads have completed this step, we move on to the actual matrix multiplication. Remember each thread executing this kernel is responsible for calculating a single value of the resulting matrix P. To do this, we iterate through the tile and multiply elements by each other and keep a running sum of this calculation. After we have multiplied all elements in the tile, we store that sum in a single element of the P matrix. Doing this for all threads, will calculate an answer for each element of P.

## Boundary Checks

Will now cover how to handle matrices of arbitrary length whose width is not a multiple of TILE_WIDTH.

Accessing elements that are outside the bounds of our data structure will result in unexpected behavior. Because the 2D matrices are stored linearly, accessing the element past the top row will return the first element of the 2nd row. Using this value in computation when a value from the top row was expected can result in incorrect results.

Similarly, accessing elements that are outside the bounds of memory allocated to the matrix/array entirely will result in undefined behavior. In some systems, this can return random values from other data structs, in others the access is rejected leading the program to abort and crash.

Good rule of thumb to use boundary checks at every single memory access to ensure we are always accessing valid memory.

- boundary condition for loading elements into tiles in shared memory
- boundary condition for accessing data in shared memory and storing into resulting matrix

In the case that these checks return false, we should store a value of 0.0 in the shared memory location.

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
	for (int p = 0; p < ceil(Width/(float)TILE_WIDTH); ++p) {
		// Collaborative loading of M and N tiles into shared memory
		if(p*TILE_WIDTH+tx < Width && Row < Width){
			ds_M[ty][tx] = M[Row*Width + p*TILE_WIDTH+tx];
		} else ds_M[ty][tx] = 0.0f;
		if (p*TILE_WIDTH+ty < Width && Col < Width){
			ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*Width + Col];
		} else ds_N[ty][tx] = 0.0f;
		__syncthreads();
	
		for (int i = 0; i < TILE_WIDTH; ++i)Pvalue += ds_M[ty][i] * ds_N[i][tx];
		__synchthreads();
	}
	if (Row < Width && Col  Width){
		P[Row*Width+Col] = Pvalue;
	}
}
```

This only works with square matrices. Revise the kernel to work with rectangular and square matrices. We need to replace Width with 3 separate variables: j, k, and l.

- Where Width is used to refer to the height of M or P, replace it with j
- Where Width is used to refer to the width of M or height of N, replace it with k
- Where Width is used to refer to the Width of N or the Width of P, replace it with l

```c
__global__ void MatrixMulKernel(float* M, float* N, float* P, Int j, Int k, Int l)
{
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;
	float Pvalue = 0;
	// Loop over the M and N tiles required to compute the P element
	for (int p = 0; p < ceil(k/(float)TILE_WIDTH); ++p) {
		// Collaborative loading of M and N tiles into shared memory
		if(p*TILE_WIDTH+tx < k && Row < k){
			ds_M[ty][tx] = M[Row*k + p*TILE_WIDTH+tx];
		} else ds_M[ty][tx] = 0.0f;
		if (p*TILE_WIDTH+ty < k && Col < k){
			ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*k + Col];
		} else ds_N[ty][tx] = 0.0f;
		__syncthreads();
	
		for (int i = 0; i < TILE_WIDTH; ++i)Pvalue += ds_M[ty][i] * ds_N[i][tx];
		__synchthreads();
	}
	if (Row < j && Col < l){
		P[Row*j+Col] = Pvalue;
	}
}
```

Shared memory usage can also hinder SM utilization just like register usage. Host code can actually dynamically scale up or down the amount of shared memory available to a kernel within the hardware limitations.

- this is done by calling `cudaGetDeviceProperties` function
- 

```c
&devProp = cudaGetDeviceProperties();
// gives the amount of shared mem that is available per block
devProp.sharedMemPerBlock
```

To declare `M_ds` and `N_ds` to support dynamic sizing, we need to adjust the declaration and use the `extern` keyword and combine the two matrices into one single array.

```c
extern __shared__ Mds_Nds[];
```

With this declaration, we can pass in the size of the array dynamically when the kernel function is called with a third parameter in the execution configurations.

```c
matrixMulKernel<<<dimGrid, dimBlock, size>>> (Md, Nd, Pd, Width, size/2, size/2);
```

May also be helpful to add input params to the kernel function to force the caller to pass in the size of M_ds and N_ds respectively so that the kernel code knows where on ends and the other begins.

# PMPP UIUC Course - Spring 2018 (Lecture 5)

* Global memory is implemented in DRAM - slow

Tiling:
- Partition data into subsets called Tiles
- Handle data from each tile in a single thread block by:
	- loading the tiles from global memory to shared memory using multiple threads to exploit memory-level parallelism
	- performing the computation on data in shared memory means each thread can efficiently access any data element in the entire tile
- load results from shared memory back into global memory

When you do high performance programming, it's not a lot more lines of code. It may only be a few lines of code difference. But the engineering decisions that go into those lines of code are much more intentional.

The professor suggests writing down the decisions that went into the design of our code because in a month we won't remember why.

# CUDA Mode Lecture on Chapter 5

Computational intensity - FLOP/Byte of memory transfer

You can hide some of the memory transfer latency if other warps can compute while other warps are waiting for data from global memory to arrive.

Implement flash attention from scratch? Not sure what flash attention is or how to implement it but will look into it.