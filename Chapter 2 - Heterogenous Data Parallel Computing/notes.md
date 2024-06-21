# Chapter 2 Notes

Data parallelism refers to the phenomenon in which the computation work to be performed on different parts of the dataset can be done independently of each other and thus can be done in parallel with each other.

* Independent evaluation of different pieces of data is the basis of data processing

## Task Parallelism vs Data Parallelism

task parallelism is typically exposed through task decomposition of applications. It exists if the two tasks can be done independently. I/O and data transfers are also common sources of tasks.

In general data parallelism is the main source of scalability for parallel programs. Nevertheless, task parallelism can also play an important role in achieveing performance goals.

## CUDA C Program structure

The structure of a CUDA C program reflects

1. Execution of a CUDA program starts with host code (CPU Serial Code). 
2. Kernel Function is called, a large number of threads are launched on a device (GPU) to execute the kernel.
3. All the threads launched by a kernel are collectively called a grid. These threads are the primary vehicle of parallel execution in a CUDA platform.
4. When all threads of a grid have completed their execution, the grid terminates, and the execution continues on the host until another grid is launched.

CUDA programmers can assume that these threads take very few clock cycles to generate and schedule, owing to efficient hardware support. This assumption contrasts with traditional CPU threads, which typically take thousands of clock cycles to generate and schedule.
* Think of the example of Team Jade utilizing Thread Pools to cut back on the amount of times a thread is spun up and down to save on the clock cycles required doing so.

### One of the best descriptions of a thread I have seen

A thread is a simplified view of how a processor executes a sequential program in modern computers. A thread consists of the code of the program, the point in the code that is being executed, and the values of its variable and data structures.

## Vector Addition Kernel

When there is a need to distinguish between host and device data, we will suffix the names of variables that are used by the host with "_h" and those of variables that are used by a device with "_d".

## Device Global Memory and Data Transfer

In current CUDA systems, devices are often hardware cards that come with their own dynamic random-access memory called device global memory, or simply global memory.

Calling it "global memory" distinguishes it from other types of device memory that are also accessible to programmers.

CUDA API functions for managing device global memory

`cudaMalloc()` allocates object in the device global memory. Two parameters:
* address of a pointer to the allocated object
* Size of allocated object in terms of bytes

`cudaFree()` frees object from device global memory
* Pointer to freed object

Dereferencing a device global memory pointer in host code can cause exceptions or other types of runtime errors. It should not be done.

Google Colab CUDA GPU notebook:
https://colab.research.google.com/drive/13gqmXA_ZvSoekshObxZfvW0PLNEwKaTW?usp=sharing

Tutorial:
https://giahuy04.medium.com/the-easiest-way-to-run-cuda-c-in-google-colab-831efbc33d7a

CUDA C Programming Guide from NVIDIA:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## Kernel Functions and Threading

In CUDA C, a kernel function specifies the code to be executed by all threads during a parallel phase.

Because all these threads execute the same code, CUDA C programming is a great example of Single program mulitple data parallel programming syte.

Key Concepts:

1. Single Program: All processors execute the same program code.

2. Multiple Data: Each processor works on its own subset of the data independently.

SPMD allows multiple processors to execute the same program on different data sets.

When a program's host code calls a kernel, the CUDA runtime system launches a grid of threads that are organized into a two-level hierarchy. Each grid is organized as an array of thread blocks, which we will refer to as blocks for brevity. All blocks of a grid are of the same size; each block can contain up to 1024 threads on current systems.

The total number of threads in each block is specified by the host code when a kernel is called.

Same kernel can be called with different numbers of threads at different parts of the host code.

for a given grid of threads, th enumber of threads in a block is available in a built-in variable named `blockDim`.

`blockDim` -> struct with three unsigned int fields (x, y, and z) that help to organize the threads into a 1, 2, or 3-D array.

* the choice of dimensionality for the threads usually reflects the dimensions of the data.

This makes sences because the threads are created to process data in parallel, so it is only natural that the organization of the threads reflects the organization of the data.

In general, it is recommended that the number of threads in each dimension of a thread block be a multiple of 32 for harware efficiency reasons.

Two more built-in variables CUDA kernels have access to:

* `threadIdx` - gives each thread a unique coordinate within a block.

* `blockIdx` - gives all threads in a block a common block coordinate

Telephone numbers in the U.S. have a similar hierarchical organization to CUDA threads. All phone numbers have a local area code that preceeds their local phone number. 

The area code specifies which locality in the U.S. the number is targeting and the phone number is the specific local telephone line.

One can think of each phone line as a CUDA thread, with the area code as the calue of `blockIdx` and the seven-digit local number as the value of `threadIdx`.

Similarly, each thread can combine its `threadIdx` and `blockIdx` values to create a unique global index for itself within the entire grid.

`__global__` - keyword that indicates a function is a kernel and that it can be called to generate a grid of threads on a device.

`__device__` - keyword indicates that the function being declared is a CUDA device function. A device function executes on a CUDA device and can be called only from a kernel function or another device function.

`__host__` - indicates a host function, simply a traditional C function that executes on the host and can be called only from another host function.

You can declare a function as both `__device__` and `__host__`. The compiler will generate a version for each the device and host.

The built-in variables are the means for threads to accesss hardware registers that provide the identifying coordinates to threads.

Automatic variables - variables local and private to a thread

## Calling Kernel Functions

When host code calls a kernel, it sets the grid and thread block dimensions via execution configuration parameters.

The configuration params are given between the `<<<` and `>>>` before the traditiona C function arguments.

The first config provides the number of blocks in the grid.

The second specifies the number of threads in each block.

To ensure that we have enough threads in the grid to cover all the vector elements, we need to set the number of blocks in the grid to the ceiling division (rounding up the quotient to the immediate higher integer value) of the desired number of threads (n in this case) by the thread block size (256 in this case).

* we use an arbitrary block size of 256 in this example, the block size should be determined by a number of factors that will be introduced later...

* was wondering why 256 was chosen as block size

Note that all the thread blocks operate on different parts of the vectors and they can be executed in any arbitrary order. DO NOT make any assumptions about execution order.

Since the example of vector addition is very simple, the CUDA method of performing this operation will likely take longer to execute than a traditional sequential method. 

CUDA and GPUs really shine when handling much larger datasets and doing much more complex computations. Real applications typically have kernels in which much more work is needed relative to the amount of data processed, which makes the additional overhead worthwhile. Real applications also tend to keep the data in the device memory across multiple kernel invocations so that the overhead can be amortized.

A specific compiler is needed to run CUDA code, NVCC is an example of a C/C++ native compiler that recognizes and compiles CUDA code as well. NVCC will complie the host's code with a standard C/C++ compiler. The device code is compiled by NVCC into virutal vinary files called PTX files. These PTX files are further compiled by a runtime component of NVCC into the real object files and executed on a CUDA-capable GPU.

### CUDA Thread Hierarchy:

CUDA organizes threads into a hierarchy of grids, blocks, and threads:

Grid: A grid consists of multiple blocks.
Block: Each block contains multiple threads.
Thread: Each thread executes a kernel function and has a unique index within its block.

How specific threads are identified:
`i=blockIdx.x*blockDim.x + threadIdx.x;`

blockIdx.x * blockDim.x:

This part computes the starting index for the current block.

Each block processes a contiguous chunk of the data.
For example, if blockDim.x is 256 and blockIdx.x is 2, the starting index for block 2 is 2 * 256 = 512.

* So the `blockDim` are the number of threads in each block. But each blocks thread id is a continous index.

For example, Blocks with dimensions of 256 - Block 0 will have threads 0-255, Block 1 will have threads 256 - 511.

This methodology gives each thread a unique id.