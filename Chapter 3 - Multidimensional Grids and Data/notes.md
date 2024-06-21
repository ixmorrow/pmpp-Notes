# Chapter 3 Notes

## Multidimensional Grid Organization

In CUDA, all threads in a grid execute the same kernel function and they rely on coordinates (thread indices) to distinguish themselves from each other and to identify the appropriate portion of the data to process.

Grids organized into a two level hierarchy:

Level 1: Blocks. A grid consists of one or more blocks.

Level 2: Threads. Each block consists of one or more threads.

All threads in a block share the same block index -> `blockIdx`. Each thread also has a thread index -> `threadIdx`.

Execution Configuration Parameters of a Kernel Call:

Contained within `<<< ... >>>` of the kernel call statement.

* 1st param is the dimensions of the grid in the number of blocks

* 2nd is dimensions of each block in terms of number of threads

These params have a particular type, `dim3` -> an int vector type of 3 elements (x, y, z).

* can use less than 3 dimensions by setting size of a dimensions to 1

For convenienve, you can also just omit the y, x dimensions from the kernel execution params altogether if you want the grids/blocks to be 1 dimensional.

```c
vecAddKernel<<<ceil(n/256.0), 256>>>(...);
```

Total size of a block in current CUDA systems is limited to 1024 threads. These threads can be distributed across the 3 dimensions in any way as long as the total number of threads does not exceed 1024.

A grid and its threads do not need to have the same dimensionality. A grid can have higher dimensionality than its blocks and vice versa.

Typically CUDA grids contain thousands to millions of threads.

## Mapping Threads to Multidimensional Data

Refer to the dimensions of multidimensional data in descending order, i.e. z dimension, then y, and x. This is the opposite of the order in which data dimensions are ordered in the `gridDim` and `blockDim` dimensions.

All multidimensionality arrays in C are linearized automatically by the compiler! Modern computers have a "flat" memory space and the only way to actually represent a multidimensional array is to flatten it into a 1D array. The compiler allows programmers to reference the array in multidimensional terms (`Pin_d[j][i]`) but under the hood, the compiler translates this indexing syntax into a 1D offset!

* Very interesting, I'm not sure if I've ever thought about this.

_Memory Space_ - a simplified view of how a processor accesses its memory in modern computers. The data to be processed by an application and instructions executed for the application are stored in locations in its memory space. Each location can typically accomodate a byte and has an address. Variables that require multiple bytes are stored in consecutive byte locations.

* This is what RAM(or Random Access Memory is used for)

* Also includes virtual memory, which is additional memory managed by the OS. It is used by running applications/processes when RAM is getting full and additional memory is required. Allows for the temporary transfer of data from RAM to disk.

* also have access to Cache memory (L1, L2, L3 caches) which are faster than RAM but much smaller. We went over using Cache memory in the online C++ course I took.
