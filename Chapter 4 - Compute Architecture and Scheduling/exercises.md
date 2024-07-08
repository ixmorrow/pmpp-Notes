# Exercises

1. Consider the following CUDA kernel and the corresponding host functions that calls it:

```c
__global__ void foo_kernel(int* a, int* b){
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(threadIdx.x < 40 || threadIdx.x >= 104){
		b[i] = a[i] + 1;
	}
	if(i%2 == 0){
		a[i] = b[i] * 2;
	}
	for(unsigned int j = 0; j < 5 - (i%3); ++j){
		b[i] += j;
	}
}

void foo(int* a_d, int* b_d){
	unsigned int N = 1024;
	foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
}
```

1. What is the number of warps per block?
    1. 128 threads per block, 32 threads per warp → 128/32 = 4 warps per block.
2. What is the number of warps in the grid?
    1. (N + 128 - 1)/128 evaluates to 9 blocks in the grid, with 128 threads per block. We know there are 4 warps per block, so 9 blocks * 4 warps per block = 36 warps in the grid.
3. For the statement on line 04 → `b[i] = a[i] + 1;`:
    1. How many warps in the grid are active?
        1. Any warps with threadIdx.x values between 0-39 and 104+. For each block, Warp 0 (0-31) Warp 1 (32-64) Warp 3 (96 - 127) will be active or partially active. That is 3/4 warps per block, with 9 blocks that is 9 * 3 = 27 warps active in total.
    2. How many warps in the grid are divergent? A block is divergent if some of its threads execute one code path, while others execute another.
        1. For each block, Warp 1 (32-63) and Warp 3 (96 - 127). That is 9 * 2 = 18 warps will be divergent.
    3. What is the SIMD efficiency (in %)  of warp 0 of block 0?
        1. Warp 0 encompasses threads 0-31. This warps will be at 100% (32/32) capacity.
    4. What is the SIMD efficiency (in %) of warp 1 of block 0?
        1. Warp 1 encompasses threads 32-63. This thread will only be at 25% (8/32) capacity.
    5. What is the SIMD efficiency (in %) of warp 3 of block 0?
        1. Warp 3 encompasses threads 96 - 127. This thread will only be at 75% (24/32) capacity.
4. For the statement on line 07 → `a[i] = b[i] * 2;` :
    1. How many warps in the grid are active?
        1. All warps will be active on this line because all warps contain some threads that will be divisible by 2. However, only 50% of threads in each warp will pass this if statement. So, 36 warps are active.
    2. How many warps in the grid are divergent?
        1. All warps in the grid are divergent on this line because half the threads will enter the if statement and half will not. 36 warps will be divergent.
    3. What is the SIMD efficiency (in %) of warp 0 of block 0?
        1. 50%
5. For the loop on line 09 → `for(unsigned int j = 0; j < 5 - (i%3); ++j)` :
    1. How many iterations have no divergence?
    2. How many iterations have divergence?
6. For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?
    1. 2000 elements / 512 threads per block = ~3.9 blocks, which means 4 blocks are required. 4 blocks * 512 threads/block = 2048 threads in the grid.
7. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length.
    1. If the vector is 1 dimensional, then I expect only the very last warp to have divergence.
8. Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads’ total execution time is spent waiting for the barrier?
    1. I think you have to take the difference between each threads execution time and the longest execution time to determine how long each thread waits. Then, you can use that to determine what percentage of time is spent waiting. The longest execution time is 3 microseconds.
        1. (3 - 2) = 1 micros, 1/(1 + 2) ⇒ 33%
        2. (3-2.3) = 0.7 micros, 0.7/3 ⇒ 23%
        3. (3-3) = 0, 0% (last thread to finish so doesn’t spend any time waiting)
        4. (3-2.8) = 0.2 micros, 0.2/3 ⇒ 6%
        5. 3 - 2.4 = 0.6 micros, 0.6/3 ⇒ 20%
        6. 3 - 1.9 = 1.1 micros, 1.1/3 ⇒ 36.6%
        7. 3 - 2.6 = 0.4 micros, 0.4/3 ⇒ 13.3%
        8. 3 - 2.9 = 0.1 micros, 0.1/3 ⇒ 3.3%
9. A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the `__syncthreads()` instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.
    1. I don’t think this is a good idea. Just because there are 32 threads in each block does not mean there could be divergence and if a barrier synchronization is needed that means all threads must be finished with some portion of the code before continuing. Enforcing all blocks to contain 32 threads is not just a general rule of thumb that can solve this. That will depend on the kernel code.
10. If a CUDA device’s SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM?
    1. b. 256 threads per block is the configuration that allows for the most number of threads in the SM while utilizing all 4 blocks. However, 512 threads per block would allow for the most threads (1536) but will only utilize 3/4 blocks.
11. Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level. 32 threads per warp, supports 2048 threads/32 threads per warp ⇒ 64 warps.
    1. 8 blocks w/ 128 threads each is possible ⇒ 128 threads / 32 threads per warp = 4 warps per block * 8 blocks = 32 warps utilized ⇒ 32/64 ⇒ 50 % occupancy level.
    2. 16 blocks w/ 64 threads each is possible ⇒ 64 threads / 32 threads per warp = 2 warps per block * 8 blocks = 16 warps utilized ⇒ 16/64 ⇒ 25% occupancy level.
    3. 32 blocks w/ 32 threads each is possible ⇒ 32 threads / 32 threads per warp = 1 warp per block * 8 blocks = 8 warps utilized ⇒ 8/64 ⇒ 12.5% occupancy level.
    4. 32 blocks with 64 threads each is possible ⇒ 64 threads / 32 threads per warp = 2 warps per block * 8 blocks = 16 warps utilized ⇒ 16/64 ⇒ 25% occupancy level.

i. Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

**Full Occupancy**

To achieve full occupancy, you need to meet the constraints for:

1.	Maximum threads per SM.

2.	Maximum blocks per SM.

3.	Register usage per SM.

1. The kernel uses 128 threads per block and 30 registers per thread → No, that is too many threads per block. 128 threads * 32 blocks = 4096 threads total. Also, the number of registers exceeds the 64K limit ⇒ 30 * 4096 = 122880
2. The kernel uses 32 threads per block and 29 registers per thread → No, not enough threads per block. 32 threads * 32 blocks = 1024 threads total.
3. The kernel uses 256 threads per block and 34 registers per thread → No, too many threads per block. 256 threads * 32 blocks = 8192 threads. Also the number of registers exceeds the 64K limit ⇒ 34 * 8192 = 278,528 registers

j. A student mentions that they were able to multiply two 1024 x 1024 matrices using a matrix multiplication kernel with 32 x 32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a block calculates one element of the result matrix. What would be your reaction and why?

1. The total elements is 1024 * 1024 = 1,048,576 elements.
2. Each block calculates 32 * 32 = 1024 elements
3. Number of blocks needed = 1048576/1024 = 1024 blocks
4. Each SM supports 8 blocks which would mean we’d need a GPU device with multiple SMs, this is feasible with today’s technology. However, each SM only supports blocks up to 512 threads each. The student’s scenario is utilizing block sizes of 1024 threads. The SMs do not support that.