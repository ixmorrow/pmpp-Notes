1. If we want to use each thread in a grid to calculate one output elelment of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?

Answer:
(C) `i=blockIdx.x*blockDim.x + threadIdx.x;`

2. Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?

Answer:
(C) `i=2*(blockIdx.x*blockDim.x + threadIdx.x)`

```c
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (i < n) {
        C[i] = A[i] + B[i];
    }
    if (i + 1 < n) {
        C[i + 1] = A[i + 1] + B[i + 1];
    }
}
```

3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes `2*blockDim.x` consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?

Answer:
(D) `i=blockIdx.x*blockDim.x*2 + threadIdx.x;`

4. For a vector addition, assume that the vecotr length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?

Answer:
(C) 8192 threads

8000/1024 = 7.8125 blocks, round up to 8 blocks required.

8 * 1024 threads per block = 8192 threads in total.

5. If we want to allocate an array of v integer elements in the CUDA device `global` memory, what would be an appropriate expression for the second argument of the `cudaMalloc` call?

Answer:
(D) `v*sizeof(int)`

6. If we want to allocate an array of `n` floating-point elements and have a floating-point pointer variable `A_d` to point to the allocated memory, what would be an appropriate expression for the first argument of the `cudaMalloc()` call?

Answer:
(D) `(void**) &A_d`

7. If we want to copy 3000 bytes of data from host array `A_h` (`A_h` is a pointer to element 0 of the source array) to device array `A_d` (`A_d` is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?

(C) `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`

for some reason, the parameter order in this API feels backwards. I feel like it should be source first, then destination. But it's destination first, then source...

8. How would one declare a variable err that can appropriately receive the returned value of a CUDA API call?

Answer:
(C) `cudaError_t err;`

9. Consider the following CUDA kernel and the corresponding host function that calls it:

``` c
__global__ void foo_kernel(float* a, float* b, unsigned int n){
    unsigned in i=blockIdx.x*blockIdx.x + threadIdx.x;

    if(i < N){
        b[i] =2.7f*a[i]-4.3f;
    }
}

void foo(float* a_d, float* b_d){
    unsigned int N=200000;
    foo_kernel<<<(N + 128-1)/128, 128>>>(a_d, b_d, N);
    }
```

a. What is the number of threads per block? -> 128 threads/block
b. What is the number of threads in the grid? -> (200,000 + 127/128) = 1,563.4921875 => round up to 1,564 blocks. At 128 threads/block => 1,564 * 128 = 200192 threads total
c. What is the number of blocks in the grid? -> 1,564 blocks
d. What is the number of threads that execute the code on line 02? -> 200,192
e. What is the number of threads that execute the code on line 04? -> 200,000

10. A new summer intern was frustrated with CUDA. He has been complaining that CUDA is very tedious. He had to declare many functiosn that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?

Answer:
Silly little intern, you work too hard. You can declare a single function as both a `__host__` and `__device__` function if you intend to use it both on the host and device. There is no need to actually define the same function twice for this purpose. Problem solved!!