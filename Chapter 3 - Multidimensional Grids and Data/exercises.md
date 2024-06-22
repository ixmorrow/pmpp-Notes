1. In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

    a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.

```c
__global__ void MatrixMultiplicationKernel(float* M, float* N, float* output, int widthM, int widthN, int heightM) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < heightM) {
        float value = 0.0f;
        for(int colIndex = 0; colIndex < widthN; colIndex++){
            for(int k = 0; k < widthM; ++k){
                value += M[row * widthM + k] * N[k * widthN + colIndex];
            }
        }
        output[row * widthN + k] = value;
    }
}

...
int threadsPerBlock = 256;
int numBlocks = (height_m + threadsPerBlock - 1)/threadsPerBlock;
MatrixMultiplicationKernel<<<numBlocks, threadsPerBlock>>>(m, n, output, width_m, width_n, height_m);
```

b. Write a kernel that has each thread produce one output matrixc olumn. Fill in the execution configuration parameters for the design.

```c
__global__ void MatrixMultiplicationKernel(float* M, float* N, float* output, int widthM, int heightN, int heightM) {
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < widthM) {
        float value = 0.0f;
        for(int rowIndex = 0; rowIndex < heightM; rowIndex++){
            for(int k = 0; k < heightM; ++k){
                value += M[col * heightM + k] * N[k * heightN + rowIndex];
            }
        }
        output[col * heightN + k] = value;
    }
}
```

2. A matrix-vector multiplication takes an input matrix B and a vector C and produces one output vector A. Each element of the output vector A is the dot product of one row of the input matrix B and C. For simplicity, we will handle only square matrices whose elements are single-precision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element.

```c
#include <cuda_runtime.h>
#include <iostream>

__global__ void MatrixVectorMultiplicationKernel(float* input_matrix, float* input_vector, float* vec_A, int num_elements){
    int row = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < num_elements){
        float value = 0.0f;
        for(int k=0; k < num_elements; ++k){
            value += input_matrix[row*num_elements+k] * input_vector[k];
        }
        vec_A[row] = value;
    }
}

void MatrixVectorMultiplication(float* input_matrix_h, float* input_vector_h, float* output_vector_h, int num_elements){
    // create device pointers
    float* input_matrix_d, input_vector_d, output_vector_d;
    size_t matrix_size = num_elements * num_elements * sizeof(float);
    size_t vector_size = num_elements * sizeof(float);

    // Allocate memory for data on device
    cudaMalloc((void**)&input_matrix_d, matrix_size);
    cudaMalloc((void**)&input_vector_d, vector_size);
    cudaMalloc(o(void**)&output_vector_d, vector_size);

    // copy data from host to device
    cudaMemcpy(input_matrix_d, input_matrix_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(input_vector_d, input_vector_h, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(output_vector_d, output_vector_h, vector_size, cudaMemcpyHostToDevice);

    // Kernel Configuration
    int threadsPerBlock = 256;
    int numBlocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // call Kernel
    MatrixVectorMultiplicationKernel<<<numBlocks, threadsPerBlock>>>(input_matrix_d, input_vector_d, ooutput_vector_d, num_elements);

    // Copy data from device to host
    cudaMemcpy(output_vector_h, output_vector_d, vector_size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(input_matrix_d);
    cudaFree(input_vector_d);
    cudaFree(output_vector_d);
}
```

3. Consider the following CUDA kernel and the corresponding host function that calls it:

```c
__global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N){
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(row < M && col < N){
        b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
    }
}

void foo(float* a_d, float* b_d){
    unsigned int M = 150;
    unsigned int N = 300;
    dim3 bd(16, 32);
    dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);
    foo_kernel <<<gd, bd>>>(a_d, b_d, M, N);
}
```

a. What is the number of threads per block? -> 16 * 32 = 512 threads per block

b. What is the number of threads in the grid? ->  13 * 5 = 65 blocks * 512 threds per block = 33280

c. What is the number of blocks in the grid? -> 13 * 5 = 65 blocks

d. What is the numer of threads that execute the kernel code? -> That will depend on the size of the input pointers, `a_d` and `b_d`. But the max amount of threads that will execute is 33280. The code in the if statement will execute once on a thread for every output element. So, there could be less threads executed if the amount of output elements does not add up to the max thread count of 33280.

4. Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a 1D array. Specify the array index of the matrix element at row 20 and column 10. -> 400 * 20 + 10 = index 8010

5. Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a 1D array in row-major order. Specify the array index of th etensor element at x = 10, y = 20, and z = 5. -> z * (w * h) + y * w + x => 5 * (400 * 500) + 500 * 400 + 400  = 1008010