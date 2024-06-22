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