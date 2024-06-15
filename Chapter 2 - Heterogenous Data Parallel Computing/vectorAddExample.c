%%cuda
#include <stdio.h>
#include <cuda_runtime.h>

// Compute vector sum C = A + B
// Each thread performs a pairwise addition
__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n* sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1: Allocate device memory for A, B, and C
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy A and B to device memory
    cudaError_t cudaError_a = cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaError_t cudaError_b = cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    if(cudaError_a != cudaSuccess && cudaError_b != cudaSuccess){
        printf("Error in cudaMemcpy from Host to Device");
        exit(1);
    }
    
    // Part 2: Call kernel - to launch a grid of threads
    // to perform the actual vector addition
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // Part 3: Copy C from the device memory
    cudaError_t cudaError_c = cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    if(cudaError_c != cudaSuccess){
        printf("Error in cudaMemcpy from Device to Host");
        exit(1);
    }

    // Free device vectors
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    int n = 10;
    float *A, *B, *C;
    A = (float*)malloc(n * sizeof(float));
    B = (float*)malloc(n * sizeof(float));
    C = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        A[i] = 5.0f * i;
        B[i] = 5.0f + i;
    }

    vecAdd(A, B, C, n);
    
    printf("C vector addition result:\n");
    for (int i = 0; i < n; i++){
        printf("%f ", C[i]);
    }
    printf("\n");

    // Free host memory
    free(A);
    free(B);
    free(C);
}