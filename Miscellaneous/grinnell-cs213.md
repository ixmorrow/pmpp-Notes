Found this example practice problem from this course online:
https://curtsinger.cs.grinnell.edu/teaching/2021S2/CSC213/exercises/gpu/intro.html

Practice Kernel:

```c
#include <stdio.h>
#include <cuda_runtime.h>

// saxpy implementation

__global__ void saxpyKernel(float *x, float *y, float a){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < blockDim.x){
        y[i] = a * x[i] + y[i];
    }
}

void saxpy(float* X_h, float* Y_h, int n){
    int size = n* sizeof(float);
    float *X_d, *Y_d;

    // Part 1: Allocate device memory for A, B, and C
    cudaMalloc((void**)&X_d, size);
    cudaMalloc((void**)&Y_d, size);

    cudaError_t cudaError_a = cudaMemcpy(X_d, X_h, size, cudaMemcpyHostToDevice);
    cudaError_t cudaError_b = cudaMemcpy(Y_d, Y_h, size, cudaMemcpyHostToDevice);

    // Part 2: Call kernel to launch grid of threads to perform reduction
    saxpyKernel<<<ceil(n/256.0), 256>>>(X_d, Y_d, n);

    // Part 3: Copy C from the device memory
    cudaError_t cudaError_c = cudaMemcpy(Y_h, Y_d, size, cudaMemcpyDeviceToHost);

    cudaFree(X_d);
    cudaFree(Y_d);
}

int main(){
    // initialize variables on host & device
    int a = 10;
    float *X, *Y;
    X = (float*)malloc(a*sizeof(float));
    Y = (float*)malloc(a*sizeof(float));

    for (int i = 0; i < a; i++) {
        X[i] = i;
        Y[i] = 0.0;
    }

    saxpy(X, Y, a);

    printf("SAXPY result:\n");
    for (int i = 0; i < a; i++){
        printf("%f ", Y[i]);
    }
    printf("\n");

    // Free host memory
    free(X);
    free(Y);

    return 0;
}
```

The distinction between threads and blocks becomes important when we care about interactions between threads. 

`__shared__` annotation means the memory used for this variable is shared between all threads used within the current block

Dot product example using shared memory:

```c
#include <stdio.h>
#include <cuda_runtime.h>

int THREADS_PER_BLOCK = 64;

__global__ void dotProductKernel(float* a, float* b, float* result, int n){
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;

    // Create space for a shared array that all threads in this block will use to store pairwise products
    __shared__ float temp[64];

    if (i < n){
        temp[threadIdx.x] = a[i] * b[i];
    }
    // synchronize threads in this block to ensure all have completed
    // dot product before moving on
    __syncthreads();

    // thread 0 will sum up the products
    if(threadIdx.x == 0){
        float sum = 0.0f;
        // sum up the products
        for (int i = 0; i < blockDim.x; i++){
            sum += temp[i];
        }
        // add the sum of the block to the global result using CUDA atomicAdd()
        atomicAdd(result, sum);
    }
}

void dotProduct(float* a_h, float* b_h, float* result_h, int n){
    int size = n* sizeof(float);
    float *a_d, *b_d, *result_d;

    // allocate device memory for A, B, and result
    cudaMalloc((void**) &a_d, size);
    cudaMalloc((void**) &b_d, size);
    cudaMalloc((void**) &result_d, sizeof(float));

    // copy A and B to device memory
    cudaError_t cudaError_a = cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaError_t cudaError_b = cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // call kernel function
    dotProductKernel<<<ceil(n/float(THREADS_PER_BLOCK)), THREADS_PER_BLOCK>>>(a_d, b_d, result_d, n);

    // copy result from device memory to host memory
    cudaError_t cudaError_c = cudaMemcpy(result_h, result_d, sizeof(float), cudaMemcpyDeviceToHost);

    if(cudaError_a != cudaSuccess && cudaError_b != cudaSuccess && cudaError_c != cudaSuccess){
        printf("Error in cudaMemcpy from Host to Device");
        exit(1);
    }

    // free device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(result_d);
}

int main(){
    // initialize variables on host
    int n = 48;
    float *a, *b, *result;
    a = (float*)malloc(n*sizeof(float));
    b = (float*)malloc(n*sizeof(float));
    result = (float*)malloc(sizeof(float));

    // initialize A and B
    for (int i = 0; i < n; i++){
        a[i] = i;
        b[i] = i + 1;
    }

    // call dotProduct function which launches kernel
    dotProduct(a, b, result, n);

    printf("Dot product result: %f\n", *result);

    // free host memory
    free(a);
    free(b);
    free(result);

    return 0;
}
```