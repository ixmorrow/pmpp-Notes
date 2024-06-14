void vecAdd(float* A, float* B, float* C), int n {
    int size = n* sizeof(float);
    float *d_a, *d_a, *d_c;

    // Part 1: Allocate device memory for A, B, and C
    // Copy A and B to device memory

    // Part 2: Call kernel - to launch a grid of threads
    // to perform the actual vector addition

    // Part 3: Copy C from the device memory
    // Free device vectors

}