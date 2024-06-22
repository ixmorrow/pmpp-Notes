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

Row major layout - where each row of a 2D matrix is stored in memory in contiguous slots, with the rows placed one after the other. This is how C Compilers store 2D matrices.

Column major layout - where the columns of a 2D matrix are stored one after the other in memory in a 1D array fashion. This is used by FORTRAN compilers.

We can extend this to 3D arrays as well. When the array is linearized, each "plane" of the array will be placed in address space one after the other.

Color to Gray scale conversion in CUDA

```c
__global__
void colorToGrayscaleConversion(unsigned char * Pout, unsigned char * Pin, int width, int height){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height){
        // Get 1D offset for the grayscale image
        int grayOffset = row*width + col;
        // One can think of the RGB image having CHANNEL
        // times more columns than the gray scale image
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset]; // Red Value
        unsigned char g = Pin[rgbOffset + 1]; // Green Value
        unsigned char b = Pin[rbgOffset + 2]; // Blue Value
        // Perform the rescaling and store it
        // we multiply by floating point constants
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}
```
## Image Blur: A More Complex Kernel

In real CUDA C programs, threads often perform complex operations on their data and need to cooperate with each other.

Image blurring smoothes out abrupt variation of pixel values while preserving the edges that are essential for recognizing the key features of the image.

Image Blur Kernel

```c
__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;
        // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for(int blurRow=-BLUR_SIZE; blurRow<BLUR_SIZE+1; ++blurRow){
            for(int blurCol=-BLUR_SIZE; blurCol<BLUR_SIZE+1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // Verify we have a valid image pixel
                if(curRow>=0 && curRow<h && curCol>=0 && curCol<w){
                    pixVal += in[curRow*w + curCol];
                    ++pixels; // Keep track of number of pixels in the avg
                }
            }
        }

        // Write our new pixel value out
        out[row*w + col] = (unsigned char) (pixVal/pixels);
    }
}
```

It seems inefficient or unnecessary if we know the number of threads that are required before execution to spin all of the threads in a block up. Why is there not a way to only spin up the number of threads that are required by the kernel function?

* Would like to research this question a little bit.

Image blurring just sets each pixel value to the average of its surrounding pixels.

## Matrix Multiplication

When accessing multidimensional data, programmers will often hae to linearize multidimensional indices into a 1D offset.

## CUDA Mode Lecture on Ch.3

Main topic of this lesson is that we can map multidimensional data to CUDA threads.

Max block size == 1024 threads

`gridDim` - number of blocks in a grid

`blockDim` - number of threads in a block

Grid can be different on each Kernel launch (due to the execution configurations passed into the kernel call)

* typical grids contain thousands to millions of threads

* threads can be scheduled in any order, cannot gurantee any specific order of thread execution (whihch is why it's important each computation a thread handles can be done independently)

