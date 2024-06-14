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

