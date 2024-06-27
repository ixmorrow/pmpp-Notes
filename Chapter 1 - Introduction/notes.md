# Chapter 1 Notes

## Heterogeneous Parallel Computing

Two main trajectories for designing microprocessors in semiconductor industry:
* multicore trajectory to maintain speed of sequential programs while moving into multiple cores
* many-thread trajectory focuses more on the execution throughput of parallel applications

"Such a large gap in peak performance between multicores and many-threads has amounted to a significant 'electrical potential' buildup"

"When there is more work to do, there is more opportunity to divide the work among cooperating parallel workers, that is, threads."

### Why are CPUs slower than GPUs?

Has to do with the design differences between the two. CPU follows a "latency-oriented design". CPUs chips have components of them that are made explicitly to reduct the execution latency of each individual thread.
* low-latency arithmetic units
* sophisticated operand delivery logic
* large cache memory
s* control logic

These all consume chip are and power that could otherwise be used to provide more arithmetic execution units and memory access channels.

GPUs however were designed for video games and designers were incentivized to optimize for performing a massive number of floating-point operations and memory accesses per video fram in advanced games. This methodology, it turns out, can be applied to areas outside of video games.

The speed of many graphics applications is limited by the rate at which data can be deliverd from the memory system into the processors and vice versa. A GPU must be capable of moving extremely large amounts of data into and out of graphics frame buffers in its DRAM.

Memory bandwidth - throughput of memory accesses

Reducing latency is much more expensive than increasing throughput in terms of power and chip area.

GPUs are designed as parallel, throughput-oriented computing engines, and they will not perform well on some tasks on which CPUs are designed to perform well.

CPUs -> programs that have one or very few threads, CPUs with lower operation latencies can achieve much high performance than GPUs

GPUs -> with higher execution throughput can achieve much higher performance than CPUs

CUDA == Compute Unified Device Architecture -> designed to suport joint CPU-GPU execution of an application.

National Institutes of Health (NIH) refused to fund parallel programming projects for some time bc they thought it was not practical for machines in the hospital setting to run the parallel software.

## Why more speed or parallelism?

If an application includes what we call "data parallelism" it is often possible to achieve a 10x speedup with just a few more hours of work in GPUs than CPUs.

Why do we need to continue to increase the speed of applications?

* Biology research moving into lower orders of magnitude to the molecular level. There are limits to what we can observe with current technology at the molecular level. Can incorporate computational models to simulate the molecular level which will always benefit from more speed in computation and allow us to model more complex systems.

* Video and audio coding and manipulation. HD TV is remarkably better than previous versions of television. Now, we have 4K and even 8K available. This type of improvement is directly reliant on the type of parallel processing that GPUs are great at. Further enhancements in this area will require more parallelism.

* User interfaces improve as computing speeds improve. 

* Digital twins in which physical objects have accurate models in the simulatied space so that stress testing and deterioration prediction can be thoroughly conducted at much lower cost. Realistic modeling and simulation of physics effects are known to demand very large amount sof computing power.

Definition of speedup for an application by computing system A over computing system B is the ratio of the time used to execute the application in system B over the time used to execute the same application in system A.

Example:

Application A takes 10s to execute a task. Application B takes 200s for the same task. THe speedup for the execution by system A over B would be 200/10 => 20X speedup.

The speedup that is achievable by a parallel computing system over a serial computing system depends on the portion of the application that can be parallelized.

Amdahl's Law - the level of speedup that one can achieve through parallel execution can be severely limited by the parallelizable portion of the application.

Another important factor for the achievable level of speedup for applications is how fast data can be accessee from and written to the memory.

## Challenges in parallel programming

It's challenging to design parallel algorithms with the same level of algorithmic (computational) complexity as that of sequential algorithms.

The execution speed of many applications is limited by memory access latency and/or throughput.

* We refer to these applications as memory bound; by contrast, compute bound applications are limited by the number of instructions performed per byte of data.

The execution speed of parallel programs is often more sensitive to the input data characteristics than is the case for their sequential counterparts.

Some applications require threads to collaborate with each other, which requires using synchronization operations such as barriers or atomic operations. These synchronization operations impose overhead on the application because threads will often find themselves waiting for other threads instead of performing useful work.

## Related parallel programming interfaces

MPI -> Message Passing Interface for scalable cluster computing

OpenMP -> for shared memeory multiprocessor systems

# PMPP UIUC Course - Spring 2018 (Lecture 1)

https://www.youtube.com/watch?v=kwtn2NtFIuA

CPUs: Latency oriented design

* high clock frequency

* large caches

* powerful ALU

Clock frequency measured in Hertz (HZ)/ GigaHertz (GH) in modern computers. It represents the number of cycles a CPU can perform per second, or the speed at which a microprocessor executes instructions.

Caches in CPU are meant to convert some memory accesses from DRAM accesses into on chip cache access.

* DRAM access today takes about 500 clock cycles

* L3 cache 50 cycles, L2 15 clock cycles, L1 5 cycles

So any memory accesses you convert to Cache access, you shorten the access latency dramatically.  

GPUs follow a throughput oriented design.

* moderate clock frequency

* small caches, to boost memory throughput

* simple control (no branch prediction, no data forwarding)

* require massive number of threads to tolerate high latency

 GPU runs around 1.1GH clock frequency.

 Applications benefit from both CPU and GPU. CPU is used for sequential parts where latency matters. GPUs for parallel parts where throughput wins.

 * CPUs can be 10x faster than GPUs for sequential code

 * GPUs can be 10x faster than CPUs for parallel code

 nvLink connects CPU and GPU? -> NVIDIA documentation says nvLink is used for direct GPU to GPU connections.
 https://www.nvidia.com/en-us/data-center/nvlink/

 Parallel Programming Workflow:

 1. Identify compute intensive parts of an application

 2. Adopt/create scalable algorithms

 3. Optimize data arrangements to maximize locality

 4. Performance Tuning

 5. Pay attention to code portability, scalability, and maintainability

 Whenever threads start to do different things, we have a phenomena called Divergence. 

 The total amount of time that it take to complete a job is limited by the thread that takes the longest to finsih.

 Data is one of the most important parts of computing. Compares accessing data to sucking a drink out of a straw when what we really want is a gushing flow of water. 

 # PMPP UIUC Course - Spring 2018 (Lecture 2)

  If your data does not support parallel data processing, you will not be able to parallelize the operations on it.

  Each thread within a block is going to be executing a kernel function (SPMD - Single Program Multiple Data ). A CUDA kernel is executed by a grid (array) of threads. 

  * Grid refers to the universe of all of the thread blocks with all of the threads

  Each thread has an index that is used to compute memory addresses and make control decisions.

Thread indexing: Divide the thread array into multiple "blocks".

* threads within a block cooperate via shared memory, atomic operations, and barrier synchronization

* threads in different blocks cooperate less

1D case: index = blockIdx.x * blockDim.x + threadIdx.x

Threads that are outside the bounds of the number of operations you want to do will have some divergence because we typically program the kernel to check that the threadId is within some bound that indicates the number of elements we want to operate on. If the id is outside this bound, then we skip the actual computation within the kernel. The kernel is still called but because we have different behavior, we have introduced Divergence.

The method of copying data from host to device and vice versa is actually a very primitive way of utilizing a GPU. It will slow down your operations a lot if you are doing something as simple as vector addition. For more comlex operations, this may not be the case. The computation speed up may be worth the overhead of copying data between host and device.

Device code can:

* R/W per thread memory registers

* R/W per block global memory

Host code can:

* Transfer data to/from per grid global memory

All of the host code that goes into invoking a kernel on the GPU can have their own verying different performances based on the CPU they are executing on. All of the CUDA specific APIs were implemented at the CPU driver level by NVIDIA, but depending on the type of CPU you are running the code on (x86, Arm, Intel, IBM, Apple Silicon, etc.) the performance of these can vary even if the kernel code is executed on the same NVIDIA GPU.

Once inside a kernel, the very first thing you should do is determine what the unique index i is. 

`i = blockIdx.x * blockDim.x + threadIdx.x`

This is the threads unique position among all of the elements.

`__golbal__` keyword tells compiler that this specific function will be launched with specific parameters and with multiple threads on the GPU.

There is a penalty for using a thread block size that is not a multiple of 32. Doing so would cause you to under utilize hardward.

Q: Does the number of blocks in a grid effect performance? Response: "Slightly"

`ceil(n/256.0)` -> in C, all non float values are truncated. For example, 3.35 will be truncated to 3. By utilizing the `ceil()` function coupled with a denominator that ends in a decimal ensures that the formula returns a float and that that float is rounded up to the nearest whole number. This is required when determining the number of blocks as you cannot have partial blocks and we need enough blocks to fit our dataset.

Doing any type of type casting on the GPU/Device/Kernel can dramatically slow your code down. As a rule of thumb, try to do all type casting on the CPU.

All the thread blocks launched by a kernel are scheduled in what is a called Streaming Multiprocessors. The streaming multiprocessors will be accessing RAM to process the data.

`__global__` -> called on the host and executed on the device. A kernel function must return `void`.

`__device__` -> can be called from the device. Kernel function can call other functions.

`__host__` -> can only be called on the host. by default, all functions are host functions unless otherwise specified.

When your code is compiled, it's passed to the NVCC compiler. This compiler separates your code into device (PTX) and host code.