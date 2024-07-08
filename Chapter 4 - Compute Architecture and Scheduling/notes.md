# Compute Architecture and Scheduling

CPUs designed to minimize the latency of instruction execution and GPUs designed to maximize the throughput of executing instructions.

Each Streaming Multiprocessor has several streaming processors or CUDA cores that share control logic and memory resources.

- these SMs come with on-chip memory that will be discussed in detail in Chapter 5
- they also come with off-chip memory called global memory

When a kernel is called, the CUDA runtime system launches a grid of threads to execute the kernel code. These threads are assigned to SMs on a block-by-block basis. All threads in a block are simultaneously assigned to the same SM.

- multiple blocks are likely to be simultaneously assigned to the same SM

There is a limit on the total number of blocks that can be simultaneously executing in a CUDA device. To ensure that all blocks in a grid get executed, the runtime system maintains a list of blocks that need to execute and assigns new blocks to SMs when previously assigned blocks complete execution.

Assignment of blocks to SMs on a block level guarantee that all threads within a given block will execute simultaneously. This guarantee allows threads within a block to interact with each other in ways threads across blocks cannot.

## Synchronization and Transparent Scalability

`__syncthreads()` - function call that allows all threads within a block to sync up within a kernel. When a thread calls this function, it will be held at the program location of the call until every thread in the same block reaches that location. Ensures all threads in a block have completed a phase of their execution before any of them can move on.

In CUDA, if a `__syncthreads()` statement is present, it must be executed by all threads in a block. Doing so will result in undefined execution behavior and either an incorrect result or a deadlock with threads waiting on each other at different points in the program.

- it’s the responsibility of the programmer to avoid such inappropriate use of barrier synchronization

When utilizing barrier synchronization, all threads involved should have access to the necessary resources to eventually arrive at the barrier. Otherwise, a deadlock can occur. The CUDA runtime satisfies this constraint by assigning execution resources to all threads in a block as a unit.

- all threads in a block must be assigned to the same SM SIMULTANEOUSLY
- That means a block can begin execution only when the runtime system has secured all the resourced needed by ALL threads in the block to complete execution

This allows the CUDA runtime system to execute blocks in any order relative to each other since none of them need to wait for each other.

- the more execution resources a GPU has, the more blocks that can be executed simultaneously
- a high end GPU today (at time of writing 2018) can execute hundreds of blocks simultaneously

Transparent Scalability - the ability to execute the same application code on different hardware with different amounts of execution resources, reduces the burden on app developers and improves the usability of applications.

## Warps and SIMD Hardware

The correctness of executing a kernel should not depend on any assumption that certain threads will execute in synchrony with each other without the use of barrier synchronizations.

In most implementations today, once a block has been assigned to an SM, it is further divided into **32-thread units called warps**.

- the size of warps is implementation specific (hardware) and can vary in future generations of GPUs

A warp is the unit of thread scheduling in SMs. -> Each warp consists of 32 threads of consecutive `threadIdx` values.

- if a block is a 1-dimensional block, only the `threadIdx.x` values are used to sort the threads into warps.
- in general, warp n starts with thread 32*n and ends with thread 32*(n+1) - 1
- for a block whose size is not a multiple of 32, the last warp will be padded with inactive threads to fill up the 32 thread positions

For blocks that consist of multiple dimensions, the dimensions will be projected into a linearized row-major layout before partitioning into warps.

- they are sorted linearly by the highest level of dimensionality in descending order

Cores in an SM are grouped into processing blocks, where each block has some cores and an instruction fetch/dispatch unit.

- threads in the same warp are assigned to the same processing block, which fetches the instruction for the warp and executes it for all threads in the warp at the same time. These threads apply the same instruction to different portions of the data.

## Control Divergence

When threads in the same warp follow different execution paths, we say that these threads exhibit control divergence as they diverge in their execution.

- introducing divergence into a kernel forces the CUDA runtime to execute multiple passes through the code.
-> will pass through with threads who take path A and set threads that take path B as inactive, then another pass through with the roles reversed
- cost of divergence is the extra passes the hardware needs to take to allow different threads in a warp to make their own decisions as well as the execution resources that are consumed by the inactive threads in each pass.
- from the Volta architecture onwards, the passes may be executed concurrently meaning that the execution of one pass may be interleaved with the execution of another pass, which is called independent thread scheduling

See Volta V100 Whitepaper here: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf

If a decision condition in a CUDA kernel is based on `threadIdx` values, the control statement can potentially cause thread divergence.

Note that the performance impact of control divergence decreases as the size of the vectors being processed increases.

- this is because only the very last warp will have divergence (in 1 dimensional kernels), so the more warps/threads there are the less impactful that divergence will be

## Warp Scheduling and Latency Tolerance

When threads are assigned to SMs, there are usually more threads assigned to an SM than there are cores in the SM which means each SM has only enough execution units to execute a subset of all the threads assigned to it at any point in time.

In earlier GPU designs, each SM can execute only one instruction for a single warp at any given point in time. In today’s SMs, each one can execute instructions for a small number of warps at any point in time.

- in either case, the SM can execute instructions only for a subset of all warps in the SM

Why are so many warps assigned to an SM if it can execute only a subset of them at any instant? –> That is how GPUs tolerate long-latency operations such as global memory accesses

- when an instruction to be executed by a warp needs to wait for the result of a previously initiated long-latency operation, the warp is not selected for execution.
- instead, another resident warp that is no longer waiting for results of previous instructions will be selected for execution
- often called “latency tolerance” or “latency hiding”

Warp scheduling is also used for tolerating other types of operation latencies (pipelined floating-point arithmetic, branch instructions, etc.). With enough warps around, the hardware will likely find a warp to execute at any point in time, thus making full use of the execution hardware while the instructions of some warps that are ready for execution does not introduce any idle or wasted time into the execution timeline.

- this is referred to as zero-overhead thread scheduling
- With warp scheduling, the long waiting time of warp execution is “hidden” by context switching and executing other warps that are ready to go

This ability to tolerate long operation latencies is the main reason why GPUs do not dedicate nearly as much chip area to cache memories and branch prediction mechanisms as CPUs do. As a result, GPUs can dedicate more chip area to floating-point execution and memory access channel resources.

### Threads, Context-Switching, and Zero-Overhead Scheduling

A thread in modern computers is a program and the state of executing the program on a von Neumann Processor.

- **thread consists of the code of a program, the instruction in the code that is being executed, and value of its variables and data structures**

In Von Neumann based processors, the code of the program is stored in memory (RAM).

- The PC keeps track of the address of the instruction of the program that is being executed
- the Instruction Register (IR) holds the instruction that is being executed
- the register and memory hold the values of the variables and data structures

Modern processors are designed to allow context-switching, where multiple threads can time-share a processor by taking turns to make progress.

By carefully saving and restoring the PC value and the contents of registers and memory, we can suspend the execution of a thread and correctly resume the execution of the thread later!

- however, this process of saving and restoring register contents during context-switching in these processors can incur significant overhead in terms of added execution time

**Zero-overhead scheduling refers to the GPUs ability to put a warp that needs to wait for a long-latency instruction result to sleep and activate a warp that is ready to go without introducing any extra idle cycles in the processing units.**

CPUs incur such a high cost for context-switching because it requires saving the execution state (such as register contents of the out-going thread) to memory and loading the execution state of the incoming thread from memory. **GPU SMs achieves zero-overhead scheduling by holding all the execution states for the assigned warps in the hardware registers so there is no need to save and restore states when switching from one warp to another.**

- Is this method of context-switching only possible on GPUs? Why?

**The oversubscription of threads to SMs is essential for latency tolerance. It increases the chances of finding another warp to execute when accurately executing warp encounters a long-latency operation.**

## Resource Partitioning and Occupancy

**Occupancy** - ratio of the number of warps assigned to an SM to the maximum number it supports

Execution resources in an SM include registers, shared memory, thread block slots, and thread slots.

- these are dynamically partitioned across threads to support their execution

The ability to dynamically partition thread slots among blocks makes SMs versatile. They can either execute many blocks each having few threads or execute few blocks each having many threads.

Example:

The Ampere A100 GPU can support a max of 32 blocks per SM, 64 warps (2048 threads → 32*64) per SM, and 1024 threads per block.

When the max number of threads per block is NOT divisible by the block size, it can negatively impact occupancy by under utilizing the thread slots.

Ampere A100 also allows a max of 65, 536 registers per SM. To run at full occupancy, each SM needs enough registers for 2048 threads (max threads per SM on Ampere A100). That is 65,535/2048 = 32 registers per thread.

- How much data can a single register hold?
    - From the reading, it sounds like each register holds a single variable? But wouldn’t that depend on the variable size in terms of bytes?

**Performance Cliff** - a slight increase in resource usage can result in significant reduction in parallelism and performance achieved

Reader is referred to the CUDA Occupancy Calculator which is a downloadable spreadsheet that calculates the actual number of threads running on each SM for a particular device implementation given the usage of resources by a kernel.

## Querying Device Properties

This raises an important question: How do we find out the amount of resources available for a particular device?

- The amount of resources in each CUDA device SM is specified as part of the *compute capability* of the device. In general, the higher the compute capability level, the more resources are available in each SM.
- In CUDA C, there is a built-in mechanism for the host code to query the properties of the devices that are available in the system.
    - `cudaGetDeviceCount` returns the number of available CUDA devices in the system

```c
int devCount;
cudaGetDeviceCount(&devCount);
```

Use the following to iterate through the available devices and query their properties:

```c
cudaDeviceProp devProp;
for(unsigned int i = 0; i < devCount; i++){
	cudaGetDeviceProperties(&devProp, i);
	// decide if device has sufficient resources/capabilities
}
```

Threads in different blocks CANNOT synchronize with each other.

## UIUC Lecture 4 - Thread Scheduling & Control Branch Divergence

Scheduling is the biggest difference between CPU and GPU architecture. In CPUs, switching from one thread to another incurs a very high performance hit in terms of latency.

In GPUs, there are enough registers for all the threads so you don’t need to save your registers before switching threads. You just move from one section of registers to another.

- in reality, with GPUs we are not just switching threads but changing warps.

When a thread in a Warp goes into execution, all the threads in that warp go into execution.

The reason for all the different limitations on number of blocks and threads a SM can support is that every thread block requires resources. Every thread also has some resource requirements.

Allocating memory with `cudaMalloc` uses global memory.

Each thread can:

- read/write per thread registers (~1 cycle)
- read/write per block shared memory (~5 cycles)
- read/write per block grid global memory (~500 cycles)
- read/only per grid constant memory (~5 cycles with caching)

Registers are “free” and do not incur a performance cost when accessing them.

- very fast, but there are very few of them

Memory is expensive and slow, but there is much more available.

All local variables declared in a kernel function will be placed in registers by default.

- every variable uses a register for each thread that will be running the kernel
    - i.e. 1 variable declared in a kernel function that is launched with a 1000 threads uses 1000 registers for that single variable
    - the SM still has limited capacity, so not every thread that will launch the kernel means a register taken at once

## CUDA Mode Lecture  on Chapter 4

https://www.youtube.com/watch?v=lTmYrKwjSOU

Each SM can execute a single warp at a given time.

Launching a CUDA kernel, you give it:

- block layout (how many threads in a block)
- grid layout (how many blocks should be launched in total)

Threads in a block are executed in parallel on a SM and can access its shared memory.

Blocks are completely independent to us. CUDA runtime is free to assign blocks to SMs as it sees fit.

Not done with a warp until all threads in a warp have completed.