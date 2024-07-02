# Compute Architecture and Scheduling

CPUs designed to minimize the latency of instruction execution and GPUs designed to maximize the throughput of executing instructions.

Each Streaming Multiprocessor has several streaming processors or CUDA cores that share control logic and memory resources.

* these SMs come with on-chip memory that will discussed in detail in Chapter 5

* they also come with off-chip memory called global memory

When a kernel is called, the CUDA runtime system launches a grid of threads to execute the kernel code. These threads are assigned to SMs on a block-by-block basis. All threads in a block are simultaneously assigned to the same SM. 

* multiple blocks are likely to be simultaneously assigned to the same SM

There is a limit on the total number of blocks that can be simultaneously executing in a CUDA device. To ensure that all blocks in a grid get executed, the runtime system maintains a list of blocks that need to execute and assigns new blocks to SMs when previously assigned blocks complete execution.

Assignment of blocks to SMs on a block level gurantee that all threads within a given block will execute simultaneously. This gurantee allows threads within a block to interact with eachother in ways threads across blocks cannot.

## Synchronization and Transparent Scalability

`__syncthreads()` - function call that allows all threads within a block to sync up within a kernel. When a thread calls this function, it will be held at the program location of the call until every thread in the same block reaches that location. Ensures all threads in a block have completed a phase of their execution before any of them can move on.

In CUDA if a `__syncthreads()` statement is present, it must be executed by all threads in a block. Doing so will result in undefined execution behavior and either an incorrect result or a deadlock with threas waiting on eachother at different points in the program.

* it's the responsibility of the programmer to avoid such inappropriate use of barrier synchronization

When utilizing barrier syncronization, all threads involved should have access to the necessary resources to eventually arrive at the barrier. Otherwise, a deadlock can occur. The CUDA runtime satisfies this constraint by assigning execution resources to all threads in a block as a unit. 

* all threads in a block must be assigned to the same SM SIMULTANEOUSLY

* That means a block can begin execution only when the runtime system has secured all the resourced needed by ALL threads in the block to complete execution

This allows the CUDA runtime system to execute blocks in any order relative to each other since none of them need to wait for each other.

* the more execution resources a GPU has, the more blocks that can be executed simultaneously

* a high end GPU today (at time of writing 2018) can execute hundreds of blocks simultaneously

Transparent Scalability - the ability to execute the same application code on different hardware wiht different amounts of execution resources, reduces the burden on app developers and improves the usability of applications.

## Warps and SIMD Hardware

The correctness of executing a kernel should not depend on any assumption that certain threads will execute in synchrony with each other without the use of barrier synchronizations.

In most implementations today, once a blokc has been assigned to an SM, it is further divided into 32-thread units called warps.

* the size of warps is implementation specific (harware) and can vary in future generations of GPUs

A wapr is the unit of thread scheduling in SMs. -> Each warp consists of 32 threads of consecutive `threadIdx` values.

* if a block is a 1-dimensional block, only the `threadIdx.x` values are used to sort the threads into warps.

* in general, warp n starts with thread 32*n and ends with thread 32*(n+1) - 1

* for a block whose size is not a multiple of 32, the last warp will be padded with inactive threads to fill up the 32 thread positions

For blocks that consist of multiple dimensions, the dimensions will be projected into a linearized row-major layout before partitioning into warps.

* they are sorted linearly by the highest level of dimensionality in descending order

Cores in an SM are grouped into processing blocks, where each block has some cores and an instruction fetch/dispatch unit.

* threads in the same warp are assigned to the same processing block, which fetches the instruction for the warp and executes it for all threads in the warp at the same time. These threads appliy the same instruction to different portions of the data.

## Control Divergence

When threads in the same warp follow different execution paths, we say that these threads exhibit control divergence as they diverge in their execution.

* introducing divergence into a kernel forces the CUDA runtime to execute multiple passes through the code.
-> will pass through with threads who take path A and set threads that take path B as inactive, then another pass through with the roles reversed

* cost of divergence is the extra passes the hardware needs to take to allow different threads in a warp to make their own decisions as well as the execution resources that are consumed by the inactive threads in each pass.

* from the Volta architecture onwards, the passes may be executed concurrently meaning that the execution of one pass may be interleaved with the execution of another pass, which is called independent thread scheduling

See Volta V100 Whitepaper here: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf

If a decision condition in a CUDA kernel is based on `threadIdx` values, the control statement can potentially cause thread divergence.

Note that the performance impact of control divergence decreases as the size of the vectors being processed increases.

* this is because only the very last warp will have divergence (in 1 dimensional kernels), so the more warps/threads there are the less impactful that divergence will be

