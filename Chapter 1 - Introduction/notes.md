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