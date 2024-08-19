# Notes

Global memory of DRAM device is implemented with DRAM.

Data in DRAM cells are stored in small capacitors as 1’s and 0’s, where the presence or absence of an electrical charge is used to determine if it’s a 1 or 0.

“Reading data from a DRAM cell requires the small capacitor to use its tiny electrical charge to drive a highly capacitive line leading to a sensor and set of the sensor’s detection mechanism that determines whether a sufficient amount of charge is present in the capacitor to qualify as a “1”.

- This process takes tens of nanoseconds in a DRAM chip which is in sharp contrast to the subnanosecond clock cycle time of modern computing systems.
- The tiny amount of charge must raise the electric potential of the large capacitance of the long bit line to a sufficiently high level that it can trigger the detection mechanism of the sense amplifier.

Each time a memory access is requested, DRAM access to consecutive memory locations that contain the data requested are made and passed to the sensors. These consecutive locations and accesses are referred to as DRAM bursts.

In CUDA, the most favorable DRAM memory access is made when all threads in a warp access consecutive global memory. In this case, the hardware combines, or coalesces, all these accesses into a consolidated access to consecutive DRAM locations.

For example, if thread A accesses memory location x and thread B accesses memory location x + 1, and so on all of these accesses will be coalesced into a single request for multiple locations when accessing the DRAM.

The main advantage of memory coalescing is that it reduces global memory traffic by combining multiple memory accesses into a single access.

Coalescing requires threads to have similar execution schedules so that their data accesses can be combined into one. Threads in the same warp are perfect candidates because they all execute a load instruction simultaneously by virtue of SIMD instruction.

## Hiding Memory Latency

DRAM systems typically have two more methods of parallel organization: banks and channels.

- Each channel is a memory controller with a bus that connects a set of DRAM banks to the processor.
- data transfer bandwidth of a bus is defined by its width and clock frequency
- Modern double data rate (DDR) buses perform two data transfers per clock cycle

Interleaved data distribution 

Cache memory in GPU devices are designed to coalesce DRAM memory accesses into one if they are requested within a sufficient amount of time of each other.

## UIUC Lecture #6 - DRAM Bandwidth

DRAM is slow.

Every cell in a Dram array is just a transistor. Stays data as 1’s and 0’s by storing a charge for 1.

Sense amplifiers connected to these transistors are responsible for detecting when there is a charge present on any of the transistors.

DDR Core speed - speed at which the sense amplifier can determine what a specific cell/transistor is holding

Interface speed - interface is the line that goes to the controller that actually send the data/bits/electrical charge into the GPU

Over the years, the interface speed has gotten faster and faster while the core speed has remained the same.

Because DRAM takes a long time to access data, every time you must access it it is beneficial to access as much data in dram as possible to limit the amount of trips required. This is called DRAM bursting.

In modern dram systems, the burst is usually about 64 bytes.

The assumption with bursting is that we are accessing consecutive memory locations.

In matrix multiplication for example, since the logic accesses multiple consecutive rows and columns from the input matrices, the hardware is designed to notice this and fetch the data from dram in Bursts so less dram accesses are required.

- seems this only applies to the n accesses

“The threads are where the time localities are across threads not within each.”