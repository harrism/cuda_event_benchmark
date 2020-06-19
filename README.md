# CUDA Event Benchmarks

Simple benchmarks of `cudaEvent_t` APIs:

 - `cudaEventCreate`
 - `cudaEventRecord`
 - `cudaEventQuery`
 - `cudaEventDestroy`
 - A simulated event pool that maintains a list of free events (more of a benchmark of `std::list` push/pop for cost comparison to `cudaEventCreate`).
