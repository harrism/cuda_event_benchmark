# CUDA Event Benchmarks

Simple benchmarks of `cudaEvent_t` APIs:

 - `cudaEventCreate`
 - `cudaEventRecord`
 - `cudaEventQuery`
 - `cudaStreamWaitEvent`
 - `cudaEventDestroy`
 - A simulated event pool that maintains a list of free events (more of a benchmark of `std::list`
   push/pop for cost comparison to `cudaEventCreate`).

Each test is performed once using default-created events (support timing) and once with events that
do not support timing.

Here are the results from running with a single GPU (`CUDA_VISIBLE_DEVICES` is set to only that GPU)
of an NVIDIA DGX1 (with Tesla V100 GPUs with 32GB each).
 - OS: `Ubuntu 18.04`.
 - CUDA: `10.2`.
 - NVIDIA Driver: `440.64.00`.

```
(cudf_dev_10.2) mharris@dgx02:~/github/cuda_event_benchmark/build$ CUDA_VISIBLE_DEVICES=3 ./cuda_event_bench
2020-06-18T18:57:26-07:00
Running ./cuda_event_bench
Run on (80 X 3600 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x40)
  L1 Instruction 32 KiB (x40)
  L2 Unified 256 KiB (x40)
  L3 Unified 51200 KiB (x2)
Load Average: 1.85, 1.80, 1.17
------------------------------------------------------------------------------------
Benchmark                          Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------
BM_EventCreate<true>             782 us          782 us          923 items_per_second=1.27941M/s
BM_EventCreate<false>            422 us          422 us         1558 items_per_second=2.36813M/s
BM_EventPool<true>              14.6 us         14.6 us        48904 items_per_second=68.6765M/s
BM_EventPool<false>             13.0 us         13.0 us        53373 items_per_second=76.659M/s
BM_EventRecord<true>            2499 us         2499 us          278 items_per_second=400.15k/s
BM_EventRecord<false>            244 us          244 us         2762 items_per_second=4.09725M/s
BM_EventQuery<true>             1046 us         1046 us          707 items_per_second=956.295k/s
BM_EventQuery<false>            1016 us         1016 us          665 items_per_second=984.674k/s
BM_StreamWaitEvent<true>         258 us          258 us         2706 items_per_second=3.88102M/s
BM_StreamWaitEvent<false>        254 us          254 us         2752 items_per_second=3.93252M/s
BM_EventDestroy<true>            121 us          121 us         5894 items_per_second=8.28793M/s
BM_EventDestroy<false>           119 us          119 us         5959 items_per_second=8.4334M/s
```
