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

Here are the results from running with a single 4090 GPU.
 - OS: `Ubuntu 20.04`.
 - CUDA: `12.0`.
 - NVIDIA Driver: `525.60.13`.

```
2023-02-10T20:49:56-08:00
Running ./cuda_event_bench
Run on (24 X 5000 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x12)
  L1 Instruction 32 KiB (x12)
  L2 Unified 1280 KiB (x12)
  L3 Unified 30720 KiB (x1)
Load Average: 0.83, 0.58, 0.51
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
---------------------------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations UserCounters...
---------------------------------------------------------------------------------------------
BM_EventCreate<true>                      328 us          328 us         2123 items_per_second=3.04504M/s
BM_EventCreate<false>                     222 us          222 us         3157 items_per_second=4.50653M/s
BM_EventPool<true>                       6.68 us         6.68 us       105980 items_per_second=149.729M/s
BM_EventPool<false>                      6.67 us         6.67 us       103529 items_per_second=149.932M/s
BM_EventRecord_MT<true>/threads:1        1213 us         1213 us          565 items_per_second=824.263k/s
BM_EventRecord_MT<true>/threads:2        1208 us         1730 us          412 items_per_second=578.072k/s
BM_EventRecord_MT<true>/threads:4        1195 us         2121 us          332 items_per_second=471.452k/s
BM_EventRecord_MT<true>/threads:8        2454 us         4569 us          280 items_per_second=218.879k/s
BM_EventRecord_MT<false>/threads:1        175 us          175 us         4005 items_per_second=5.73065M/s
BM_EventRecord_MT<false>/threads:2        323 us          508 us         1368 items_per_second=1.96742M/s
BM_EventRecord_MT<false>/threads:4        437 us         1344 us          540 items_per_second=744.013k/s
BM_EventRecord_MT<false>/threads:8        435 us         2777 us          248 items_per_second=360.064k/s
BM_EventQuery<true>                       100 us          100 us         7078 items_per_second=9.97617M/s
BM_EventQuery<false>                      103 us          103 us         6788 items_per_second=9.75454M/s
BM_StreamWaitEvent<true>                  115 us          115 us         5892 items_per_second=8.67769M/s
BM_StreamWaitEvent<false>                 119 us          119 us         5959 items_per_second=8.39113M/s
BM_EventDestroy<true>/manual_time         135 us         2957 us         5168 items_per_second=7.38311M/s
BM_EventDestroy<false>/manual_time        102 us          399 us         6858 items_per_second=9.79941M/s
```