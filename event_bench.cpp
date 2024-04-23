/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ex  ess or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <benchmark/benchmark.h>

#include <cuda_runtime_api.h>

#include <list>
#include <thread>

template <bool timing = true>
cudaEvent_t create_event() {
  cudaEvent_t event;
  if (timing)
    cudaEventCreate(&event);
  else
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  return event;
}

void record_event(cudaEvent_t event, cudaStream_t stream) {
  cudaEventRecord(event, stream);
}

void query_event(cudaEvent_t event) {
  cudaEventQuery(event);
}

void await_event(cudaEvent_t event, cudaStream_t stream) {
  cudaStreamWaitEvent(stream, event, 0);
}

void destroy_event(cudaEvent_t event) {
    cudaEventDestroy(event);
}

template <bool timing = true> struct event_pool {
  event_pool() = default;
  ~event_pool() {
    for (auto e : events)
      cudaEventDestroy(e);
  }

  cudaEvent_t get_event() {
    if (!events.empty()) {
      cudaEvent_t e = events.front();
      events.pop_front();
      return e;
    } else {
      return create_event();
    }
  }

  void return_event(cudaEvent_t e) { events.push_back(e); }

private:
  cudaEvent_t create_event() {
    cudaEvent_t e{};
    if (timing)
      cudaEventCreate(&e);
    else
      cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    return e;
  }

  std::list<cudaEvent_t> events{};
};

// Benchmark creating events with or without timing
template <bool timing = true>
static void BM_EventCreate(benchmark::State &state) {
  // ensure we don't time context load on first benchmark
  cudaFree(0);

  for (auto _ : state) {
    // only time event creation
    auto start = std::chrono::high_resolution_clock::now();
    auto event = create_event<timing>();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
    std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());

    destroy_event(event);
  }

  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_TEMPLATE(BM_EventCreate, true)
  ->Unit(benchmark::kNanosecond)
  ->UseManualTime();
BENCHMARK_TEMPLATE(BM_EventCreate, false)
  ->Unit(benchmark::kNanosecond)
  ->UseManualTime();

// Benchmark cost of moving events between STL containers
// This is to simulate the cost of getting an event from a pool (vs. creating))
template <bool timing = true>
static void BM_EventPool(benchmark::State &state) {
  event_pool<timing> pool;

  for (auto _ : state) {
    cudaEvent_t e = pool.get_event();
    pool.return_event(e);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_TEMPLATE(BM_EventPool, true)->Unit(benchmark::kNanosecond);
BENCHMARK_TEMPLATE(BM_EventPool, false)->Unit(benchmark::kNanosecond);

// Benchmark recording events with or without timing, and varying number of threads
template <bool timing = true>
static void BM_EventRecord_MT(benchmark::State &state) {
  cudaStream_t stream = 0;

  for (auto _ : state) {
    auto event = create_event<timing>();

    // only time event creation
    auto start = std::chrono::high_resolution_clock::now();
    record_event(event, stream);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
    std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());

    destroy_event(event);
  }

  if (state.thread_index() == 0)
    state.SetItemsProcessed(state.iterations() * state.threads());

  destroy_events(events);
}
BENCHMARK_TEMPLATE(BM_EventRecord_MT, true)
    ->Unit(benchmark::kNanosecond)
    ->Threads(1)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_EventRecord_MT, true)
    ->Unit(benchmark::kNanosecond)
    ->Threads(2)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_EventRecord_MT, true)
    ->Unit(benchmark::kNanosecond)
    ->Threads(4)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_EventRecord_MT, true)
    ->Unit(benchmark::kNanosecond)
    ->Threads(8)
    ->UseManualTime();

BENCHMARK_TEMPLATE(BM_EventRecord_MT, false)
    ->Unit(benchmark::kNanosecond)
    ->Threads(1)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_EventRecord_MT, false)
    ->Unit(benchmark::kNanosecond)
    ->Threads(2)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_EventRecord_MT, false)
    ->Unit(benchmark::kNanosecond)
    ->Threads(4)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_EventRecord_MT, false)
    ->Unit(benchmark::kNanosecond)
    ->Threads(8)
    ->UseManualTime();

// Benchmark querying events with or without timing
template <bool timing = true>
static void BM_EventQuery(benchmark::State &state) {
  cudaStream_t streamA{};
  cudaStreamCreate(&streamA);

  for (auto _ : state) {
    auto event = create_event<timing>();

    record_event(event, streamA);

    // only time event query
    auto start = std::chrono::high_resolution_clock::now();
    query_event(event);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());

    destroy_event(event);
  }
  state.SetItemsProcessed(state.iterations());

  cudaStreamDestroy(streamA);
}
BENCHMARK_TEMPLATE(BM_EventQuery, true)
  ->Unit(benchmark::kNanosecond)
  ->UseManualTime();
BENCHMARK_TEMPLATE(BM_EventQuery, false)
  ->Unit(benchmark::kNanosecond)
  ->UseManualTime();

// Benchmark querying events with or without timing
template <bool timing = true>
static void BM_StreamWaitEvent(benchmark::State &state) {
  cudaStream_t streamA{}, streamB{};
  cudaStreamCreate(&streamA);
  cudaStreamCreate(&streamB);

  for (auto _ : state) {
    auto event = create_event<timing>();

    record_event(event, streamA);

    // only time event query
    auto start = std::chrono::high_resolution_clock::now();
    await_event(event, streamB);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());

    destroy_event(event);
  }
  state.SetItemsProcessed(state.iterations());

  cudaStreamDestroy(streamA);
  cudaStreamDestroy(streamB);
}
BENCHMARK_TEMPLATE(BM_StreamWaitEvent, true)
  ->Unit(benchmark::kNanosecond)
  ->UseManualTime();
BENCHMARK_TEMPLATE(BM_StreamWaitEvent, false)
  ->Unit(benchmark::kNanosecond)
  ->UseManualTime();

// Benchmark destroying events with or without timing
template <bool timing = true>
static void BM_EventDestroy(benchmark::State &state) {
  cudaStream_t stream = 0;

  for (auto _ : state) {
    auto event = create_event<timing>();

    // only time event destruction
    auto start = std::chrono::high_resolution_clock::now();
    destroy_event(event);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_TEMPLATE(BM_EventDestroy, true)
    ->Unit(benchmark::kNanosecond)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_EventDestroy, false)
    ->Unit(benchmark::kNanosecond)
    ->UseManualTime();

BENCHMARK_MAIN();
