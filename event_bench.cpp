/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <iostream>
#include <list>
#include <thread>
#include <vector>

constexpr std::size_t events_size{1000};

template <bool timing = true>
void create_events(std::vector<cudaEvent_t> &events) {
  for (int i = 0; i < events.size(); i++) {
    if (timing)
      cudaEventCreate(&events[i]);
    else
      cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
  }
}

void record_events(std::vector<cudaEvent_t> &events, cudaStream_t stream) {
  for (auto const &e : events)
    cudaEventRecord(e, stream);
}

void query_events(std::vector<cudaEvent_t> &events) {
  for (auto const &e : events)
    cudaEventQuery(e);
}

void await_events(std::vector<cudaEvent_t> &events, cudaStream_t stream) {
  for (auto const &e : events)
    cudaStreamWaitEvent(stream, e, 0);
}

template <typename Container> void destroy_events(Container events) {
  for (auto const &e : events)
    cudaEventDestroy(e);
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
  std::vector<cudaEvent_t> events(events_size);

  // ensure we don't time context load on first benchmark
  cudaFree(0);

  for (auto _ : state) {
    create_events<timing>(events);
  }
  state.SetItemsProcessed(state.iterations() * events_size);

  destroy_events(events);
}
BENCHMARK_TEMPLATE(BM_EventCreate, true)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_EventCreate, false)->Unit(benchmark::kMicrosecond);

// Benchmark cost of moving events between STL containers
// This is to simulate the cost of getting an event from a pool (vs. creating))
template <bool timing = true>
static void BM_EventPool(benchmark::State &state) {
  event_pool<timing> pool;

  for (auto _ : state) {
    for (int i = 0; i < events_size; i++) {
      cudaEvent_t e = pool.get_event();
      pool.return_event(e);
    }
  }
  state.SetItemsProcessed(state.iterations() * events_size);
}
BENCHMARK_TEMPLATE(BM_EventPool, true)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_EventPool, false)->Unit(benchmark::kMicrosecond);

// Benchmark recording events with or without timing, and varying number of
// threads
template <bool timing = true>
static void BM_EventRecord_MT(benchmark::State &state) {
  thread_local std::vector<cudaEvent_t> events(events_size);
  create_events<timing>(events);

  cudaStream_t stream = 0;

  for (auto _ : state) {
    record_events(events, stream);
  }

  if (state.thread_index == 0)
    state.SetItemsProcessed(state.iterations() * events_size * state.threads);

  destroy_events(events);
}
BENCHMARK_TEMPLATE(BM_EventRecord_MT, true)
    ->Unit(benchmark::kMicrosecond)
    ->Threads(1);
BENCHMARK_TEMPLATE(BM_EventRecord_MT, true)
    ->Unit(benchmark::kMicrosecond)
    ->Threads(2);
BENCHMARK_TEMPLATE(BM_EventRecord_MT, true)
    ->Unit(benchmark::kMicrosecond)
    ->Threads(4);
BENCHMARK_TEMPLATE(BM_EventRecord_MT, true)
    ->Unit(benchmark::kMicrosecond)
    ->Threads(8);

BENCHMARK_TEMPLATE(BM_EventRecord_MT, false)
    ->Unit(benchmark::kMicrosecond)
    ->Threads(1);
BENCHMARK_TEMPLATE(BM_EventRecord_MT, false)
    ->Unit(benchmark::kMicrosecond)
    ->Threads(2);
BENCHMARK_TEMPLATE(BM_EventRecord_MT, false)
    ->Unit(benchmark::kMicrosecond)
    ->Threads(4);
BENCHMARK_TEMPLATE(BM_EventRecord_MT, false)
    ->Unit(benchmark::kMicrosecond)
    ->Threads(8);

// Benchmark querying events with or without timing
template <bool timing = true>
static void BM_EventQuery(benchmark::State &state) {
  std::vector<cudaEvent_t> events(events_size);
  create_events<true>(events);

  cudaStream_t streamA{};
  cudaStreamCreate(&streamA);

  record_events(events, streamA);

  for (auto _ : state) {
    query_events(events);
  }
  state.SetItemsProcessed(state.iterations() * events_size);

  destroy_events(events);
  cudaStreamDestroy(streamA);
}
BENCHMARK_TEMPLATE(BM_EventQuery, true)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_EventQuery, false)->Unit(benchmark::kMicrosecond);

// Benchmark querying events with or without timing
template <bool timing = true>
static void BM_StreamWaitEvent(benchmark::State &state) {
  std::vector<cudaEvent_t> events(events_size);
  create_events<true>(events);

  cudaStream_t streamA{}, streamB{};
  cudaStreamCreate(&streamA);
  cudaStreamCreate(&streamB);

  record_events(events, streamA);

  for (auto _ : state) {
    await_events(events, streamB);
  }
  state.SetItemsProcessed(state.iterations() * events_size);

  destroy_events(events);
  cudaStreamDestroy(streamA);
  cudaStreamDestroy(streamB);
}
BENCHMARK_TEMPLATE(BM_StreamWaitEvent, true)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_StreamWaitEvent, false)->Unit(benchmark::kMicrosecond);

// Benchmark destroying events with or without timing
template <bool timing = true>
static void BM_EventDestroy(benchmark::State &state) {
  std::vector<cudaEvent_t> events(events_size);
  create_events<timing>(events);

  cudaStream_t stream = 0;

  record_events(events, stream);

  for (auto _ : state) {
    destroy_events(events);
  }
  state.SetItemsProcessed(state.iterations() * events_size);
}
BENCHMARK_TEMPLATE(BM_EventDestroy, true)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_EventDestroy, false)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
