#include "common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>

#define      COUNT_MASK  (0x07FFFFFFu);
#define  MAX_BLOCK_SIZE  (1024u)
#define  HISTOGRAM_SIZE  (256)
#define  LOG2_WARP_SIZE  (5)
#define       WARP_SIZE  (32)
#define  WARP_PER_BLOCK  (32) // MAX_BLOCK_SIZE / WARP_SIZE

static float createHistogramCpu(std::vector<uint32_t> const& values, std::vector<uint32_t>& histogram);
static float createHistogramGpu(std::vector<uint32_t> const& data, std::vector<uint32_t>& histogram);
static void fillWithNormalDistribution(std::vector<uint32_t>& values, size_t size);
static void createOnce(size_t size);
static void createMany(size_t min_size, size_t max_size, size_t step);
static __global__ void blockHistogram(uint32_t* result, uint32_t const* data, size_t size);
static __global__ void mergeHistogramKernel(uint32_t const* partial_histograms, uint32_t* out_histogram);
static __device__ void addByte(volatile uint32_t* warp_hist, uint32_t index, uint32_t tag);

void Task2()
{
    //createOnce(1024 * 1024);
    createMany(256 * 1024, 512 * 1024, 1024);
}

float createHistogramCpu(std::vector<uint32_t> const& values, std::vector<uint32_t>& histogram)
{
    histogram.resize(HISTOGRAM_SIZE, 0u);
    auto start = std::chrono::high_resolution_clock::now();
    for (auto item : values) {
        ++histogram[item];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return ns.count() / 1.0e6f;
}

float createHistogramGpu(std::vector<uint32_t> const& data, std::vector<uint32_t>& histogram)
{
    int num_partials = data.size() / (WARP_PER_BLOCK * WARP_SIZE);
    auto partial_histograms = thrust::device_vector<uint32_t>(num_partials * HISTOGRAM_SIZE);
    auto raw_partial = thrust::raw_pointer_cast(partial_histograms.data());
    auto gpu_data = thrust::device_vector<uint32_t>(data);
    auto gpu_hist = thrust::device_vector<uint32_t>(HISTOGRAM_SIZE);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    blockHistogram<<<num_partials, MAX_BLOCK_SIZE>>>(raw_partial,
                                                     gpu_data.data().get(),
                                                     data.size());
    mergeHistogramKernel<<<num_partials, HISTOGRAM_SIZE>>>(raw_partial,
                                                           gpu_hist.data().get());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    histogram.resize(HISTOGRAM_SIZE);
    thrust::copy(gpu_hist.begin(), gpu_hist.end(), histogram.begin());
    return ms;
}

void fillWithNormalDistribution(std::vector<uint32_t>& values, size_t size)
{
    values.resize(size);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(HISTOGRAM_SIZE / 2.0 + 1, HISTOGRAM_SIZE / 8.0);
    for (size_t i = 0; i < size; ++i) {
        auto value = -1.0;
        while (value < 0.0 || value >= HISTOGRAM_SIZE) {
            value = distribution(generator);
        }
        values[i] = static_cast<uint32_t>(value);
    }
}

void createOnce(size_t size)
{
    std::vector<uint32_t> values, cpu_histogram, gpu_histogram;
    fillWithNormalDistribution(values, size);

    std::cout << "CPU: " << createHistogramCpu(values, cpu_histogram) << " ms.\n";
    std::cout << "GPU: " << createHistogramGpu(values, gpu_histogram) << " ms.\n";

    std::cout << "Histograms are ";
    if (!std::equal(cpu_histogram.begin(), cpu_histogram.end(),
                    gpu_histogram.begin(), gpu_histogram.end())) {
        std::cout << "not ";
    }
    std::cout << "equals.\n";
    std::ofstream out("hist.txt");
    WriteVector(cpu_histogram, out);
    out.close();
}

void createMany(size_t min_size, size_t max_size, size_t step)
{
    std::vector<float> times_cpu, times_gpu;
    std::vector<size_t> sizes;
    std::vector<uint32_t> values, histogram;
    for (auto size = min_size; size <= max_size; size += step) {
        std::cout << size << ": " << max_size << std::endl;
        sizes.push_back(size);
        fillWithNormalDistribution(values, size);
        times_cpu.push_back(createHistogramCpu(values, histogram));
        times_gpu.push_back(createHistogramGpu(values, histogram));
        system("cls");
    }
    std::ofstream out("times2.txt");
    WriteVector(sizes, out);
    out << ";";
    WriteVector(times_cpu, out);
    out << ";";
    WriteVector(times_gpu, out);
    out.close();
}

__global__ void blockHistogram(uint32_t* result, uint32_t const* data, size_t size)
{
    __shared__ uint32_t shared_hist[HISTOGRAM_SIZE * WARP_PER_BLOCK];  // 8192
    for (size_t i = 0; i < HISTOGRAM_SIZE / WARP_SIZE; ++i) {
        shared_hist[threadIdx.x + i * MAX_BLOCK_SIZE] = 0;
    }
    auto warp_index = threadIdx.x >> LOG2_WARP_SIZE;
    auto idex_shift = warp_index * HISTOGRAM_SIZE;
    auto tag = threadIdx.x << (32 - LOG2_WARP_SIZE);
    __syncthreads();
    auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto thread_count = blockDim.x * gridDim.x;
    for (size_t i = global_tid; i < size; i += thread_count) {
        addByte(shared_hist + idex_shift, data[i], tag);
    }
    __syncthreads();
    if (threadIdx.x >= HISTOGRAM_SIZE) {
        return;
    }
    // TODO: try with all threads
    uint32_t sum = 0;
    for (size_t i = 0; i < WARP_PER_BLOCK; ++i) {
        sum += shared_hist[threadIdx.x + i * HISTOGRAM_SIZE] & COUNT_MASK;
    }
    result[blockIdx.x * HISTOGRAM_SIZE + threadIdx.x] = sum;
}

__global__ void mergeHistogramKernel(uint32_t const* partial_histograms, uint32_t* out_histogram)
{
    uint32_t sum = 0;
    const auto SIZE = HISTOGRAM_SIZE * gridDim.x;
    for (int i = threadIdx.x; i < gridDim.x; i += HISTOGRAM_SIZE) {
        auto index = blockIdx.x + i * HISTOGRAM_SIZE;
        if (index < SIZE) {
            sum += partial_histograms[index];
        }
    }
    __shared__ uint32_t data[HISTOGRAM_SIZE];
    data[threadIdx.x] = sum;
    for (uint32_t stride = HISTOGRAM_SIZE / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride) {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }
    if (threadIdx.x == 0) {
        out_histogram[blockIdx.x] = data[0];
    }
}

__device__ void addByte(volatile uint32_t* warp_hist, uint32_t index, uint32_t tag)
{
    uint32_t count;
    do {
        // прочесть текущее значение счетчика и снять идентификатор нити
        count = warp_hist[index] & COUNT_MASK;
        // увеличить его на единицу и поставить свой идентификатор
        count = tag | (count + 1);
        warp_hist[index] = count; //осуществить запись
    } while (warp_hist[index] != count); // проверить, прошла ли запись
}