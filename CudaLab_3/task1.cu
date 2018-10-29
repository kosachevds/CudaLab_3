#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>

const size_t SHARED_BLOCK_SIZE = 1024;

template <typename T>
extern void WriteVector(std::vector<T> const& values, std::ostream& out);

static unsigned getMinCpu(std::vector<unsigned> const& values, float* ms_out);
static unsigned getMinGpu(std::vector<unsigned> const& values, float* ms_out);
static void fillRandom(std::vector<unsigned>& values, size_t size);
static void createOnce(size_t size);
static void createMany(size_t min_size, size_t max_size, size_t step);
static __global__ void reduceMin(unsigned const* inData, unsigned* outData);

void Task1()
{
    //createOnce(8192 * SHARED_BLOCK_SIZE);
    createMany(SHARED_BLOCK_SIZE, 500 * SHARED_BLOCK_SIZE, SHARED_BLOCK_SIZE);
}

unsigned getMinCpu(std::vector<unsigned> const& values, float* ms_out)
{
    //auto min_value = values.front();
    auto start = std::chrono::high_resolution_clock::now();
    auto min_value = *std::min_element(values.begin(), values.end());
    //std::this_thread::sleep_for(std::chrono::microseconds(10LL));
    //for (auto value: values) {
    //    if (value < min_value) {
    //        min_value = value;
    //    }
    //}
    //for (size_t i = 0; i < values.size(); ++i) {
    //    if (values[i] < min_value) {
    //        min_value = values[i];
    //    }
    //}
    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    if (ms_out != nullptr) {
        *ms_out = ns.count() / 1.0e6F;
    }
    return min_value;
}

unsigned getMinGpu(std::vector<unsigned> const& values, float* ms_out)
{
    auto gpu_values = thrust::device_vector<unsigned>(values);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    unsigned block_count = ceil(static_cast<double>(values.size()) / SHARED_BLOCK_SIZE);
    auto out_gpu = thrust::device_vector<unsigned>(block_count);

    cudaEventRecord(start);
    reduceMin<<<block_count, SHARED_BLOCK_SIZE>>>
        (gpu_values.data().get(), out_gpu.data().get());
    cudaEventRecord(end);
    auto error = cudaEventSynchronize(end);
    float ms1;
    cudaEventElapsedTime(&ms1, start, end);

    auto out_cpu = thrust::host_vector<unsigned>(out_gpu);
    float ms2;
    auto min_value = getMinCpu({ out_cpu.begin(), out_cpu.end() }, &ms2);
    if (ms_out != nullptr) {
        *ms_out = ms1 + ms2;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    std::cout << std::endl;
    return min_value;
}

void fillRandom(std::vector<unsigned>& values, size_t size)
{
    values.resize(size);
    auto add = size / (rand() % size + 1);
    for (size_t i = 0; i < size; ++i) {
        values[i] = rand() % (2 * size) + add;
    }
}

void createOnce(size_t size)
{
    std::vector<unsigned> values;
    fillRandom(values, size);
    float ms;
    auto min_cpu = getMinCpu(values, &ms);
    std::cout << "CPU: " << min_cpu << " " << ms << " ms." << std::endl;
    auto min_gpu = getMinGpu(values, &ms);
    std::cout << "GPU: " << min_gpu << " " << ms << " ms." << std::endl;
}

void createMany(size_t min_size, size_t max_size, size_t step)
{
    std::vector<float> times_cpu, times_gpu;
    std::vector<uint32_t> values;
    std::vector<size_t> sizes;
    for (auto size = min_size; size <= max_size; size += step) {
        std::cout << size << ": " << max_size << std::endl;
        sizes.push_back(size);
        fillRandom(values, size);
        float ms;
        auto min_cpu = getMinCpu(values, &ms);
        times_cpu.push_back(ms);
        auto min_gpu = getMinGpu(values, &ms);
        if (min_cpu != min_gpu) {
            ms = -1;
        }
        times_gpu.push_back(ms);
        system("cls");
    }
    std::ofstream out("times1.txt");
    WriteVector(sizes, out);
    out << ";";
    WriteVector(times_cpu, out);
    out << ";";
    WriteVector(times_gpu, out);
    out.close();
}

__global__ void reduceMin(unsigned const* inData, unsigned* outData)
{
    __shared__ unsigned shared [SHARED_BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    shared[tid] = inData[i];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] < shared[tid]) {
                shared[tid] = shared[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        outData[blockIdx.x] = shared[0];
    }
}