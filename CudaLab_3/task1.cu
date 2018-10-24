#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>

const size_t SHARED_BLOCK_SIZE = 1024;

__global__ void reduceMin(unsigned* inData, unsigned* outData, size_t size)
{
    __shared__ uint32_t data[SHARED_BLOCK_SIZE];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > size) {
        return;
    }
    data[tid] = inData[i]; // load into shared memory
    __syncthreads();
    for (int s = 1; s < blockDim.x; s <<= 1) {
    //for (int s = 1; s < size; s <<= 1) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            if (data[index + s] < data[index]) {
                data[index] = data[index + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) { // write result of block reduction
        outData[blockIdx.x] = data[0];
    }
}

///////////////////////////////////////////////////////////////////////////////

unsigned getMinCpu(std::vector<unsigned> const& values, float* ms_out);
unsigned getMinGpu(std::vector<unsigned> const& values, float* ms_out);
void fillRandom(std::vector<unsigned>& values, size_t size);

void task1()
{
    std::vector<unsigned> values;
    fillRandom(values, SHARED_BLOCK_SIZE);
    float ms;
    auto min_cpu = getMinCpu(values, &ms);
    std::cout << min_cpu << " " << ms << std::endl;
    auto min_gpu = getMinGpu(values, &ms);
    std::cout << min_gpu << " " << ms << std::endl;
}

unsigned getMinCpu(std::vector<unsigned> const& values, float* ms_out)
{
    auto min_value = values.front();
    auto start = std::chrono::high_resolution_clock::now();
    //auto min_value = *std::min_element(values.begin(), values.end());
    for (auto value: values) {
        if (value < min_value) {
            min_value = value;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (ms_out != nullptr) {
        *ms_out = static_cast<float>(us.count() / 1000.0);
    }
    return min_value;
}

unsigned getMinGpu(std::vector<unsigned> const& values, float* ms_out)
{
    auto gpu_values = thrust::device_vector<unsigned>(values);
    auto out_vector = thrust::device_vector<unsigned>(1u); // TODO: with pointer
    auto raw_gpu_values = thrust::raw_pointer_cast(gpu_values.data());
    auto raw_out = thrust::raw_pointer_cast(out_vector.data());
    // TODO: extend to 2^n size
    // TODO: split array for SHARED_BLOCK_SIZE subarrays and find min in each
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    unsigned block_count = ceil(static_cast<double>(values.size()) / SHARED_BLOCK_SIZE);
    cudaEventRecord(start);
    reduceMin<<<block_count, SHARED_BLOCK_SIZE>>>(raw_gpu_values, raw_out, values.size());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    if (ms_out != nullptr) {
        cudaEventElapsedTime(ms_out, start, end);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    auto cpu_out = thrust::host_vector<unsigned>(out_vector);
    std::cout << std::endl;
    return out_vector.front();
}

void fillRandom(std::vector<unsigned>& values, size_t size)
{
    values.resize(size);
    auto add = size / (rand() % size + 1);
    for (size_t i = 0; i < size; ++i) {
        values[i] = rand() % (2 * size) + add;
    }
}
