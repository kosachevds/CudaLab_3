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
    __shared__ unsigned shared [SHARED_BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i + i + blockDim.x < size && inData[i + blockDim.x] < inData[i]) {
        shared[tid] = inData[i + blockDim.x];
    } else {
        shared[tid] = inData[i];
    }
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

///////////////////////////////////////////////////////////////////////////////

unsigned getMinCpu(std::vector<unsigned> const& values, float* ms_out);
unsigned getMinGpu(std::vector<unsigned> const& values, float* ms_out);
void fillRandom(std::vector<unsigned>& values, size_t size);

void task1()
{
    std::vector<unsigned> values;
    fillRandom(values, 8192 * SHARED_BLOCK_SIZE);
    float ms;
    auto min_cpu = getMinCpu(values, &ms);
    std::cout << "CPU: " << min_cpu << " " << ms << std::endl;
    auto min_gpu = getMinGpu(values, &ms);
    std::cout << "GPU: " << min_gpu << " " << ms << std::endl;
}

unsigned getMinCpu(std::vector<unsigned> const& values, float* ms_out)
{
    auto min_value = values.front();
    auto start = std::chrono::high_resolution_clock::now();
    //auto min_value = *std::min_element(values.begin(), values.end());
    //std::this_thread::sleep_for(std::chrono::microseconds(10LL));
    //for (auto value: values) {
    //    if (value < min_value) {
    //        min_value = value;
    //    }
    //}
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] < min_value) {
            min_value = values[i];
        }
    }
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
    auto out_ptr = thrust::device_malloc<unsigned>(1);
    auto raw_gpu_values = thrust::raw_pointer_cast(gpu_values.data());
    auto raw_out = thrust::raw_pointer_cast(out_ptr);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float ms = 0;
    if (ms_out != nullptr) {
        *ms_out = 0.0F;
    }
    auto min_value = values.front();
    for (size_t i = 0; i < values.size(); i += SHARED_BLOCK_SIZE) {
        cudaEventRecord(start);
        reduceMin<<<1u, SHARED_BLOCK_SIZE>>>
            (raw_gpu_values + i, raw_out, SHARED_BLOCK_SIZE);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
        if (ms_out != nullptr) {
            *ms_out += ms;
        }
        auto block_min = out_ptr[0];
        if (block_min < min_value) {
            min_value = block_min;
        }
    }
    thrust::device_free(out_ptr);
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
