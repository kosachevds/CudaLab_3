#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>

const auto HISTOGRAM_SIZE = 256;

#define              N  (6*1024*1024)
#define       NUM_BINS  (256)  // число счетчиков в гистограмме
#define LOG2_WARP_SIZE  (5)    // логарифм размера warp's по основанию 2
#define      WARP_SIZE  (32)   // Размер warp'а
#define         WARP_N  (6)    // Число warp'ов в блоке

__device__ inline void addByte(volatile unsigned* warp_hist, unsigned data, unsigned ttag)
{
    unsigned count;
    do {
        // прочесть текущее значение счетчика и снять идентификатор нити
        count = warp_hist[data] & 0x07FFFFFFU;
        // увеличить его на единицу и поставить свой идентификатор
        count = ttag | (count + 1);
        warp_hist[data] = count; //осуществить запись
    } while (warp_hist[data] != count); // проверить, прошла ли запись
}


__global__ void histogramKernel(unsigned* result, unsigned const* data, size_t n)
{
    __shared__ unsigned hist[NUM_BINS * WARP_N]; //1536 элементов
    // очистить счетчики гистограмм
    for (int i = 0; i < NUM_BINS / WARP_SIZE; i++)
        hist[threadIdx.x + i * WARP_N * WARP_SIZE/*число нитей в блоке=192*/] = 0;
    int warp_base = (threadIdx.x >> LOG2_WARP_SIZE) * NUM_BINS;
    unsigned ttag = threadIdx.x << (32 - LOG2_WARP_SIZE); // получить id для данной нити
    __syncthreads();
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    for (int i = global_tid; i < n; i += numThreads) {
        unsigned data4 = data[i];
        addByte(hist + warp_base, (data4 >> 0) & 0xFFU, ttag);
        addByte(hist + warp_base, (data4 >> 8) & 0xFFU, ttag);
        addByte(hist + warp_base, (data4 >> 16) & 0xFFU, ttag);
        addByte(hist + warp_base, (data4 >> 24) & 0xFFU, ttag);
    }
    __syncthreads();
    // объединить гистограммы данного блока и записать результат в глобальную память
    // 192 нити суммируют данные до 256 элементов гистограмм
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += (WARP_N * WARP_SIZE)) {
        unsigned sum = 0;
        for (int i = 0; i < WARP_N; i++)
            sum += hist[bin + i * NUM_BINS] & 0x07FFFFFFU;
        result[blockIdx.x * NUM_BINS + bin] = sum;
    }
}

// объединить гистограммы, один блок на каждый NUM_BINS элементов
__global__ void mergeHistogramKernel(unsigned* out_histogram, unsigned* partial_histograms, int histogram_count)
{
    unsigned sum = 0;
    for (int i = threadIdx.x; i < histogram_count; i += 256)
        sum += partial_histograms[blockIdx.x + i * NUM_BINS];
    __shared__ unsigned data[NUM_BINS];
    data[threadIdx.x] = sum;
    for (unsigned stride = NUM_BINS / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride)
            data[threadIdx.x] += data[threadIdx.x + stride];
    }
    if (threadIdx.x == 0)
        out_histogram[blockIdx.x] = data[0];
}

///////////////////////////////////////////////////////////////////////////////

float createHistogramCpu(std::vector<unsigned> const& values, std::vector<unsigned>& histogram);
void fillWithNormalDistribution(std::vector<unsigned>& values, size_t size);
void writeVector(std::vector<unsigned> const& values, std::ostream& out);
void createHistogram(unsigned* histogram, unsigned const* data, size_t n);

void task2()
{
    auto values = std::vector<unsigned>();
    fillWithNormalDistribution(values, 1024 * 1024);
    auto cpu_histogram = std::vector<unsigned>();
    createHistogramCpu(values, cpu_histogram);
    //std::cout << "CPU: " << createHistogramCpu(values, cpu_histogram) << " ms" << std::endl;
    auto gpu_histogram = std::vector<unsigned>(HISTOGRAM_SIZE);
    createHistogram(gpu_histogram.data(), values.data(), values.size());
    std::cout << std::equal(cpu_histogram.begin() + 1, cpu_histogram.end(),
                            gpu_histogram.begin() + 1, gpu_histogram.end()) << std::endl;
    //std::ofstream out("hist.txt");
    //writeVector(cpu_histogram, out);
    //out.close();
}

float createHistogramCpu(std::vector<unsigned> const& values, std::vector<unsigned>& histogram)
{
    auto start = std::chrono::high_resolution_clock::now();
    histogram.resize(HISTOGRAM_SIZE, 0u);
    for (auto item : values) {
        ++histogram[item];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return ns.count() / 1.0e6f;
}

void fillWithNormalDistribution(std::vector<unsigned>& values, size_t size)
{
    values.resize(size);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(HISTOGRAM_SIZE / 2.0 + 1, HISTOGRAM_SIZE / 8.0);
    for (size_t i = 0; i < size; ++i) {
        auto value = -1;
        while (value < 0.0 || value >= HISTOGRAM_SIZE) {
            value = distribution(generator);
        }
        values[i] = static_cast<unsigned>(value);
    }
}

void writeVector(std::vector<unsigned> const& values, std::ostream& out)
{
    for (auto item : values) {
        out << item << " ";
    }
    out << std::endl;
}

void createHistogram(unsigned* histogram, unsigned const* data, size_t n)
{
    //int num_blocks = n / (WARP_N * WARP_SIZE);
    int num_partials = 240;
    unsigned* partial_histograms = nullptr;
    //unsigned h[NUM_BINS] = {0};
    //int* pdata = (int*)data;
    //выделить память под гистограммы блока
    cudaMalloc(&partial_histograms, num_partials * NUM_BINS * sizeof(unsigned));
    unsigned* gpu_data;
    cudaMalloc(&gpu_data, n * sizeof(unsigned));
    cudaMemcpy(gpu_data, data, n * sizeof(unsigned), cudaMemcpyHostToDevice);
    // построить гистограмму для каждого блока
    histogramKernel<<<num_partials, WARP_N * WARP_SIZE>>>(partial_histograms, gpu_data, n);

    unsigned* gpu_hist;
    cudaMalloc(&gpu_hist, NUM_BINS * sizeof(unsigned));
    //объдинить гистограммы отдельных блоков вместе
    mergeHistogramKernel<<<NUM_BINS, 256>>>(gpu_hist, partial_histograms, num_partials);
    cudaMemcpy(histogram, gpu_hist, NUM_BINS * sizeof(unsigned), cudaMemcpyDeviceToHost);
    // освободить выделенную память
    cudaFree(partial_histograms);
    cudaFree(gpu_data);
    cudaFree(gpu_hist);
}
