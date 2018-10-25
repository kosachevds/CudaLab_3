#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>

const auto HISTOGRAM_SIZE = 256;

__global__ void histogramKernel( unsigned * result, unsigned * data, int n )
{

}

///////////////////////////////////////////////////////////////////////////////

float fillHistogramCpu(std::vector<unsigned> const& values, std::vector<unsigned>& histogram);
void fillWithNormalDistribution(std::vector<unsigned>& values, size_t size);
void writeVector(std::vector<unsigned> const& values, std::ostream& out);

void task2()
{
    auto values = std::vector<unsigned>();
    fillWithNormalDistribution(values, 1024 * 1024);
    auto histogram = std::vector<unsigned>();
    std::cout << "CPU: " << fillHistogramCpu(values, histogram) << " ms" << std::endl;

    std::ofstream out("hist.txt");
    writeVector(histogram, out);
    out.close();
}

float fillHistogramCpu(std::vector<unsigned> const& values, std::vector<unsigned>& histogram)
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
