#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void checkPrimeKernel (long long start, long long end, bool *results, long long *numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    long long num = start + (idx * 2); 
    

    if(num > end) return;

    bool isPrime = true;
    

    if (num <= 1) {
        isPrime = false;
        return;
    }

    if (num == 2) {
        isPrime = true;
        return;
    }

    if (num % 2 == 0) {
        isPrime = false;
        return;
    }

    if (num > end) {
        return;
    }

    for (long long i = 3; i * i <= num; i += 2) {
        if(num % i == 0) {
            isPrime = false;
            break;
        }
    }

    numbers[idx] = num;
    results[idx] = isPrime;

}


bool cpuPrimeKernel (long long num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;

    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            return false;
        }
    }

    return true;
}



int main() {
    long long start = 237'023LL;
    long long end = 337'025LL;

    int threadsPerBlock = 256;
    int totalNumbers = (end - start) / 2 + 1 ;
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;
    bool *d_results, *h_results;
    long long *d_numbers, *h_numbers; 

cudaMalloc(&d_results, totalNumbers * sizeof(bool));
h_results = new bool[totalNumbers];

cudaMalloc(&d_numbers, totalNumbers * sizeof(long long));
h_numbers = new long long[totalNumbers];

    cudaEvent_t startEvent, stopEvent;

    cudaEventCreate (&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    checkPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_results, d_numbers);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaMemcpy(h_results, d_results, totalNumbers * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_numbers, d_numbers, totalNumbers * sizeof(long long), cudaMemcpyDeviceToHost);


    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

    std::cout << "Time taken on GPU: " << gpuDuration << std::endl;
    for (int i = 0; i < totalNumbers; i++){
        if(h_results[i]) {
            std::cout << h_numbers[i] << " ";
        }

        std::cout << std::endl;
    }

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (long long num = start; num <= end; num += 2) {
        cpuPrimeKernel(num);

    auto endTime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;

    std::cout << "Time on CPU: " << std::fixed << cpuDuration.count() << std::endl;
    std::cout << "speed gain: " << cpuDuration.count() / gpuDuration << std::endl;

    }

    cudaFree(d_numbers);
    delete[] h_numbers;



};
