#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cublas_v2.h>

__global__ void scalarMul(float P, float *Arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        Arr[idx] *= P;
    }
}

int main() {

    float h_P = 2.5;
    float h_Arr[5]= {5.3, 3.4, 2.3, 5.3, 2.4}; 
    float *d_Arr;
    int N = 5;
    int size = N * sizeof(float);

    int threadsPerBlock = 256;
    int Blocks = 1;


    cudaMalloc(&d_Arr, size);

    cudaMemcpy(d_Arr, h_Arr, size, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    scalarMul<<<Blocks, threadsPerBlock>>>(h_P, d_Arr, N);

    cudaEventRecord(stopEvent, 0);

    float gpuDuration = 0;

    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

    cudaMemcpy(h_Arr, d_Arr, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        std::cout << "h_Arr: " << h_Arr[i] << std::endl;
        };

    std::cout << "Computation performed in " << gpuDuration << "ms." << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_Arr);


}
