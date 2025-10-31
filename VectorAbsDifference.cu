#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

__global__ void vectorAbsDiff(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = sqrtf((A[idx] - B[idx]) * (A[idx] - B[idx]));
    }
}


int main() {

    float h_A[5]= {5.3, 3.4, 2.3, 5.3, 2.4}; 
    float h_B[5]= {3.5, 1.7, 2.5, 5.8, 0.4};
    float h_C[5];
    float *d_A, *d_B, *d_C;
   
    int N = 5;
    int size = N * sizeof(float);

    int threadsPerBlock = 256;
    int Blocks = (N + threadsPerBlock - 1)/ threadsPerBlock;


    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    vectorMul<<<Blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent); 
    

    float gpuDuration = 0;

    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    for (int i = 0; i < N; ++i) {
        std::cout << "C[" << i << "] =" << h_C[i] << std::endl;
        };

    std::cout << "Computation performed in " << gpuDuration << "ms." << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}
