#include <stdio.h>
#include <stdlib.h> 
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaLib.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


float timing_experiment(kernelCall func) {
    float time_ms = 0.0;
    int N_rep = 5;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    for (int i = 0; i < N_rep; ++i) {
        func();
    }
    cudaEventRecord(end, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time_ms, start, end);
    time_ms /= N_rep;
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return time_ms;
}

__global__ void checkId(){
	printf("threadIdx: (%d, %d, %d)	blockIdx: (%d, %d, %d)	blockDim: (%d, %d, %d) gridDim: (%d, %d, %d)\n", 
	threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
	blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z
	);
}

void printDeviceInfo() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}

__global__ void general2DConvolution(float* d_in, float* d_out, int width, int height, float* d_kernel, int kernelWidth, int kernelHeight) {
    int rx = kernelWidth / 2, ry = kernelHeight / 2;
    for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < height; iy += blockDim.y * gridDim.y) {
        for (int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < width; ix += blockDim.x * gridDim.x) {
            float value = 0.0;
            for (int yshift = -ry; yshift <= ry; ++yshift) {
                for (int xshift = -rx; xshift <= rx; ++xshift) {
                    if (ix + xshift >= 0 && ix + xshift < width && iy + yshift >= 0 && iy + yshift < height) {
                        value += d_in[(iy + yshift) * width + (ix + xshift)] * d_kernel[(ry + yshift) * kernelWidth + (rx + xshift)];
                    }
                }
            }
            d_out[iy * width + ix] = value;
        }
    }
}

void Convolution2D(float* h_in, float* h_out, int width, int height, float* h_kernel, int kernelWidth, int kernelHeight) {
    if (!(kernelWidth % 2 == 1 && kernelHeight % 2 == 1)) {
        std::cout << "Invaid input: Input kernel must have odd size!" << std::endl;
        return;
    }
    float* d_in, * d_out, * d_kernel;
    int imageSizeInByte = width * height * sizeof(float), kernelSizeInByte = kernelWidth * kernelHeight * sizeof(float);
    //allocate device memory
    gpuErrchk(cudaMalloc((void**)&d_in, imageSizeInByte));
    gpuErrchk(cudaMalloc((void**)&d_out, imageSizeInByte));
    gpuErrchk(cudaMalloc((void**)&d_kernel, kernelSizeInByte));
    //copy data from host to device
    gpuErrchk(cudaMemcpy(d_in, h_in, imageSizeInByte, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_out, h_out, imageSizeInByte, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_kernel, h_kernel, kernelSizeInByte, cudaMemcpyHostToDevice));
    //call the kernel
    dim3 blockSize(32, 32);
    dim3 gridSize(16, 16);
    general2DConvolution << <gridSize, blockSize >> > (d_in, d_out, width, height, d_kernel, kernelWidth, kernelHeight);
    //copy data back
    gpuErrchk(cudaMemcpy(h_out, d_out, imageSizeInByte, cudaMemcpyDeviceToHost));
    //free memory
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
    gpuErrchk(cudaFree(d_kernel));
}

