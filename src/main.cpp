#ifdef USE_CUDA
#include "cudaLib.h"
#endif
#include <iostream>
#include <chrono>

void cpuConv(float* h_in, float* h_out_ref, int width, int height, float* h_kernel, int kernelWidth, int kernelHeight) {
	auto start = std::chrono::high_resolution_clock::now();
	int rx = kernelWidth / 2, ry = kernelHeight / 2;
	for (int ix = 0; ix < width; ++ix) {
		for (int iy = 0; iy < height; ++iy) {
			float value = 0.0;
			for (int xshift = -rx; xshift <= rx; ++xshift) {
				for (int yshift = -ry; yshift <= ry; ++yshift) {
					if (ix + xshift >= 0 && ix + xshift < width && iy + yshift >= 0 && iy + yshift < height) {
						value += h_in[(iy + yshift) * width + ix + xshift] * h_kernel[(ry + yshift) * kernelWidth + (rx + xshift)];
					}
				}
			}
			h_out_ref[iy * width + ix] = value;
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout <<"CPU time :"<<  duration.count() << std::endl;
}

void checkResult(float* ptr1, float* ptr2, int size) {
	std::cout << "Checking Result" << std::endl;
	bool isOK = true;
	for (int i = 0; i < size; i++) {
		float v1 = ptr1[i], v2 = ptr2[i];
		if (isnan(v1) || isnan(v2)) {
			std::cout << "NAN found!" << std::endl;
			isOK = false;
		}
		double err = (abs(v2) < 1e-5) ? abs(v1) : (abs(v1 - v2) / abs(v2));
		if (err > 1e-5) {
			isOK = false;
			break;
		}
	}
	if (isOK) {
		std::cout << "Passed Check!" << std::endl;
	}
	else {
		std::cout << "Failed Check!" << std::endl;
	}
}

void testConv() {
	int kernelWidth = 3;
	int kernelHeight = 5;
	int width = 3000;
	int height = 3000;
	int imageSizeInByte = width * height * sizeof(float);
	int kernelSizeInByte = kernelWidth * kernelHeight * sizeof(float);
	float* h_in = static_cast<float*>(malloc(imageSizeInByte));
	float* h_out = static_cast<float*>(malloc(imageSizeInByte));
	float* h_out_ref = static_cast<float*>(malloc(imageSizeInByte));
	float* h_kernel = static_cast<float*>(malloc(kernelSizeInByte));
	if (h_in == nullptr || h_out == nullptr || h_out_ref==nullptr || h_kernel == nullptr) {
		std::cout << "Malloc failed!" << std::endl;
		exit(0);
	}
	for (int i = 0; i < width * height; i++) { h_in[i] = (float)rand() / (float)RAND_MAX; }
	for (int i = 0; i < width * height; i++) { h_out[i] = 0.0; }
	for (int i = 0; i < width * height; i++) { h_out_ref[i] = 0.0; }
	for (int i = 0; i < kernelWidth * kernelHeight; i++) { h_kernel[i] = (float)rand() / (float)RAND_MAX; }

	Convolution2D(h_in, h_out, width, height, h_kernel, kernelWidth, kernelHeight);
	cpuConv(h_in, h_out_ref, width, height, h_kernel, kernelWidth, kernelHeight);

	checkResult(h_out, h_out_ref, width * height);

	auto func = [h_in, h_out, width, height, h_kernel, kernelWidth, kernelHeight]() {Convolution2D(h_in, h_out, width, height, h_kernel, kernelWidth, kernelHeight); };
	std::cout << "Time used: " << timing_experiment(func) << "ms" << std::endl;

	free(h_in);
	free(h_out);
	free(h_out_ref);
	free(h_kernel);
}


int main() {

#ifdef USE_CUDA
	testConv();
#else
	std::cout << "Cannot find cuda!" << std::endl;
#endif // USE_CUDA

}
