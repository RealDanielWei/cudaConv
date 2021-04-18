#include <functional>
void cudaFunction();

void Convolution2D(float* h_in, float* h_out, int width, int height, float* h_kernel, int kernelWidth, int kernelHeight);

using kernelCall = std::function<void()>;
float timing_experiment(kernelCall func);
