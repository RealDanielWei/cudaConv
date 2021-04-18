#ifdef USE_CUDA
#include "cudaLib.h"
#endif
#include <iostream>

int main() {

#ifdef USE_CUDA
	cudaFunction();
#else
	std::cout << "No cuda found!" << std::endl;
#endif // USE_CUDA

}
