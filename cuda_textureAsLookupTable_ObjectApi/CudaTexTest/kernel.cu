
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <stdint.h>

#include "kernels.cuh"

#define BLOCK 128

// Convenience error printing function
#define my_cuda_safe_call(err) __my_cuda_safe_call(err, __FILE__, __LINE__);
bool __my_cuda_safe_call(cudaError_t err, const char *file, const int line)
{
	if (cudaSuccess != err) {
		std::cout << "(GPU) Error " << cudaGetErrorString(err) << " at " << file << " " << line << " raw " << err << std::endl;
		return false;
	}
	return true;
};

// allocates and loads linear float buffers for the user
template <typename T>
void setupBuffer(T *const& buffer, unsigned int sizeInUnits, T const*const bufferToCopy = nullptr)
{
	// malloc result buffer
	my_cuda_safe_call(cudaMalloc( (void**) &buffer, sizeInUnits*sizeof(float)));
	if( bufferToCopy == nullptr ) {
		// blank start, cant use memset
		T* ones = (T*) malloc (sizeInUnits*sizeof(T));
		for (unsigned i=0; i<sizeInUnits; i++)
			ones[i] = 1.0f;
		my_cuda_safe_call(cudaMemcpy( buffer, ones, sizeInUnits*sizeof(T), cudaMemcpyHostToDevice));
		free(ones); ones = nullptr;
	}
	else {
		my_cuda_safe_call(cudaMemcpy( buffer, bufferToCopy, sizeInUnits*sizeof(T), cudaMemcpyHostToDevice));
	}
};


// convenience printing function
template <typename T>
void printArray(T const*const inputArray, unsigned int length)
{
	for(unsigned k=0;k<length-1;++k) {
		std::cout << inputArray[k] << ", ";
	}
	std::cout << inputArray[length-1] << std::endl;
}


// Implements a 2D texture with lookup using the texture object API
//http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
int main(void)
{
	// IGNORE 1D FOR NOW
	/*
    const unsigned int array1DSize = 5;
	const float array1D[array1DSize] =           {1.0, 2.0, 3.0, 4.0, 5.0 };
	const float array1DXPositions[array1DSize] = {0.5, 1.5, 2.5, 3.5, 4.5};
	*/
	const unsigned int arrayNumX = 5;

	// for now just a 2D test
	const unsigned int array2DSize = 20;
	const unsigned int arrayNumY = 4;
	const float array2D[array2DSize] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
	// See this guide concerning indexing to get the points you want with textures!
	// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching
	const float array2DXPositions[array2DSize] = {0.5, 1.5, 2.5, 3.5, 4.5,   0.5, 1.5, 2.5, 3.5, 4.5,   0.5, 1.5, 2.5, 3.5, 4.5,   0.5, 1.5, 2.5, 3.5, 4.5};
	const float array2DYPositions[array2DSize] = {0.5, 0.5, 0.5, 0.5, 0.5,   1.5, 1.5, 1.5, 1.5, 1.5,   2.5, 2.5, 2.5, 2.5, 2.5,   3.5, 3.5, 3.5, 3.5, 3.5};

	// IGNORE 3D for now
	/*
	const unsigned int array3DSize = 60;
	const unsigned int arrayNumZ = 3;
	const float array3D[array3DSize] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
										21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
										41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60};
	*/
    
	// pick your cuda device!
	my_cuda_safe_call( cudaSetDevice( 0 ) );
	
	// setup and load textures
	cudaChannelFormatDesc texLUT_desc = cudaCreateChannelDesc<float>(); // we are loading floats
	cudaArray* texLUT_cuarr;
	my_cuda_safe_call( cudaMallocArray( &texLUT_cuarr, &texLUT_desc, arrayNumX, arrayNumY ) ); // note sizeX and sizeY are in # of data points, not bytes!
	int arrLUT_size = array2DSize*sizeof(float);
	my_cuda_safe_call(cudaMemcpyToArray( texLUT_cuarr, 0, 0, array2D, arrLUT_size, cudaMemcpyHostToDevice)); // size here is in BYTES!
    // Build resource descriptor
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = texLUT_cuarr;
    // Build texture object
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
	// Address Modes are described in the 6th bullet point here
	// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory
	texDesc.addressMode[0]   = cudaAddressModeBorder;
    texDesc.addressMode[1]   = cudaAddressModeBorder;
	// Filter Mode is described in the 7th bullet point from the link above
	texDesc.filterMode       = cudaFilterModeLinear;
	// Read Mode is described in the 4th bullet point in the link above
    texDesc.readMode         = cudaReadModeElementType;
	// Will you index into the texture using a value from [0,N) or [0,1] ?
    texDesc.normalizedCoords = 0;
	cudaTextureObject_t texObj = std::numeric_limits<uint64_t>::max(); // this is just an unsigned long long
    my_cuda_safe_call( cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL) );
	// As we are using the first class texture object API, we get a textureIndex back from the create call, we need to hang on to this!
	std::cout << "created texture-id: " << texObj << std::endl;

	// setup execution parameters
	// Why BLOCK == 128?  These numbers would come from tuning your kernel and optimizing it.
	// For this simple test I simply took the value I was using in my most recent side project.
	dim3 threads(BLOCK, 1, 1);
	unsigned int blocksX = 1;
	unsigned int blocksY = 1;
	dim3 grid( blocksX, blocksY, 1);
	
	// Create the arrays on the device and load data when needed
	float* device_answers;
	float* device_xPositions;
	float* device_yPositions;
	setupBuffer(device_answers, array2DSize);
	setupBuffer(device_xPositions, array2DSize, array2DXPositions);
	setupBuffer(device_yPositions, array2DSize, array2DYPositions);
	
	// copy in a value to a constant.  This way the kernels can ensure they don't write to an array position out of bounds
	my_cuda_safe_call(cudaMemcpyToSymbol(c_positions, &array2DSize, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
	// !!! LAUNCH !!!
	testLutValues2D <<< grid, threads >>>( texObj, device_answers, device_xPositions, device_yPositions );
	// ensure all is good in the world
	cudaDeviceSynchronize();
	// Get the answer back
	float cpu_answers[array2DSize];
	my_cuda_safe_call(cudaMemcpy( cpu_answers, device_answers, array2DSize*sizeof(float), cudaMemcpyDeviceToHost)); // size here is in BYTES!

	std::cout << "input:" << std::endl;
	printArray(array2D, array2DSize);
	std::cout << "output:" << std::endl;
	printArray(cpu_answers, array2DSize);

	// FINISHED, clean up
	// NVIDIA sample code says to call cudaDeviceReset to ensure profiling data is accurate
	my_cuda_safe_call( cudaDeviceReset() );
	
    return EXIT_SUCCESS;
};