#pragma once


__device__ __constant__ unsigned int c_positions;


__global__ void testLutValues1D(cudaTextureObject_t const tex, float* answers, float* xPositions) 
{
	// access thread id, we use blockDim.y also
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	
	float const xPos = xPositions[tid];
	if (tid < c_positions) {
		answers[tid] = tex1D<float>(tex, xPos); 
	}
	return;
};

__global__ void testLutValues2D(cudaTextureObject_t const tex, float* answers, float* xPositions, float* yPositions) 
{
	// access thread id, we use blockDim.y also
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
	
	if (tid < c_positions) {
		float const xPos = xPositions[tid];
		float const yPos = yPositions[tid];
		answers[tid] = tex2D<float>(tex, xPos, yPos); 
	}
	return;
};