#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpuErrchk.h"

using namespace std;

constexpr size_t ELEMENTS = 100;

__global__ void simple_multiply(float *output_c, unsigned int width_A, unsigned int height_A, unsigned int width_B, unsigned int height_B, float *input_a, float *input_b)
{
	// Get global position in Y direction
	unsigned int row = (blockIdx.y * 10) + threadIdx.y;
	// Get global position in X direction
	unsigned int col = (blockIdx.x * 10) + threadIdx.x;

	float sum = 0.0f;

	// Calculate result of one element of the output matrix (C)
	for (unsigned int i = 0; i < width_A; ++i)
	{
		float sum_addition = input_a[row * width_A + i] * input_b[i * width_B + col];

		// CUDA printf: Output debugging information for each thread
		printf("sum += input_a[%d * %d + %d] (%f) * input_b[%d * %d + %d] (%f); -> sum += %f;\n",
			row, width_A, i, input_a[row * width_A + i],
			i, width_B, col, input_b[i * width_B + col]), sum_addition;

		sum += sum_addition;
	}

	// CUDA printf: Output debugging information for each thread
	printf("output_c[%d * %d + %d] = %f\n", row, width_B, col, sum);

	// Write result to output matrix
	output_c[row * width_B + col] = sum;
}

int main(int argc, char **argv)
{
	// Create host memory
	auto data_size = sizeof(int) * ELEMENTS;
	vector<float> A(ELEMENTS);    // Input array
	vector<float> B(ELEMENTS);    // Input array
	vector<float> C(ELEMENTS);    // Output array

	// Initialise input data
	for (unsigned int i = 0; i < ELEMENTS; ++i)
	{
		printf("A[%d] = B[%d] = %d\n", i, i, i);
		A[i] = B[i] = i;
	}

	// Declare buffers
	float *buffer_A, *buffer_B, *buffer_C;

	// Initialise buffers
	cudaMalloc((void**)&buffer_A, data_size);
	cudaMalloc((void**)&buffer_B, data_size);
	cudaMalloc((void**)&buffer_C, data_size);

	// Write host data to device
	cudaMemcpy(buffer_A, &A[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_B, &B[0], data_size, cudaMemcpyHostToDevice);

	// Run kernel with one thread for each element
	// First value is number of blocks, second is threads per block.  Max 1024 threads per block
	simple_multiply<<<ELEMENTS / 100, 100>>>(buffer_C, 10, 10, 10, 10, buffer_A, buffer_B);

	// Read output buffer back to the host
	cudaMemcpy(&C[0], buffer_C, data_size, cudaMemcpyDeviceToHost);

	// Clean up resources
	cudaFree(buffer_A);
	cudaFree(buffer_B);
	cudaFree(buffer_C);

	/*
	// Test that the results are correct
	for (int i = 0; i < ELEMENTS; ++i)
		// To be figured out still
	*/

	cout << "Finished" << endl;

	return 0;
}
