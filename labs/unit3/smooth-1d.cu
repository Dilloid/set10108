#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace std;

// 5x5 gaussian kernel radius
constexpr int RADIUS = 2;

__global__ void smooth(float* values_in, float* values_out, int width)
{

	constexpr float KERNEL[] = { 0.06136,	0.24477,	0.38774,	0.24477,	0.06136, };
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	// prevent out-of-bounds
	if (x >= width) return;

	// temporary variable to put the weighted sum
	float tmp = 0.0f;

	for (int ox = -RADIUS; ox <= RADIUS; ++ox)
	{
		int xnb = min(max(x + ox, 0), width - 1);
		int kernel_col = ox + RADIUS;// in the range of [0,2*RADIUS]
		float value = values_in[xnb] * KERNEL[kernel_col];
		tmp += value;
	}
	values_out[x] = tmp;
}

void save_grayscale_png(const char* filename, const std::vector<float>& values, int width, int height)
{
	std::vector<uint8_t> imgdata(width * height);
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
			imgdata[x + y * width] = uint8_t(values[x + y * width] * 255);
	stbi_write_png(filename, width, height, 1, imgdata.data(), width);
}

int main(int argc, char** argv)
{
	if (argc == 1)
	{
		printf("Please provide a folder as the argument (full path).");
		exit(0);
	}

	std::string filename_in = argv[1] + (string)"image_1d_before.png";

	int width = 32;
	vector<float> h_values_in(width);    // Input array

	cout << "Input: ";
	for (int i = 0; i < width; i++)
	{
		float number = (float)rand() / RAND_MAX;
		cout << number << ", ";
		h_values_in[i] = number;
	}

	save_grayscale_png(filename_in.c_str(), h_values_in, width, 1);

	// Create host memory
	const size_t NUM_BYTES = sizeof(float) * width;
	vector<float> h_values_out(width);    // Output array

	float* d_values_in = nullptr;
	float* d_values_out = nullptr;

	// Initialise buffers
	cudaMalloc((void**)&d_values_in, NUM_BYTES);
	cudaMalloc((void**)&d_values_out, NUM_BYTES);

	// Write host data to device
	cudaMemcpy(d_values_in, h_values_in.data(), NUM_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_values_out, h_values_out.data(), NUM_BYTES, cudaMemcpyHostToDevice);

	int M = 32;
	int N = unsigned(width);
	smooth <<<(N + (M - 1)) / M, M>> > (d_values_in, d_values_out, width);

	// Read output buffer back to the host
	cudaMemcpy(h_values_out.data(), d_values_out, NUM_BYTES, cudaMemcpyDeviceToHost);

	// Clean up resources
	cudaFree(d_values_in);
	cudaFree(d_values_out);

	cout << "\nOutput: ";
	for (int i = 0; i < width; i++)
	{
		cout << h_values_out[i] << ", ";
	}

	std::string filename_out = argv[1] + (string)"image_1d_out.png";
	save_grayscale_png(filename_out.c_str(), h_values_out, width, 1);

	return 0;
}
