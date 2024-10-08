#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpuErrchk.h"

using namespace std;


int main(int argc, char **argv)
{
    // Get number of devices on system
    int deviceCount; 
    gpuErrchk(cudaGetDeviceCount(&deviceCount));

    cout << "Number of devices: " << deviceCount << endl;
    for (int i = 0; i < deviceCount; ++i) 
    {
        // Get properties for device
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        cout << "Device: " << i << endl;
        cout << "Name: " << deviceProp.name << endl;
        cout << "Revision: " << deviceProp.major << "." << deviceProp.minor << endl;
        cout << "Memory: " << deviceProp.totalGlobalMem / 1024 / 1024 << "MB" << endl;
        cout << "Warp Size: " << deviceProp.warpSize << endl;
        cout << "Clock: " << deviceProp.clockRate << endl;
        cout << "Multiprocessors: " << deviceProp.multiProcessorCount << endl;
		cout << "Max Grid Size: " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << endl;
		cout << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << endl;
    } 
    return 0;
}