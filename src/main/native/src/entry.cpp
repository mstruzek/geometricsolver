#include <iostream>
#include <stdio.h>
#include <cuda_runtime_api.h>

int main(int argc, char* args[]) 
{
    int deviceId;
    cudaError_t error_t;

    int count;
    
    printf("test driver response \n");


    error_t = cudaGetDeviceCount(&count);
    if(error_t != cudaSuccess) {
        printf("driver  error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
        exit(EXIT_FAILURE);

    }

    printf("device count = %d \n", count);
   

    deviceId = 0;

    error_t = cudaSetDevice(deviceId);
    if(error_t != cudaSuccess) {
        printf("driver  error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
        exit(EXIT_FAILURE);

    }


    printf("device set properly \n");


    error_t = cudaDeviceReset();
    if(error_t != cudaSuccess) {
        printf("driver  error %d  = %s \n", static_cast<int>(error_t), cudaGetErrorString(error_t));
        exit(EXIT_FAILURE);

    }


    printf("device reset \n");

    return 0;
}