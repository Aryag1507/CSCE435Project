
//source: https://github.com/ufukomer/cuda-radix-sort/blob/master/radix-sort/kernel.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";

const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_small = "comm_large";
const char *whole_computation = "whole_computation";

const char* data_init = "data_init";

double whole_compute_start, whole_compute_end;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t cudaError = call; \
        if (cudaError != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaError)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__device__ int getMax(int* arr, int numElements) {
    int max = arr[0];
    for (int i = 1; i < numElements; i++)
        if (arr[i] > max)
            max = arr[i];
    return max;
}

__device__ void countSort(int* arr, int* output, int numElements, int exp) {
    const int bucketSize = 10;
    int count[bucketSize] = {0};

    for (int i = 0; i < numElements; i++)
        count[(arr[i] / exp) % bucketSize]++;

    for (int i = 1; i < bucketSize; i++)
        count[i] += count[i - 1];

    for (int i = numElements - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % bucketSize] - 1] = arr[i];
        count[(arr[i] / exp) % bucketSize]--;
    }

    for (int i = 0; i < numElements; i++)
        arr[i] = output[i];
}

__global__ void radixSort(int* d_data, int* d_output, int numElements) {
    int max = getMax(d_data, numElements);

    for (int exp = 1; max / exp > 0; exp *= 10){
        countSort(d_data, d_output, numElements, exp);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <array_length> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    CALI_MARK_BEGIN(whole_computation);

    int array_length = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    int* h_data = (int*)malloc(array_length * sizeof(int));

    double init_start, init_end;
    double comp_large_begin, comp_large_end;

    CALI_MARK_BEGIN(data_init);

    // Initialize the array with random values
    for (int i = 0; i < array_length; ++i) {
        h_data[i] = rand();
    }

    CALI_MARK_END(data_init);

    // Display the original array
    printf("Original Array:\n");
    for (int i = 0; i < array_length; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Allocate and copy the array to the device
    int* d_data;
    int* d_output;
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CUDA_CHECK(cudaMalloc((void**)&d_data, array_length * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, array_length * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, array_length * sizeof(int), cudaMemcpyHostToDevice));
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Launch the kernel
    int threadsPerBlock = num_threads;
    int blocksPerGrid = (array_length + threadsPerBlock - 1) / threadsPerBlock;

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);    

    radixSort<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_output, array_length);

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the sorted array back to the host
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CUDA_CHECK(cudaMemcpy(h_data, d_data, array_length * sizeof(int), cudaMemcpyDeviceToHost));
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Display the sorted array
    printf("Sorted Array:\n");
    for (int i = 0; i < array_length; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    CALI_MARK_END(whole_computation);

    // Cleanup
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}