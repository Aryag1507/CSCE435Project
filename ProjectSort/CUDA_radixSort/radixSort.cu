
//source: https://github.com/ufukomer/cuda-radix-sort/blob/master/radix-sort/kernel.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


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

__device__ void countSort(int* arr, int numElements, int exp) {
    const int bucketSize = 10;
    int* output = new int[numElements];
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

    delete[] output;
}

__global__ void radixSort(int* d_data, int numElements) {
    int max = getMax(d_data, numElements);

    for (int exp = 1; max / exp > 0; exp *= 10)
        countSort(d_data, numElements, exp);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <array_length> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int array_length = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    int* h_data = (int*)malloc(array_length * sizeof(int));

    // Initialize the array with random values
    for (int i = 0; i < array_length; ++i) {
        h_data[i] = rand() % 1000; // Assuming values are in the range [0, 999]
    }

    // Display the original array
    printf("Original Array:\n");
    for (int i = 0; i < array_length; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Allocate and copy the array to the device
    int* d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, array_length * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, array_length * sizeof(int), cudaMemcpyHostToDevice));

    // Launch the kernel
    int threadsPerBlock = num_threads;
    int blocksPerGrid = (array_length + threadsPerBlock - 1) / threadsPerBlock;

    radixSort<<<blocksPerGrid, threadsPerBlock>>>(d_data, array_length);

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the sorted array back to the host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, array_length * sizeof(int), cudaMemcpyDeviceToHost));

    // Display the sorted array
    printf("Sorted Array:\n");
    for (int i = 0; i < array_length; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Cleanup
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}