
// source: ChatGPT

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define MAX_WIDTH 512

const char* mem_alloc = "mem_alloc";
const char* sorting = "sorting";
const char* whole_computation = "whole_computation";

__global__ void merge(int* d_data, int size, int width) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start = idx * width * 2;

    int middle = min(start + width, size);
    int end = min(start + width * 2, size);

    int *temp = new int[width * 2];
    int i = start, j = middle, k = 0;

    // Merge the two sorted subarrays into temp
    while (i < middle && j < end) {
        if (d_data[i] < d_data[j]) {
            temp[k++] = d_data[i++];
        } else {
            temp[k++] = d_data[j++];
        }
    }

    // Copy the remaining elements of the left subarray if there are any
    while (i < middle) {
        temp[k++] = d_data[i++];
    }
    // Copy the remaining elements of the right subarray if there are any
    while (j < end) {
        temp[k++] = d_data[j++];
    }

    // Copy the merged subarray back into the original array
    for (i = start, k = 0; i < end; i++, k++) {
        d_data[i] = temp[k];
    }

    delete[] temp;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <n> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    int* h_data = (int*)malloc(n * sizeof(int));

    // Initialize the array with random values
    for (int i = 0; i < n; ++i) {
        h_data[i] = rand();
    }

    // Display the original array
    printf("Original Array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    int* d_data;
    cudaMalloc((void**)&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    for (int width = 1; width < n; width *= 2) {
        int blocks = (n + (2 * num_threads) - 1) / (2 * num_threads);
        merge<<<blocks, num_threads>>>(d_data, n, width);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sorted array: \n";
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    free(h_data);
    cudaFree(d_data);

    return 0;
}