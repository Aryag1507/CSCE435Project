
// source: ChatGPT

#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include <caliper/cali.h>

const char* mem_alloc = "mem_alloc";
const char* sorting = "sorting";
const char* whole_computation = "whole_computation";

__global__ void simpleMerge(int *d_array, int size, int width) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start = idx * width * 2;

    int middle = min(start + width, size);
    int end = min(start + width * 2, size);

    int *temp = new int[width * 2];
    int i = start, j = middle, k = 0;

    // Merge the two sorted subarrays into temp
    while (i < middle && j < end) {
        if (d_array[i] < d_array[j]) {
            temp[k++] = d_array[i++];
        } else {
            temp[k++] = d_array[j++];
        }
    }

    // Copy the remaining elements of the left subarray if there are any
    while (i < middle) {
        temp[k++] = d_array[i++];
    }
    // Copy the remaining elements of the right subarray if there are any
    while (j < end) {
        temp[k++] = d_array[j++];
    }

    // Copy the merged subarray back into the original array
    for (i = start, k = 0; i < end; i++, k++) {
        d_array[i] = temp[k];
    }

    delete[] temp;
}

void printArray(int *array, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main() {


    int n;
    int threadsPerBlock;

    std::cout << "Enter the number of elements (must be a power of 2): ";
    std::cin >> n;

    std::cout << "Enter the number of threads per block: ";
    std::cin >> threadsPerBlock;

    CALI_MARK_BEGIN(whole_computation);

    int *h_array = new int[n];

    std::srand(static_cast<unsigned int>(std::time(0)));
    for (int i = 0; i < n; i++) {
        h_array[i] = std::rand() % 1000;
    }

    std::cout << "Unsorted array: \n";
    printArray(h_array, n);

    CALI_MARK_BEGIN(mem_alloc);
    int *d_array;
    cudaMalloc(&d_array, n * sizeof(int));
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END(mem_alloc);

    CALI_MARK_BEGIN(sorting);
    for (int width = 1; width < n; width *= 2) {
        int blocks = (n + threadsPerBlock - 1) / (threadsPerBlock * 2);
        simpleMerge<<<blocks, threadsPerBlock>>>(d_array, n, width);
        cudaDeviceSynchronize();
    }
    CALI_MARK_END(sorting);

    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sorted array: \n";
    printArray(h_array, n);

    delete[] h_array;
    cudaFree(d_array);

    CALI_MARK_BEGIN(whole_computation);


    return 0;
}