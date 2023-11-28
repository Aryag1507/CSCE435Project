#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <float.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


enum InputType { SORTED, RANDOM, REVERSE_SORTED, PERTURBED };

// Function to generate input data
void generateInput(int* array, InputType type, int size) {
    srand(time(NULL)); // Seed for random number generation

    switch(type) {
        case SORTED:
            // Generate a sorted array
            for(int i = 0; i < size; i++) {
                array[i] = i;
            }
            break;

        case RANDOM:
            // Generate a random array
            for(int i = 0; i < size; i++) {
                array[i] = rand() % size;
            }
            break;

        case REVERSE_SORTED:
            // Generate a reverse sorted array
            for(int i = 0; i < size; i++) {
                array[i] = size - i - 1;
            }
            break;

        case PERTURBED:
            // Generate a mostly sorted array with a few random elements
            for(int i = 0; i < size; i++) {
                array[i] = i;
            }
            // Perturb about 1% of the elements
            for(int i = 0; i < size / 100; i++) {
                int index = rand() % size;
                array[index] = rand() % size;
            }
            break;

        default:
            printf("Invalid input type\n");
            break; // Add 'break' here
    }
}

// CUDA Kernel for Merge Sort
__global__ void mergeKernel(int* deviceArray, int* auxArray, int size, int width) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = threadId * width * 2;

    // Check if the thread is working within the bounds of the array
    if (start >= size) return;

    // Calculate the middle and end indices of the sections to merge
    int middle = min(start + width, size);
    int end = min(start + width * 2, size);

    // Merge the two halves
    int i = start, j = middle, k = start;
    while (i < middle && j < end) {
        if (deviceArray[i] < deviceArray[j]) {
            auxArray[k++] = deviceArray[i++];
        } else {
            auxArray[k++] = deviceArray[j++];
        }
    }

    // Copy remaining elements from the left half
    while (i < middle) {
        auxArray[k++] = deviceArray[i++];
    }

    // Copy remaining elements from the right half
    while (j < end) {
        auxArray[k++] = deviceArray[j++];
    }
}


void merge(int* array, int left, int middle, int right) {
    int n1 = middle - left + 1;
    int n2 = right - middle;
    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = array[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = array[middle + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            array[k++] = L[i++];
        } else {
            array[k++] = R[j++];
        }
    }

    while (i < n1) {
        array[k++] = L[i++];
    }
    while (j < n2) {
        array[k++] = R[j++];
    }

    free(L);
    free(R);
}

void gpuMergeSort(int* hostArray, int size, int numThreads, int subArraySize) {
    int* deviceArray;
    int* auxArray;
    cudaMalloc(&deviceArray, size * sizeof(int));
    cudaMalloc(&auxArray, size * sizeof(int));
    cudaMemcpy(deviceArray, hostArray, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(numThreads);
    dim3 gridSize((subArraySize + blockSize.x - 1) / blockSize.x);

    for (int width = 1; width < subArraySize; width *= 2) {
        for (int start = 0; start < size; start += subArraySize) {
            mergeKernel<<<gridSize, blockSize>>>(deviceArray + start, auxArray + start, subArraySize, width);
        }
        cudaDeviceSynchronize(); 
        cudaMemcpy(deviceArray, auxArray, size * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(hostArray, deviceArray, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceArray);
    cudaFree(auxArray);
}


void hybridMergeSort(int* hostArray, int size, int numThreads, int subArraySize) {
    gpuMergeSort(hostArray, size, numThreads, subArraySize);

    for (int width = subArraySize; width < size; width *= 2) {
        for (int i = 0; i < size; i += 2 * width) {
            int mid = min(i + width - 1, size - 1);
            int right = min(i + 2 * width - 1, size - 1);
            merge(hostArray, i, mid, right);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <num_threads> <array_size> <input_type>\n", argv[0]);
        return 1;
    }

    // Parse command line arguments
    int numThreads = atoi(argv[1]);
    int size = atoi(argv[2]);
    InputType inputType = static_cast<InputType>(atoi(argv[3]));

    if (numThreads <= 0 || size <= 0 || inputType < SORTED || inputType > PERTURBED) {
        fprintf(stderr, "Invalid arguments\n");
        return 1;
    }

    // Allocate memory for the array
    int* hostArray = (int*)malloc(sizeof(int) * size);
    if (hostArray == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    // Generate input data based on the specified type
    generateInput(hostArray, inputType, size);

    // Set the size of subarrays to be sorted on the GPU (you can adjust this based on your needs)
    int subArraySize = 1024;

    // Call the hybrid merge sort function
    hybridMergeSort(hostArray, size, numThreads, subArraySize);

    // Print the sorted array
    printf("Sorted Array:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", hostArray[i]);
    }
    printf("\n");

    // Free the allocated memory
    free(hostArray);

    return 0;
}
