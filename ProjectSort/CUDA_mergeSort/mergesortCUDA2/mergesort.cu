#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <float.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* mainn = "main";
const char* data_init = "data_init";
const char* comm = "comm";
const char* MPIBarrier = "MPI_Barrier";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* MPIRecv = "MPI_Recv";
const char* MPISend = "MPI_Send";

const char* cuda_memcpy = "cudaMemcpy";

const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

enum InputType { SORTED, RANDOM, REVERSE_SORTED, PERTURBED };

// Function to generate input data
void generateInput(float* array, InputType type, int size) {
    srand(time(NULL)); // Seed for random number generation

    switch(type) {
        case SORTED:
            // Generate a sorted array
            for(int i = 0; i < size; i++) {
                array[i] = (float)i / size;
            }
            break;

        case RANDOM:
            // Generate a random array
            for(int i = 0; i < size; i++) {
                array[i] = (float)rand() / RAND_MAX;
            }
            break;

        case REVERSE_SORTED:
            // Generate a reverse sorted array
            for(int i = 0; i < size; i++) {
                array[i] = (float)(size - i - 1) / size;
            }
            break;

        case PERTURBED:
            // Generate a mostly sorted array with a few random elements
            for(int i = 0; i < size; i++) {
                array[i] = (float)i / size;
            }
            // Perturb about 1% of the elements
            for(int i = 0; i < size / 100; i++) {
                int index = rand() % size;
                array[index] = (float)rand() / RAND_MAX;
            }
            break;

        default:
            printf("Invalid input type\n");
            break;
    }
}

// CUDA Kernel for Merge Sort
__global__ void mergeKernel(float* deviceArray, float* auxArray, int size, int width) {
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

void merge(float* array, int left, int middle, int right) {
    int n1 = middle - left + 1;
    int n2 = right - middle;
    float* L = (float*)malloc(n1 * sizeof(float));
    float* R = (float*)malloc(n2 * sizeof(float));

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

void gpuMergeSort(float* hostArray, int size, int numThreads, int subArraySize, float* gpuTimes, int* gpuOps) {
    float* deviceArray;
    float* auxArray;
    cudaMalloc(&deviceArray, size * sizeof(float));
    cudaMalloc(&auxArray, size * sizeof(float));

    // CUDA event creation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start measuring time
    cudaEventRecord(start);

    // Copy data to device
    cudaMemcpy(deviceArray, hostArray, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(numThreads);
    dim3 gridSize((subArraySize + blockSize.x - 1) / blockSize.x);

    for (int width = 1; width < subArraySize; width *= 2) {
        for (int start = 0; start < size; start += subArraySize) {
            mergeKernel<<<gridSize, blockSize>>>(deviceArray + start, auxArray + start, subArraySize, width);
        }
        cudaDeviceSynchronize(); 
        cudaMemcpy(deviceArray, auxArray, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Copy data back to host
    cudaMemcpy(hostArray, deviceArray, size * sizeof(float), cudaMemcpyDeviceToHost);

    // End measuring time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculating time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    gpuTimes[*gpuOps] = milliseconds; // Store the time
    (*gpuOps)++; // Increment the operation count

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(deviceArray);
    cudaFree(auxArray);
}

void hybridMergeSort(float* hostArray, int size, int numThreads, int subArraySize, float* gpuTimes, int* gpuOps) {
    gpuMergeSort(hostArray, size, numThreads, subArraySize, gpuTimes, gpuOps);

    // CPU-based merge for larger sections
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
    float* hostArray = (float*)malloc(sizeof(float) * size);
    if (hostArray == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    // Generate input data
    generateInput(hostArray, inputType, size);

    // Array to store GPU times and operation count
    float gpuTimes[10]; // Assuming a maximum of 10 GPU operations
    int gpuOps = 0;

    // Set the size of subarrays to be sorted on the GPU
    int subArraySize = 1024; // Adjust as needed

    // Perform hybrid merge sort
    hybridMergeSort(hostArray, size, numThreads, subArraySize, gpuTimes, &gpuOps);

    // Check if the array is sorted
    bool isSorted = true;
    for (int i = 0; i < size - 1; i++) {
        if (hostArray[i] > hostArray[i + 1]) {
            isSorted = false;
            break;
        }
    }
    printf("Array is %s\n", isSorted ? "correctly sorted" : "not correctly sorted");

    // Calculate and print GPU times
    float totalGPUTime = 0, minGPUTime = FLT_MAX, maxGPUTime = FLT_MIN;
    for (int i = 0; i < gpuOps; i++) {
        totalGPUTime += gpuTimes[i];
        if (gpuTimes[i] < minGPUTime) minGPUTime = gpuTimes[i];
        if (gpuTimes[i] > maxGPUTime) maxGPUTime = gpuTimes[i];
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", size); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", numThreads); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 2); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    printf("Avg GPU time/rank: %f ms\n", totalGPUTime / gpuOps);
    printf("Min GPU time/rank: %f ms\n", minGPUTime);
    printf("Max GPU time/rank: %f ms\n", maxGPUTime);
    printf("Total GPU time: %f ms\n", totalGPUTime);

    // Free the allocated memory
    free(hostArray);

    return 0;
}


/*


*/
