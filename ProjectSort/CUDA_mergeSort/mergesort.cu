#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <float.h>

#include <stdlib.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// Assuming array_fill function is defined elsewhere
void array_fill_random(float *array, int size) {
    // Seed the random number generator to get different results each run
    srand(time(NULL));

    for (int i = 0; i < size; i++) {
        // Generate a random float between 0 and 1
        array[i] = (float)rand() / RAND_MAX;
    }
}

// Fill array in sorted order
void array_fill_sorted(float *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (float)i;  // or any other sorted pattern
    }
}

// Fill array in reverse sorted order
void array_fill_reverse_sorted(float *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (float)(size - i - 1);  // reverse order
    }
}

// Fill array with 1% perturbation
void array_fill_perturbed(float *array, int size) {
    array_fill_sorted(array, size);  // Start with a sorted array
    int perturb_count = size / 100;   // 1% of size
    for (int i = 0; i < perturb_count; i++) {
        int index = rand() % size;
        array[index] = (float)rand() / RAND_MAX;  // Random perturbation
    }
}

bool is_sorted(float *array, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            return false; // Not sorted
        }
    }
    return true;
}

void adiak_stuff() {
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    // adiak::value("InputSize", array_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    // adiak::value("num_procs", size); // The number of processors (MPI ranks)
    adiak::value("group_num", 2); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
}


int THREADS;
int BLOCKS;
int NUM_VALS;

__global__ void mergeKernel(float *dev_values, int size, int mid, int upper) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= upper) return; // Check if the thread is within the range

    // Calculate starting points of the two halves
    int start1 = idx > mid ? mid : idx;
    int start2 = idx > mid ? idx : mid;
    int end1 = mid, end2 = upper;

    // Calculate the size needed for the temporary array in shared memory
    int tempSize = upper - start1;
    extern __shared__ float temp[];

    // Merge the two halves into temp
    int i = start1, j = start2, k = 0;
    while (i < end1 && j < end2) {
        if (dev_values[i] < dev_values[j]) {
            temp[k++] = dev_values[i++];
        } else {
            temp[k++] = dev_values[j++];
        }
    }

    // Copy the remaining elements of the first half, if any
    while (i < end1) {
        temp[k++] = dev_values[i++];
    }

    // Copy the remaining elements of the second half, if any
    while (j < end2) {
        temp[k++] = dev_values[j++];
    }

    // Copy back the merged elements to the original array
    k = 0;
    for (i = start1; i < upper; i++) {
        dev_values[i] = temp[k++];
    }
}


void mergeSort(float *dev_values, int size) {
    dim3 threads(THREADS, 1);
    dim3 blocks(BLOCKS, 1);

    // Calculate shared memory size per block
    int sharedMemSize = size * sizeof(float);
    
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (int width = 1; width < size; width *= 2) {
        for (int i = 0; i < size; i = i + 2 * width) {
            mergeKernel<<<blocks, threads, sharedMemSize>>>(dev_values, size, i, min(i + 2 * width, size));
            // Check for errors after kernel launch
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }
        cudaDeviceSynchronize();
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <number of threads> <number of values> <input type>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = (NUM_VALS + THREADS - 1) / THREADS;
    char* inputType = argv[3]; // New argument for input type


    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);
    printf("Input type: %s\n", inputType);

    float total = 0.0f, minTime = FLT_MAX, maxTime = 0.0f;
    const int numRuns = 10;
    float times[numRuns];

    // Allocate host array
    float *values = (float*) malloc(NUM_VALS * sizeof(float));
    CALI_MARK_BEGIN("main");

    // Allocate device memory
    float *dev_values;
    cudaMalloc((void**) &dev_values, NUM_VALS * sizeof(float));

    for (int run = 0; run < numRuns; ++run) {

        // Fill the array based on input type
        if (strcmp(inputType, "Sorted") == 0) {
            array_fill_sorted(values, NUM_VALS);
        }
        else if (strcmp(inputType, "ReverseSorted") == 0) {
            array_fill_reverse_sorted(values, NUM_VALS);
        }
        else if (strcmp(inputType, "Random") == 0) {
            array_fill_random(values, NUM_VALS);
        }
        else if (strcmp(inputType, "1%Perturbed") == 0) {
            array_fill_perturbed(values, NUM_VALS);
        }
        else {
            fprintf(stderr, "Invalid input type!\n");
            exit(EXIT_FAILURE);
        }

        // Copy data from host to device
        cudaMemcpy(dev_values, values, NUM_VALS * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        mergeSort(dev_values, NUM_VALS);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&times[run], start, stop);
        total += times[run];
        if (times[run] < minTime) minTime = times[run];
        if (times[run] > maxTime) maxTime = times[run];

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Copy sorted data back to host and verify
        cudaMemcpy(values, dev_values, NUM_VALS * sizeof(float), cudaMemcpyDeviceToHost);
        if (!is_sorted(values, NUM_VALS)) {
            fprintf(stderr, "Array is not correctly sorted!\n");
            exit(EXIT_FAILURE);
        }
    }

    float average = total / numRuns;

    CALI_MARK_END("main");

    // Adiak value updates
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    // adiak::value("InputSize", array_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    // adiak::value("num_procs", size); // The number of processors (MPI ranks)
    adiak::value("group_num", 2); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Print timing statistics
    printf("Average GPU Time: %f ms\n", average);
    printf("Minimum GPU Time: %f ms\n", minTime);
    printf("Maximum GPU Time: %f ms\n", maxTime);
    printf("Total GPU Time: %f ms\n", total);

    // Clean up
    cudaFree(dev_values);
    free(values);

    return 0;
}
