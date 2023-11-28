
//source: https://github.com/ufukomer/cuda-radix-sort/blob/master/radix-sort/kernel.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";

const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_small = "comm_large";
const char *whole_computation = "whole_computation";

const char* data_init = "data_init";

double whole_compute_start, whole_compute_end;

void rng(int* arr, int n, string input) {
    if(input == "random"){
        int seed = 13516095;
        srand(seed);
        for(long i = 0; i < n; i++) {
            arr[i] = (int)rand() % 10000;
        }
    } else if(input == "sorted"){
        for (int i = 0; i < n; i++) {
            arr[i] = i;
        }
    } else if(input == "reverse_sorted"){
        for (int i = 0; i < n; i++) {
            arr[i] = n - i - 1;
        }
    } else if(input == "one_percent"){
        for (int i = 0; i < n; i++) {
            arr[i] = i;
        }
        for (int i = 0; i < n / 100; i++) {
            int idx1 = random() % n;
            int idx2 = random() % n;
            int temp = arr[idx1];
            arr[idx1] = arr[idx2];
            arr[idx2] = temp;
        }
    }
}

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
    if (argc != 4) {
        printf("Usage: %s <array_length> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    CALI_MARK_BEGIN(whole_computation);

    int array_length = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    string input_type = argv[3];

    int* h_data = (int*)malloc(array_length * sizeof(int));

    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);

    // Initialize the array with random values
    for (int i = 0; i < array_length; ++i) {
        h_data[i] = rand();
    }

    rng(h_data, array_length, input_type);

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

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "RadixSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "Cuda"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", array_length); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_threads); // The number of processors (MPI ranks)
    adiak::value("group_num", 2); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Cleanup
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}