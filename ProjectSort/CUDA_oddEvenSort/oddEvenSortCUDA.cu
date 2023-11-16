#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char *data_init = "data_init";
const char *comp = "comp";
const char *main = "main";
const char *comp_large = "comp_large";

void initializeArray(int *array, int size, int input_type) {
    srand(time(0)); // Seed for random number generation

    switch (input_type) {
        case 1: // Random array
            for (int i = 0; i < size; i++) {
                array[i] = rand() % 1000; // Random number between 0 and 999
            }
            break;

        case 2: // Sorted array
            for (int i = 0; i < size; i++) {
                array[i] = i;
            }
            break;

        case 3: // Reverse sorted array
            for (int i = 0; i < size; i++) {
                array[i] = size - i - 1;
            }
            break;

        case 4: // 1% perturbed array
            for (int i = 0; i < size; i++) {
                array[i] = i;
            }
            for (int i = 0; i < size / 100; i++) {
                int idx1 = rand() % size;
                int idx2 = rand() % size;
                swap(array[idx1], array[idx2]);
            }
            break;

        default:
		    fprintf(stderr, "Invalid input type: %d\n", input_type);
            break;
    }
}

__device__ inline void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

__global__ void oddEvenSort(int *a, int n, bool *sorted)
{
    __shared__ int isSorted;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int phase = 0; phase < n; ++phase)
    {
        isSorted = 1;
        __syncthreads();

        // Even phase
        if (phase % 2 == 0)
        {
            if (idx < n / 2)
            {
                int i = idx * 2;
                if (a[i] > a[i + 1])
                {
                    swap(a[i], a[i + 1]);
                    isSorted = 0;
                }
            }
        }
        // Odd phase
        else
        {
            if (idx < (n - 1) / 2)
            {
                int i = idx * 2 + 1;
                if (a[i] > a[i + 1])
                {
                    swap(a[i], a[i + 1]);
                    isSorted = 0;
                }
            }
        }

        __syncthreads();

        // Check if the array is sorted
        if (isSorted == 0)
            *sorted = false;
    }
}

void printArray(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    
    CALI_CXX_MARK_FUNCTION;
    CALI_MARK_BEGIN(main);

    int *a, *d, n, *c;
    int threadsPerBlock;

    CALI_MARK_BEGIN(data_init);
    n = atoi(argv[1]);
    threadsPerBlock = atoi(argv[2]);

    a = new int[n];
    c = new int[n];

    int inputType = atoi(argv[3]); 
    initializeArray(a, n, inputType);

    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_END(data_init);

    std::cout << "Unsorted Array is: ";
    printArray(a, n);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    cudaMalloc((void **)&d, n * sizeof(int));
    cudaMemcpy(d, a, n * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    bool sorted = false;
    bool *d_sorted;
    cudaMalloc((void **)&d_sorted, sizeof(bool));

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    while (!sorted)
    {
        sorted = true;
        cudaMemcpy(d_sorted, &sorted, sizeof(bool), cudaMemcpyHostToDevice);

        oddEvenSort<<<blocksPerGrid, threadsPerBlock>>>(d, n, d_sorted);
        cudaDeviceSynchronize();

        cudaMemcpy(&sorted, d_sorted, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    cudaMemcpy(c, d, n * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    std::cout << "Sorted Array is: ";
    printArray(c, n);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "OddEvenSort");
    adiak::value("ProgrammingModel", "CUDA");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", n);
    adiak::value("InputType", sizeof(int));
    adiak::value("num_threads", threadsPerBlock);
    adiak::value("group_num", 2);
    adiak::value("implementation_source", "Online/Handwritten");

    cudaFree(d);
    cudaFree(a);
    cudaFree(c);

    mgr.stop();
    mgr.flush();

    CALI_MARK_END(main);

    return 0;
}
