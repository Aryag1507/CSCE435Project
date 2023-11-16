
// source:

#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

__device__ inline void swap(int &a, int &b) {
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

int main(int argc, char** argv)
{
    int *a, *d, n, *c;
    int threadsPerBlock;

    n = atoi(argv[1]);
    threadsPerBlock = atoi(argv[2]);

    a = new int[n];
    c = new int[n];

    srand(time(0));

    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 1000;
    }

    std::cout << "Unsorted Array is: ";
    printArray(a, n);

    cudaMalloc((void **)&d, n * sizeof(int));
    cudaMemcpy(d, a, n * sizeof(int), cudaMemcpyHostToDevice);

    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    bool sorted = false;
    bool *d_sorted;
    cudaMalloc((void **)&d_sorted, sizeof(bool));

    while (!sorted)
    {
        sorted = true;
        cudaMemcpy(d_sorted, &sorted, sizeof(bool), cudaMemcpyHostToDevice);
        
        oddEvenSort<<<blocksPerGrid, threadsPerBlock>>>(d, n, d_sorted);
        cudaDeviceSynchronize();

        cudaMemcpy(&sorted, d_sorted, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(c, d, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sorted Array is: ";
    printArray(c, n);

    cudaFree(d);
    cudaFree(a);
    cudaFree(c);

    return 0;
}

