
// source: 

#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

__global__ void oddeven(int* x, int I, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (I == 0 && id < n / 2) {
        int ix = id * 2;
        if (x[ix] > x[ix + 1]) {
            int temp = x[ix];
            x[ix] = x[ix + 1];
            x[ix + 1] = temp;
        }
    }
    else if (I == 1 && id < (n - 1) / 2) {
        int ix = id * 2 + 1;
        if (x[ix] > x[ix + 1]) {
            int temp = x[ix];
            x[ix] = x[ix + 1];
            x[ix + 1] = temp;
        }
    }
}

void printArray(int *array, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    int *a, *d, n, *c;
    int threadsPerBlock;

    std::cout << "Enter the number of elements to sort: ";
    std::cin >> n;
    std::cout << "Enter the number of threads per block: ";
    std::cin >> threadsPerBlock;

    a = new int[n];
    c = new int[n];

    srand(time(0));

    for (int i = 0; i < n; i++) {
        a[i] = rand() % 1000;
    }

	std::cout << "Unsorted Array is: ";
    printArray(a, n);

    cudaMalloc((void**)&d, n * sizeof(int));
    cudaMemcpy(d, a, n * sizeof(int), cudaMemcpyHostToDevice);

    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < n; i++) {
        oddeven<<<blocksPerGrid, threadsPerBlock>>>(d, i % 2, n);
        cudaDeviceSynchronize(); // Ensure each stage of the sort is complete before starting the next
    }

    cudaMemcpy(c, d, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sorted Array is: ";
    printArray(c, n);

    cudaFree(d);
    delete[] a;
    delete[] c;

    return 0;
}