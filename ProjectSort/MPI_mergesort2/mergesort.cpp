#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// Function Prototypes
void mergeSort(int *array, int size);
void merge(const int *left, int leftCount, const int *right, int rightCount, int *result);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int array_size = (argc > 1) ? std::atoi(argv[1]) : 100;  // Default to 100 if not provided

    int *array = nullptr;
    if (world_rank == 0) {
        // Root process initializes the full array
        array = new int[array_size];
        for (int i = 0; i < array_size; ++i) {
            array[i] = std::rand() % array_size;
        }
    }

    int local_size = array_size / world_size;
    int *local_array = new int[local_size];

    // Root process sends chunks of the array to all processes
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            if (i == 0) {
                std::copy(array, array + local_size, local_array);
            } else {
                MPI_Send(array + i * local_size, local_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        // Other processes receive their chunk of the array
        MPI_Recv(local_array, local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    CALI_CXX_MARK_FUNCTION;

    // Perform local merge sort
    mergeSort(local_array, local_size);

    // Root process receives the sorted sub-arrays from all processes
    if (world_rank == 0) {
        for (int i = 1; i < world_size; ++i) {
            MPI_Recv(array + i * local_size, local_size, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        std::copy(local_array, local_array + local_size, array);
    } else {
        // Other processes send their sorted chunk back to the root process
        MPI_Send(local_array, local_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        // Output the sorted array in the root process
        for (int i = 0; i < array_size; ++i) {
            // std::cout << array[i] << " ";
        }
        // std::cout << std::endl;

        delete[] array;
    }

    delete[] local_array;

    MPI_Finalize();
    return 0;
}

void mergeSort(int *array, int size) {
    if (size <= 1) return;

    int mid = size / 2;
    int *left = new int[mid];
    int *right = new int[size - mid];

    std::copy(array, array + mid, left);
    std::copy(array + mid, array + size, right);

    mergeSort(left, mid);
    mergeSort(right, size - mid);

    merge(left, mid, right, size - mid, array);

    delete[] left;
    delete[] right;
}

void merge(const int *left, int leftCount, const int *right, int rightCount, int *result) {
    int i = 0, j = 0, k = 0;
    while (i < leftCount && j < rightCount) {
        if (left[i] < right[j]) {
            result[k++] = left[i++];
        } else {
            result[k++] = right[j++];
        }
    }
    while (i < leftCount) {
        result[k++] = left[i++];
    }
    while (j < rightCount) {
        result[k++] = right[j++];
    }
}
