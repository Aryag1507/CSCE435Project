
// source:

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void merge(double arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    double L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(double arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <array_size> <num_processors>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int array_size = atoi(argv[1]);
    int num_processors = atoi(argv[2]);

    if (array_size % num_processors != 0) {
        if (rank == 0) {
            printf("Array size must be divisible by the number of processors.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int chunk_size = array_size / num_processors;
    double *local_array = (double *)malloc(sizeof(double) * chunk_size);
    double *arr = NULL;

    if (rank == 0) {
        arr = (double *)malloc(sizeof(double) * array_size);

        // Initialize your array, or read it from somewhere
        if (rank == 0) {
            printf("Original Array: ");
            for (int i = 0; i < array_size; i++) {
                arr[i] = (double)rand() / RAND_MAX; // Example: Initialize with random doubles between 0 and 1
                printf("%.2f ", arr[i]);
            }
            printf("\n");
        }

        // Distribute the array to all processes
        MPI_Scatter(arr, chunk_size, MPI_DOUBLE, local_array, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        // Receive the local array from the root process
        MPI_Scatter(arr, chunk_size, MPI_DOUBLE, local_array, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Perform local merge sort
    mergeSort(local_array, 0, chunk_size - 1);

    // Gather the sorted subarrays
    MPI_Gather(local_array, chunk_size, MPI_DOUBLE, arr, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Final merge on the root process
        mergeSort(arr, 0, array_size - 1);

        // Print the sorted array
        printf("Sorted Array: ");
        for (int i = 0; i < array_size; i++) {
            printf("%.2f ", arr[i]);
        }
        printf("\n");

        // Free memory
        free(arr);
    }

    free(local_array);
    MPI_Finalize();

    return 0;
}
