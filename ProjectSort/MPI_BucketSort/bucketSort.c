
// source: ChatGPT

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void printArray(double *arr, int size) {
    printf("Array: ");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int compareDoubles(const void a, const voidb) {
    double diff = ((double)a) - ((double)b);
    if (diff < -1e-10) return -1;
    if (diff > 1e-10) return 1;
    return 0;
}


void bucketSort(double *arr, int size, int my_rank, int num_procs) {
    const int num_buckets = num_procs;
    double *buckets[num_buckets];
    int local_bucket_sizes[num_buckets];

    // Initialize buckets
    for (int i = 0; i < num_buckets; i++) {
        buckets[i] = (double *)malloc(size * sizeof(double));
        local_bucket_sizes[i] = 0;
    }

    // Scatter values into buckets based on range
    for (int i = 0; i < size; i++) {
        int bucket_index = (int)(arr[i] * num_buckets);
        if (bucket_index == num_buckets)
            bucket_index--;

        buckets[bucket_index][local_bucket_sizes[bucket_index]++] = arr[i];
    }

    // Sort each local bucket
    for (int i = 0; i < num_buckets; i++) {
        qsort(buckets[i], local_bucket_sizes[i], sizeof(double), compareDoubles);
    }

    // Gather sorted buckets
    double *sorted_array = (double *)malloc(size * sizeof(double));
    int displacement = 0;
    for (int i = 0; i < num_buckets; i++) {
        MPI_Gather(buckets[i], local_bucket_sizes[i], MPI_DOUBLE, sorted_array + displacement,
                   local_bucket_sizes[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        displacement += local_bucket_sizes[i];
        free(buckets[i]);
    }

    // Broadcast sorted array to all processes
    MPI_Bcast(sorted_array, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Copy the sorted array back to the original array
    for (int i = 0; i < size; i++) {
        arr[i] = sorted_array[i];
    }

    free(sorted_array);
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc != 3) {
        if (my_rank == 0) {
            printf("Usage: %s <array_size> <num_processors>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int array_size = atoi(argv[1]);
    int num_processors = atoi(argv[2]);

    if (array_size % num_processors != 0) {
        if (my_rank == 0) {
            printf("Array size must be divisible by the number of processors.\n");
        }
        MPI_Finalize();
        return 1;
    }

    double *arr = NULL;

    if (my_rank == 0) {
        // Generate or read the array based on your needs
        arr = (double *)malloc(array_size * sizeof(double));
        for (int i = 0; i < array_size; i++) {
            arr[i] = (double)rand() / RAND_MAX; // Example: Initialize with random doubles between 0 and 1
        }

        // Display the original array
        printArray(arr, array_size);
    }

    // Broadcast array size to all processes
    MPI_Bcast(&array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast number of processors to all processes
    MPI_Bcast(&num_processors, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the array to all processes
    double *local_arr = (double *)malloc(array_size / num_processors * sizeof(double));
    MPI_Scatter(arr, array_size / num_processors, MPI_DOUBLE, local_arr, array_size / num_processors, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    // Perform bucket sort
    bucketSort(local_arr, array_size / num_processors, my_rank, num_processors);

    // Gather the sorted array
    MPI_Gather(local_arr, array_size / num_processors, MPI_DOUBLE, arr, array_size / num_processors, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    // Display the sorted array
    if (my_rank == 0) {
        printArray(arr, array_size);
    }

    free(local_arr);
    if (my_rank == 0) {
        free(arr);
    }

    MPI_Finalize();

    return 0;
}
