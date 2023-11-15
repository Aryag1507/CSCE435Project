/ source: ChatGPT
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* initialization = "initialization";
const char* gathering = "gathering";
const char* final_sort = "final_sort";
const char* local_sort_time = "local_sort_time";
const char* whole_computation = "whole_computation";

const char* comp_large = "comp_large";
const char * comp_small = "comp_small";

void rng(int* arr, int n) {
    int seed = 13516095;
    srand(seed);
    for(long i = 0; i < n; i++) {
        arr[i] = (int)rand();
    }
}

int get_max(int* arr, int n) { 
    int mx = arr[0]; 
    for (int i = 1; i < n; i++) 
        if (arr[i] > mx) 
            mx = arr[i]; 
    return mx; 
} 

void count_sort(int* arr, int n, int divisor, int num_process, int rank) {
    // Scatter random numbers to all processes
    int n_per_proc = n / num_process;
    int* sub_arr = (int*) malloc(sizeof(int) * n_per_proc);
    MPI_Scatter(arr, n_per_proc, MPI_INT, sub_arr, n_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute sub count in each processes
    int i, sub_count[10] = {0};
    for (i = 0; i < n_per_proc; i++) {
        sub_count[(sub_arr[i] / divisor) % 10]++;
    }

    // Reduce all the sub counts to root process
    if (rank == 0) {
        int count[10] = {0};
        MPI_Reduce(sub_count, count, 10, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        for (i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        
        int* temp_arr = (int*) malloc(sizeof(int) * n);
        for (i = n - 1; i >= 0; i--) { 
            temp_arr[count[(arr[i] / divisor) % 10] - 1] = arr[i]; 
            count[(arr[i] / divisor) % 10]--; 
        }
        memcpy(arr, temp_arr, sizeof(int) * n);
        free(temp_arr);

    } else {
        MPI_Reduce(sub_count, 0, 10, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    free(sub_arr);
}

void radix_sort(int* arr, int n, int num_process, int rank) { 
    int m = get_max(arr, n);
    for (int divisor = 1; m / divisor > 0; divisor *= 10) {
        count_sort(arr, n, divisor, num_process, rank);
    }
}

void print(int* arr, int n) { 
    for (int i = 0; i < n; i++) 
        printf("%d ", arr[i]);
    printf("\n");
} 

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: mpirun -n <num_processors> ./radixSort <array_size>\n");
        return 1;
    }

    int num_process, rank; 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double init_start, init_end, local_sort_start, local_sort_end, gather_start, gather_end, complete_sort_start, complete_sort_end;
    double whole_compute_start, whole_compute_end;

    cali::ConfigManager mgr;
    mgr.start();

    whole_compute_start = MPI_Wtime();
    CALI_MARK_BEGIN(whole_computation);

    // Initialize MPI
    init_start = MPI_Wtime();
    CALI_MARK_BEGIN(initialization);

    init_end = MPI_Wtime();
    CALI_MARK_END(initialization);

    // Get array size from command line argument
    int n = atoi(argv[1]);

    // Check if the array size is a multiple of the number of processors
    if (n % num_process != 0) {
        if (rank == 0) {
            fprintf(stderr, "Array size (%d) must be a multiple of the number of processors (%d).\n", n, num_process);
        }
        MPI_Finalize();
        return 1;
    }

    // Allocate memory and populate the array with random numbers
    int* arr = (int*)malloc(sizeof(int) * n);
    if (arr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    rng(arr, n);

    // Only rank 0 prints the original array
    if (rank == 0) {
        printf("[Original array]\n");
        print(arr, n);
    }

    // Synchronize before sorting
    MPI_Barrier(MPI_COMM_WORLD);

    // Sort the array
    complete_sort_start = MPI_Wtime();
    CALI_MARK_BEGIN(comp_large);

    radix_sort(arr, n, num_process, rank);

    complete_sort_end = MPI_Wtime();
    CALI_MARK_END(comp_large);

    // Only rank 0 prints the sorted array and timing information
    if (rank == 0) {
        printf("[Sorted array]\n");
        print(arr, n);
        printf("[Sorted in %f seconds]\n", complete_sort_end - complete_sort_start);
    }

    whole_compute_end = MPI_Wtime();
    CALI_MARK_END(whole_computation);

    printf("Local Sort Time (Rank %d): %f seconds\n", rank, complete_sort_end - complete_sort_start);
    printf("whole computation Time: %f seconds\n", whole_compute_end - whole_compute_start);

    free(arr);
    mgr.stop();
    mgr.flush();
    MPI_Finalize();

    return 0;
}