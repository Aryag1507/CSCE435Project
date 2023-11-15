#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* initialization = "initialization";
const char* gathering = "gathering";
const char* final_sort = "final_sort";
const char* local_sort_time = "local_sort_time";
const char* whole_computation = "whole_computation";
const char* comp_large = "comp_large";

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
    CALI_MARK_BEGIN(comp_large);
    if (l < r) {
        int m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
    CALI_MARK_END(comp_large);
}


int main(int argc, char **argv) {
    CALI_CXX_MARK_FUNCTION;
    
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <array_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int array_size = atoi(argv[1]);
    if (array_size <= 0 || array_size % size != 0) {
        if (rank == 0) {
            printf("Array size must be a positive integer divisible by the number of processors.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int chunk_size = array_size / size;
    double *local_array = (double *)malloc(sizeof(double) * chunk_size);
    double *arr = NULL;
    double init_start, init_end, local_sort_start, local_sort_end, gather_start, gather_end, final_sort_start, final_sort_end;
    double whole_compute_start, whole_compute_end;

    whole_compute_start = MPI_Wtime();
    CALI_MARK_BEGIN(whole_computation);
    
    cali::ConfigManager mgr;
    mgr.start();

    if (rank == 0) {
        arr = (double *)malloc(sizeof(double) * array_size);
        init_start = MPI_Wtime();
        CALI_MARK_BEGIN(initialization);
        for (int i = 0; i < array_size; i++) {
            arr[i] = (double)rand() / RAND_MAX;
        }
        CALI_MARK_END(initialization);
        init_end = MPI_Wtime();
    }

    MPI_Scatter(arr, chunk_size, MPI_DOUBLE, local_array, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_sort_start = MPI_Wtime();
    CALI_MARK_BEGIN(local_sort_time);
    mergeSort(local_array, 0, chunk_size - 1);
    CALI_MARK_END(local_sort_time);
    local_sort_end = MPI_Wtime();

    gather_start = MPI_Wtime();
    CALI_MARK_BEGIN(gathering);
    MPI_Gather(local_array, chunk_size, MPI_DOUBLE, arr, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    CALI_MARK_END(gathering);
    gather_end = MPI_Wtime();

    if (rank == 0) {
        final_sort_start = MPI_Wtime();
        CALI_MARK_BEGIN(final_sort);
        mergeSort(arr, 0, array_size - 1);
        CALI_MARK_END(final_sort);
        final_sort_end = MPI_Wtime();

        for (int i = 0; i < array_size; i++) {
            printf("%.2f ", arr[i]);
        }
        printf("\n");
    }

    whole_compute_end = MPI_Wtime();
    CALI_MARK_END(whole_computation);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", array_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    adiak::value("group_num", -1); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    if (rank == 0) {
        printf("Initialization Time: %f seconds\n", init_end - init_start);
        printf("Final Merge Sort Time: %f seconds\n", final_sort_end - final_sort_start);
    }
    printf("Local Sort Time (Rank %d): %f seconds\n", rank, local_sort_end - local_sort_start);
    printf("Gather Time (Rank %d): %f seconds\n", rank, gather_end - gather_start);
    printf("whole computation Time: %f seconds\n", whole_compute_end - whole_compute_start);

    free(local_array);
    if (rank == 0) {
        free(arr);
    }

    mgr.stop();
    mgr.flush();
    MPI_Finalize();

    return 0;
}

