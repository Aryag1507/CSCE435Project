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

const char* mainn = "main";
const char* data_init = "data_init";
const char* comm = "comm";
const char* MPIBarrier = "MPI_Barrier";
const char* comm_small = "comp_small";
const char* comm_large = "comm_large";
const char* MPIBcast = "MPI_Bcast";
const char* MPISend = "MPI_Send";
const char* cudaMemcpy = "cudaMemcpy";

const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";


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


        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        merge(arr, l, m, r);
        CALI_MARK_END(comp_small);
        CALI_MARK_END(comp);
    }
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
    CALI_MARK_BEGIN(mainn);
    
    cali::ConfigManager mgr;
    mgr.start();

    init_start = MPI_Wtime();
    CALI_MARK_BEGIN(data_init);
    if (rank == 0) {
        arr = (double *)malloc(sizeof(double) * array_size);
        for (int i = 0; i < array_size; i++) {
            arr[i] = (double)rand() / RAND_MAX;
        }
    }
    init_end = MPI_Wtime();
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(arr + i * chunk_size, chunk_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        memcpy(local_array, arr, chunk_size * sizeof(double));
    } else {
        MPI_Recv(local_array, chunk_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    local_sort_start = MPI_Wtime();
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    mergeSort(local_array, 0, chunk_size - 1);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    local_sort_end = MPI_Wtime();

    gather_start = MPI_Wtime();
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    if (rank == 0) {
        double* recv_buffer = (double *)malloc(sizeof(double) * chunk_size);
        for (int i = 1; i < size; i++) {
            MPI_Recv(recv_buffer, chunk_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memcpy(arr + i * chunk_size, recv_buffer, chunk_size * sizeof(double));
        }
        memcpy(arr, local_array, chunk_size * sizeof(double));
        free(recv_buffer);
    } else {
        MPI_Send(local_array, chunk_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    gather_end = MPI_Wtime();

    if (rank == 0) {
        final_sort_start = MPI_Wtime();
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        mergeSort(arr, 0, array_size - 1);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
        final_sort_end = MPI_Wtime();

        for (int i = 0; i < array_size; i++) {
            printf("%.2f ", arr[i]);
        }
        printf("\n");
    }

    whole_compute_end = MPI_Wtime();
    CALI_MARK_END(mainn);

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
    adiak::value("group_num", 2); // The number of your group (integer, e.g., 1, 10)
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

