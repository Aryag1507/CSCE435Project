#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* data_init = "data_init";
const char* comm = "comm";
const char* MPIBarrier = "MPI_Barrier";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* MPIRecv = "MPI_Recv";
const char* MPISend = "MPI_Send";

const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correctness_Check = "correctness_check";

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

void initializeArray(double arr[], int size, const char* type) {
    if (strcmp(type, "Sorted") == 0) {
        for (int i = 0; i < size; i++)
            arr[i] = (double)i;
    } else if (strcmp(type, "ReverseSorted") == 0) {
        for (int i = 0; i < size; i++)
            arr[i] = (double)(size - i);
    } else if (strcmp(type, "1%perturbed") == 0) {
        for (int i = 0; i < size; i++) {
            arr[i] = (double)i;
            if (rand() % 100 < 1) // 1% chance to perturb the value
                arr[i] += (double)(rand() % 10) - 5; // small perturbation
        }
    } else {
        // Default to Random
        for (int i = 0; i < size; i++)
            arr[i] = (double)rand() / RAND_MAX;
    }
}

bool correctness_check(double *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false; // Array is not sorted
        }
    }
    return true; // Array is sorted
}

int main(int argc, char **argv) {
    CALI_CXX_MARK_FUNCTION;
    // . build.sh
    // sbatch mpi.grace_job

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <array_size> <input_type>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int array_size = atoi(argv[1]);
    const char* input_type = argv[2];
    int chunk_size = array_size / size;
    double *local_array = (double *)malloc(sizeof(double) * chunk_size);
    double *arr = NULL;

    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);
    if (rank == 0) {
        arr = (double *)malloc(sizeof(double) * array_size);
        initializeArray(arr, array_size, input_type);
    }
    CALI_MARK_END(data_init);
    

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Scatter(arr, chunk_size, MPI_DOUBLE, local_array, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Start measuring time for local sort
    double local_start, local_end, local_elapsed;
    local_start = MPI_Wtime();

    // Perform local sort
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    mergeSort(local_array, 0, chunk_size - 1);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);


    // Stop measuring time for local sort
    local_end = MPI_Wtime();
    local_elapsed = local_end - local_start;


    // Gather sorted subarrays at root
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Gather(local_array, chunk_size, MPI_DOUBLE, arr, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    double final_sort_start, final_sort_end, final_sort_elapsed;
    // Final merge sort at root
    if (rank == 0) {
        final_sort_start = MPI_Wtime();
        mergeSort(arr, 0, array_size - 1);
        final_sort_end = MPI_Wtime();
        final_sort_elapsed = final_sort_end - final_sort_start;

        // Check for correctness
        CALI_MARK_BEGIN(correctness_Check);
        bool is_sorted = correctness_check(arr, array_size);
        if (is_sorted) {
            printf("The array is correctly sorted.\n");
        } else {
            printf("Error: The array is not correctly sorted.\n");
        }
        CALI_MARK_END(correctness_Check);

        // Optionally, print sorted array
        // for (int i = 0; i < array_size; i++) {
        //     printf("%.2f ", arr[i]);
        // }
        // printf("\n");
    }

    // Gather total elapsed time
    double total_elapsed, min_elapsed, max_elapsed, avg_elapsed;
    MPI_Reduce(&local_elapsed, &min_elapsed, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_elapsed, &total_elapsed, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


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
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    adiak::value("group_num", 2); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").


    if (rank == 0) {
        avg_elapsed = total_elapsed / size;
        double *all_elapsed_times = (double *)malloc(sizeof(double) * size);
        all_elapsed_times[0] = local_elapsed;

        // Receive elapsed times from other ranks
        for (int i = 1; i < size; i++) {
            MPI_Recv(&all_elapsed_times[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Calculate variance
        double sum = 0.0, mean, variance = 0.0, std_deviation;

        // Compute sum of all elements
        for (int i = 0; i < size; i++) {
            sum += all_elapsed_times[i];
        }
        mean = sum / size;

        // Compute variance
        for (int i = 0; i < size; i++) {
            variance += pow(all_elapsed_times[i] - mean, 2);
        }
        variance /= size;

        // Calculate standard deviation
        std_deviation = sqrt(variance);

        // Print performance metrics
        printf("Array size: %d\n", array_size);
        printf("input type: %s\n", input_type);
        printf("Min Time/Rank: %f seconds\n", min_elapsed);
        printf("Max Time/Rank: %f seconds\n", max_elapsed);
        printf("Avg Time/Rank: %f seconds\n", avg_elapsed);
        printf("Total Time: %f seconds\n", total_elapsed);
        printf("Variance Time/Rank: %f\n", variance);
        printf("Standard Deviation Time/Rank: %f\n", std_deviation);
    } else {
        // Send local elapsed time to root for variance calculation
        MPI_Send(&local_elapsed, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }


    // Clean up
    free(local_array);
    if (rank == 0) {
        free(arr);
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(MPIBarrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(MPIBarrier);
    CALI_MARK_END(comm);

    mgr.stop();
    mgr.flush();
    MPI_Finalize();
    return 0;
}

