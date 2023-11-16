#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
using namespace std;

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

enum ArrayType
{
    RANDOM,
    SORTED,
    REVERSE_SORTED,
    PERTURBED
};

const char* mainn = "main";
const char* data_init = "data_init";
const char* comm = "comm";
const char* MPIBarrier = "MPI_Barrier";
const char* comm_small = "comp_small";
const char* comm_large = "comm_large";
const char* MPIRecv = "MPI_Recv";
const char* MPIGather = "MPI_Gather";
const char* MPISend = "MPI_Send";
const char* cudaMemcpy = "cudaMemcpy";

const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

// Function to sort arr[] of size n using bucket sort in parallel
void bucketSort(float arr[], int n, int rank, int size) {
    // Calculate the range of data for each process
    int range = n / size;
    int start = rank * range;
    int end = (rank + 1) * range;

    // Each process creates its own bucket
    vector<float> local_bucket;

    // Distribute data among processes
    for (int i = start; i < end; i++) {
        local_bucket.push_back(arr[i]);
    }

    // Sort individual local buckets
    sort(local_bucket.begin(), local_bucket.end());

    // Gather sorted data from each process
    vector<float> all_sorted(n); // Use a vector of size n for gathering

    MPI_Gather(local_bucket.data(), range, MPI_FLOAT, all_sorted.data(), range, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Only rank 0 will have the complete sorted data
    if (rank == 0) {
        // Merge sorted data from each process
        // This is a simplified merge step, can be optimized further
        sort(all_sorted.begin(), all_sorted.end());
        copy(all_sorted.begin(), all_sorted.end(), arr);
    }
}

void generateArray(float arr[], int arraySize, ArrayType type, int rank) {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count() + rank;
    mt19937 generator(seed);

    switch (type) {
        case RANDOM:
            for (int i = 0; i < arraySize; ++i) {
                arr[i] = static_cast<float>(generator()) / generator.max();
            }
            break;
        case SORTED:
            for (int i = 0; i < arraySize; ++i) {
                arr[i] = static_cast<float>(i) / arraySize;
            }
            break;
        case REVERSE_SORTED:
            for (int i = 0; i < arraySize; ++i) {
                arr[i] = static_cast<float>(arraySize - i - 1) / arraySize;
            }
            break;
        case PERTURBED:
            for (int i = 0; i < arraySize; ++i) {
                arr[i] = static_cast<float>(i) / arraySize;
                arr[i] += static_cast<float>(generator()) / (generator.max() * 100);
            }
            break;
    }
}

int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double init_start, init_end, local_sort_start, local_sort_end, gather_start, gather_end, final_sort_start, final_sort_end;
    double whole_compute_start, whole_compute_end;
    double start_time, end_time;

    if (argc != 4)
    {
        if (rank == 0)
        {
            cout << "Usage: mpiexec -n <number_of_processors> ./bucketsort <array_size> <number_of_processors> <array_type>" << endl;
            cout << "Array types: 0 (Random), 1 (Sorted), 2 (Reverse Sorted), 3 (Perturbed)" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    start_time = MPI_Wtime(); // Start the timer
    CALI_MARK_BEGIN(mainn);

    cali::ConfigManager mgr;
    mgr.start();

    int arraySize = atoi(argv[1]);
    int numProcessors = atoi(argv[2]);
    ArrayType type = static_cast<ArrayType>(atoi(argv[3]));

    float *arr = new float[arraySize];

    init_start = MPI_Wtime();
    CALI_MARK_BEGIN(data_init);
    // Generate array based on user's choice
    generateArray(arr, arraySize, type, rank);
    init_end = MPI_Wtime();
    CALI_MARK_END(data_init);


    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    bucketSort(arr, arraySize, rank, size);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    end_time = MPI_Wtime(); // End the timer

    // Print sorted array from rank 0
    if (rank == 0)
    {
        // cout << "Sorted array is \n";
        // for (int i = 0; i < arraySize; i++)
        // {
        //     cout << arr[i] << " ";
        // }
        // cout << endl;

        double elapsed_time = end_time - start_time;
        cout << "Total time taken: " << elapsed_time << " seconds." << endl;
    }
    CALI_MARK_END(mainn);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "bucketSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", arraySize); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    adiak::value("group_num", 2); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online/AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    delete[] arr;

    mgr.stop();
    mgr.flush();
    MPI_Finalize();

    return 0;
}
