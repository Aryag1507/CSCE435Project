#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            cout << "Usage: mpiexec -n <number_of_processors> ./bucketsort <array_size> <number_of_processors>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int arraySize = stoi(argv[1]);
    int numProcessors = stoi(argv[2]);

    float *arr = new float[arraySize];

    // Initialize the array with random values (you can use your own randomization logic here)
    for (int i = 0; i < arraySize; i++) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    bucketSort(arr, arraySize, rank, size);

    // Print sorted array from rank 0
    if (rank == 0) {
        cout << "Sorted array is \n";
        for (int i = 0; i < arraySize; i++) {
            cout << arr[i] << " ";
        }
        cout << endl;
    }

    delete[] arr;
    MPI_Finalize();
    return 0;
}
