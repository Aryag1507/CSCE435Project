#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

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

// Function to perform Odd-Even Sort
void oddEvenSort(float arr[], int n) {
    bool isSorted = false;

    while (!isSorted) {
        isSorted = true;

        for (int i = 1; i <= n - 2; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }

        for (int i = 0; i <= n - 2; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
    }
}

// Function to print an array
// void printArray(float arr[], int n) {
//     for (int i = 0; i < n; i++) {
//         cout << arr[i] << " ";
//     }
//     cout << endl;
// }

int main(int argc, char* argv[]) {
    CALI_CXX_MARK_FUNCTION;

    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <size> <type>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    char* inputType = argv[2];

    float *arr = new float[n];

    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);
    // Generate array based on input type
    if (strcmp(inputType, "Sorted") == 0) {
        for (int i = 0; i < n; ++i) arr[i] = static_cast<float>(i) / n;
    } else if (strcmp(inputType, "Random") == 0) {
        srand(time(0));
        for (int i = 0; i < n; ++i) arr[i] = static_cast<float>(rand()) / RAND_MAX;
    } else if (strcmp(inputType, "Reverse") == 0) {
        for (int i = 0; i < n; ++i) arr[i] = static_cast<float>(n - i - 1) / n;
    } else if (strcmp(inputType, "Perturbed") == 0) {
        for (int i = 0; i < n; ++i) arr[i] = static_cast<float>(i) / n;
        int perturbCount = n / 100;
        srand(time(0));
        while (perturbCount--) {
            int idx1 = rand() % n;
            int idx2 = rand() % n;
            swap(arr[idx1], arr[idx2]);
        }
    } else {
        cerr << "Invalid input type. Use: Sorted, Random, Reverse, or Perturbed." << endl;
        delete[] arr;
        return 1;
    }
    CALI_MARK_END(data_init);

    // cout << "Original array:" << endl;
    // printArray(arr, n);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    oddEvenSort(arr, n);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // cout << "Sorted array:" << endl;
    // printArray(arr, n);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "oddEven"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "sequential"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("group_num", 2); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();
    delete[] arr;
    return 0;
}
