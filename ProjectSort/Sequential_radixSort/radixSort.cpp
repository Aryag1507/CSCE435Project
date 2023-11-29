#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <iostream>
#include <vector>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";

const char *mainn = "main";

const char* data_init = "data_init";

void rng(int* arr, int n, string input) {
    if(input == "random"){
        int seed = 13516095;
        srand(seed);
        for(long i = 0; i < n; i++) {
            arr[i] = (int)rand() % 10000;
        }
    } else if(input == "sorted"){
        for (int i = 0; i < n; i++) {
            arr[i] = i;
        }
    } else if(input == "reverse_sorted"){
        for (int i = 0; i < n; i++) {
            arr[i] = n - i - 1;
        }
    } else if(input == "one_percent"){
        for (int i = 0; i < n; i++) {
            arr[i] = i;
        }
        for (int i = 0; i < n / 100; i++) {
            int idx1 = random() % n;
            int idx2 = random() % n;
            int temp = arr[idx1];
            arr[idx1] = arr[idx2];
            arr[idx2] = temp;
        }
    }
}

int getMax(int arr[], int n) {
    int mx = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > mx)
            mx = arr[i];
    return mx;
}

void countSort(int arr[], int n, int exp) {
    int* output = new int[n];
    int i, count[10] = {0};

    for (i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    for (i = 0; i < n; i++)
        arr[i] = output[i];

    delete[] output;
}

void radixSort(int arr[], int n) {

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small); 
    int m = getMax(arr, n);
    CALI_MARK_END(comp_small); 
    CALI_MARK_END(comp);
    

    for (int exp = 1; m / exp > 0; exp *= 10)
        countSort(arr, n, exp);
}


int main(int argc, char const *argv[])
{
    int array_length = atoi(argv[1]);
    string input_type = argv[2];

    int* h_data = new int[array_length];

    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);

    for (int i = 0; i < array_length; ++i) {
        h_data[i] = rand();
    }

    rng(h_data, array_length, input_type);

    CALI_MARK_END(data_init);

    printf("Original Array:\n");
    for (int i = 0; i < array_length; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large); 
    radixSort(h_data, array_length);
    CALI_MARK_END(comp_large); 
    CALI_MARK_END(comp);

    printf("Sorted Array:\n");
    for (int i = 0; i < array_length; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "RadixSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "Sequential"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", array_length); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("group_num", 2); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

}