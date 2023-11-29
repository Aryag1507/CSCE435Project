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

void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary arrays
    int* L = new int[n1];
    int* R = new int[n2];

    // Copy data to temp arrays L[] and R[]
    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Merge the temp arrays back into arr[l..r]
    int i = 0;
    int j = 0;
    int k = l;
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

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    // Deallocate temporary arrays
    delete[] L;
    delete[] R;
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
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
    mergeSort(h_data, 0, array_length - 1);
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
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "Sequential"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", array_length); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("group_num", 2); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

}
