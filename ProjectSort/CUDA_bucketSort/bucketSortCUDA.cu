#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <errno.h>

#define BUCKETS 10
#define RANGE 10

__global__ void bucketSort(int N, int *d_input, int *d_bucket_counters, int *d_bucket_offsets, int *d_output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        // Determine the bucket
        int bucket = d_input[idx] / RANGE;

        // Calculate index where to put the element
        int bucket_idx = atomicAdd(&d_bucket_counters[bucket], 1);
        
        // Wait until all updates to bucket counters are done
        __syncthreads();

        // Compute global index for the output array
        int output_idx = d_bucket_offsets[bucket] + bucket_idx;

        // Place the element into the right position in the output array
        d_output[output_idx] = d_input[idx];
    }
}

void randomArray(int *array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % (RANGE * BUCKETS);
    }
}

void sortBuckets(int *array, int *bucket_sizes) {
    for (int i = 0; i < BUCKETS; i++) {
        if (bucket_sizes[i] > 0) {
            // You would call a GPU sorting function here, like bitonic sort, for each bucket.
            // For illustration, we use qsort from the C standard library, which is not parallel.
            qsort(array, bucket_sizes[i], sizeof(int), [](const void *a, const void *b) {
                return (*(int*)a - *(int*)b);
            });
            array += bucket_sizes[i]; // Move to the next bucket
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    errno = 0;
    long N = strtol(argv[1], NULL, 10);
    if (errno != 0 || N <= 0 || N > INT_MAX) {
        perror("Invalid value for N");
        exit(EXIT_FAILURE);
    }

    int *h_input = (int*)malloc(N * sizeof(int));
    int *h_output = (int*)malloc(N * sizeof(int));
    int *h_bucket_counters = (int*)calloc(BUCKETS, sizeof(int));
    int *d_input, *d_output, *d_bucket_counters, *d_bucket_offsets;

    // Initialize the input array with random numbers
    randomArray(h_input, N);

    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));
    cudaMalloc((void**)&d_bucket_counters, BUCKETS * sizeof(int));
    cudaMalloc((void**)&d_bucket_offsets, BUCKETS * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bucket_counters, h_bucket_counters, BUCKETS * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate offsets for each bucket
    // This can be done using an exclusive prefix sum (scan), but here we do it manually for simplicity.
    int *h_bucket_offsets = (int*)malloc(BUCKETS * sizeof(int));
    int offset = 0;
    for (int i = 0; i < BUCKETS; i++) {
        h_bucket_offsets[i] = offset;
        offset += h_bucket_counters[i];
    }
    cudaMemcpy(d_bucket_offsets, h_bucket_offsets, BUCKETS * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 dimGrid((N + 255) / 256);
    dim3 dimBlock(256);
    bucketSort<<<dimGrid, dimBlock>>>(N, d_input, d_bucket_counters, d_bucket_offsets, d_output);

    // Copy back the bucket counters and the output array
    cudaMemcpy(h_bucket_counters, d_bucket_counters, BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Now we need to sort the contents of each bucket
    sortBuckets(h_output, h_bucket_counters);

        printf("Sorted array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Free resources and exit
    free(h_input);
    free(h_output);
    free(h_bucket_counters);
    free(h_bucket_offsets);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bucket_counters);
    cudaFree(d_bucket_offsets);

    return 0;
}
