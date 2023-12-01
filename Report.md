# CSCE 435 Group project

## 1. Group members:
1. Arya Gupta
2. Zia Uddin
3. Rijul Ranjan
4. JP Pham

---

We plan to use discord to communicate details regarding the project.

## 2. Project topic
MPI and CUDA-based parallelized implementation of sorting algorithms and performance analysis based on problem size and number of processors / threads.

## 2a. Brief project description (what algorithms will you be comparing and on what architectures)

* Merge Sort (Cuda)
* Merge Sort (MPI)
* Radix Sort (Cuda)
* Radix Sort (MPI)
* Odd-Even Sort (Cuda)
* Odd-Even Sort (MPI)
* Bucket Sort (Cuda)
* Bucket Sort (MPI)

## 2b. Pseudocode for each parallel algorithm

Merge Sort:
Pseudocode Source: https://pseudoeditor.com/guides/merge-sort

The only parameter for this merge sort algorithm is the array that we're trying to sort.

```
If len(array) > 1 Then
   // This is the point where the array is divided into two subarrays
   halfArray = len(array) / 2

   FirstHalf = array[:halfArray]
   // The first half of the data set

   SecondHalf = array[halfArray:]
   // The second half of the data set

   // Sort the two halves
   mergeSort(FirstHalf)
   mergeSort(SecondHalf)

   k = 0

   // Begin swapping values
   While i < len(FirstHalf) and j < len(SecondHalf)
      If FirstHalf[i] < SecondHalf[j] Then
        array[k] = FirstHalf[i]
        i += 1
      Else 
        array[k] = SecondHalf[j]
        j += 1
        k += 1
      EndIf
   EndWhile
EndIf
```
MPI Code Source: Used AI Assistance to implement
Cuda Code Source: Used AI Assistance to implement


Radix Sort: 
Pseudocode Source: https://www.codingeek.com/algorithms/radix-sort-explanation-pseudocode-and-implementation/

```
//It works same as counting sort for d number of passes.
//(Digits are numbered 1 to d from right to left.)
    for j = 1 to d do
        //A[]-- Initial Array to Sort
        int count[10] = {0};
        //Store the count of "keys" in count[]
        //key- it is number at digit place j
        for i = 0 to n do
         count[key of(A[i]) in pass j]++
        for k = 1 to 10 do
         count[k] = count[k] + count[k-1]
        //Build the resulting array by checking
        //new position of A[i] from count[k]
        for i = n-1 down to 0 do
         result[ count[key of(A[i])] ] = A[j]
         count[key of(A[i])]--
        //Now main array A[] contains sorted numbers
        //according to current digit place
        for i=0 to n do
          A[i] = result[i]
    end for(j)
end func
```
MPI Code Source: Used AI Assistance to implement + GitHub: https://github.com/mpseligson/radix
Cuda Code Source: https://github.com/ufukomer/cuda-radix-sort/blob/master/radix-sort/kernel.cu

Odd Even Sort:
Pseudocode Source: https://sortvisualizer.com/oddevensort/ 
```
void oddEvenSort(int* arr, int n) {
    bool sorted = false;
    while (!sorted) {
        sorted = true;
        for (int i = 1; i < n-1; i+=2) {
            if (arr[i] > arr[i+1]) {
                int temp = arr[i];
                arr[i] = arr[i+1];
                arr[i+1] = temp;
                sorted = false;
            }
        }

        for (int i = 0; i < n-1; i+=2) {
            if (arr[i] > arr[i+1]) {
                int temp = arr[i];
                arr[i] = arr[i+1];
                arr[i+1] = temp;
                sorted = false;
            }
        }
    }
}
```
MPI Code: https://github.com/ashantanu/Odd-Even-Sort-using-MPI/blob/master/oddEven.cpp
Cuda Code Source: https://github.com/Kshitij421/Odd-Even-Sort-using-Cuda-/blob/master/oddeven.cu

Bucket Sort:
Pseudocode Source: https://www.cs.umd.edu/class/spring2021/cmsc351-0201/files/bucketSort.pdf
```
BucketSort(A):
n = A.length
Let B[0, . . . , n − 1] be a new array
for i = 0 to n - 1
B[i] ← 0
for i = 1 to n
B[bnA[i]c] ← A[i]
for i = 0 to n-1
sort list B[i] using insertion sort
concatenate the lists B[0], B[1], . . . , B[n − 1]
return B
```
MPI Code Source: Used AI Assistance to implement
Cuda Code Source: Used AI Assistance to implement

## 2c. Evaluation Plan

Input sizes, Input types
* We plan to have varying input sizes for the unsorted arrays. We will use doubles (and integers for radix sort) for input types.
* Also plan to use varying number of processors vs the input size
* Array input sizes will be: {2^4, 2^8, 2^16, 2^20, 2^24}
  
Strong scaling (same problem size, increase number of processors/nodes)
* Will use strong scaling on all the algorithms. We want to see the relationship between time and the increase in processors. We’ll continually increase the number of processors and see the effect it has on the time and record it. Doing this, we’ll be able to see certain trends and make some data driven decisions based on what we see.
* (MPI) Increase number of cores while problem size is constant: {2, 4, 8, 16, 32}
* (CUDA) Increase number of threads while problem size is constant: {64, 128, 512, 1024}
  
Weak scaling (increase problem size, increase number of processors)
* Will use weak scaling on all algorithms. We want to see the relationship between time as both the problem size and number of processors. Here, we will increase both and measure the effect it has on the time.
* (MPI) Increase number of cores: {2,4,8,16,32}
* (CUDA) Increase number of threads: {64, 128, 512,1024}
* Increase problem sizes for both implementations: {2^4, 2^8, 2^16, 2^20, 2^24}
  
Number of threads in a block on the GPU
* Mainly applicable to CUDA 
* Increase the number of blocks: {1,8,16,64,128,1024}

## 3a. Caliper Instrumentation
We used the way mentioned in the original Report.md to generate our caliper files. Comm_small and comm_large were nested in comm and then comp_small and comp_large were nested in comp.

## 3b. Collect Metadata
We used Adiak to collect all our metadata. We modified the algorithm, programming model, inputsize, input type, num_procs, num_threads, num_blocks, group_number, and implementation sort for the respective algorithm implementations.

## 4. Algorithm Analysis


### Strong/Weak Scaling Graphs

#### Cuda Merge Sort
Here are the weak and strong scaling plots for CUDA Merge Sort

Weak scaling:

<img width="1000" height="600" alt="Screen Shot 2023-11-15 at 11 55 24 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/146949c51510d32b5c72116e42b89ca56dfc18bd/ProjectSort/Graphs/weak_scaling_mergesort_cuda.png">

Strong scaling:

<img width="1000" height="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/64009bfbbcec232879d75e71d34e8cc7a0b955ea/ProjectSort/Graphs/strong_scaling_mergesort_cuda.png">

As we can see from the weak scaling graphs, the merge sort algorithm is parallelizing well and generalizes well to larger problem sizes. Given that the total time is relatively constant for main and comp_large, we can see that as the input size is increasing and the number of threads are increasing, the algorithm is able to make effiecient usage of the computational resources for the given problem


For the strong scaling graphs, looking at main, random and sorted total time seems to decrease with an increase in threads. Reverse sorted and perturbed decrease upto a point and then increase again with higher counts, perhaps due to inefficiencies in the merge steps. For comp_large, we see a similar trend to the main function execution. For comm, sorted shows a relatively flat trend, indicating that communication time is not heavily impacted by thread count for sorted data. Random and reverse inputs show an initial decrease in communication time as and then increase. The perturbed input shows an irregular pattern, suggesting that the small amount of disorder in the array leads to unpredictable communication costs

#### Cuda Odd Even

Weak scaling:

<img width="1000" height="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/8f72034b55aebf98e5d7714eed9443bcaddcd8bc/ProjectSort/Graphs/weak_scaling_odd_even.png">

Strong scaling:

<img width="1000" height="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/8f72034b55aebf98e5d7714eed9443bcaddcd8bc/ProjectSort/Graphs/cuda_odd_even_strongscale.png">

As we can see from the weak scaling graphs, the odd even sort algorithm is parallelizing well and generalizes well to larger problem sizes. Given that the total time is relatively constant for main and comp_large, we can see that as the input size is increasing and the number of threads are increasing, the algorithm is able to make effiecient usage of the computational resources for the given problem

#### Radix Odd Even

Weak scaling:

<img width="1000" height="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/db151f56009316a3c0296878322e1bb9fc5002a4/ProjectSort/Graphs/radix_cuda_weak_scaling.png">

Look at the weak scaling graphs for CUDA Radix Sort, we did not see mostly constant execution time for comm and comp_large. This could be due to bottlenecks with the Radix Cuda implementation.  

#### Merge Sort MPI

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/c76f6eec46f4e5dc8f4d102510fc7587fb788bab/ProjectSort/Graphs/mergeSortMPI/Screen%20Shot%202023-11-30%20at%203.13.08%20AM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/c76f6eec46f4e5dc8f4d102510fc7587fb788bab/ProjectSort/Graphs/mergeSortMPI/Screen%20Shot%202023-11-30%20at%203.13.24%20AM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/c76f6eec46f4e5dc8f4d102510fc7587fb788bab/ProjectSort/Graphs/mergeSortMPI/Screen%20Shot%202023-11-30%20at%203.13.35%20AM.png">

Above are the weak scaling graphs for the MPI implementation of Merge Sort. We can see the algorithm is not really parallelizing well. As the input size and number of processors are both increasing, we are seeing an uptrend in the total time. In weak scaling, we want to see a relatively constant time, as that means the algorithm is able to effieciently utilizes the parallel computing resources for larger problem sizes, but that is not the case here. 

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/c76f6eec46f4e5dc8f4d102510fc7587fb788bab/ProjectSort/Graphs/mergeSortMPI/mergeSortMPI_strongScaling_comm.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/c76f6eec46f4e5dc8f4d102510fc7587fb788bab/ProjectSort/Graphs/mergeSortMPI/mergeSortMPI_strongScaling_comp_large.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/c76f6eec46f4e5dc8f4d102510fc7587fb788bab/ProjectSort/Graphs/mergeSortMPI/mergeSortMPI_strongScaling_main.png">

Looking at the strong scaling graphs, we can once again see that this algorithm is not very effecient. With higher thread counts, there should be a reduction in the total time taken by the algorithm to sort the array. It looks like the sub problems are more effeciently solved at lower thread counts, meaning the algorithm is not parallelizing well. There could be issues regarding overhead associated with communication within processors that may be causing this

<img width="800" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/mergeSortMPI/sample1.png">

<img width="800" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/mergeSortMPI/sample2.png">

<img width="800" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/mergeSortMPI/sample3.png">

<img width="800" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/mergeSortMPI/sample4.png">



#### Odd Even MPI

Strong scaling:

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-29%20at%2011.12.43%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/master/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-29%20at%2011.14.18%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-29%20at%2011.14.46%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-29%20at%2011.15.01%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-29%20at%2011.15.24%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-29%20at%2011.15.42%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-29%20at%2011.15.56%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-29%20at%2011.16.10%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-29%20at%2011.16.22%20PM.png">

As the number of processors increases, we see a sharp increase in total time, especially past the 60-processor mark. This could be due to the communication overhead between processors becoming significant as the number of processors increases. This indicates that the algorithm is not really scaling well with an increased number of processors.

Weak scaling:

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-30%20at%203.12.27%20AM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-30%20at%203.12.38%20AM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/oddEvenMPI/Screen%20Shot%202023-11-30%20at%203.12.50%20AM.png">

As the number of processors doubles, the time taken for both problem sizes increases. This increase should ideally be constant in a perfect weak scaling scenario since each processor is handling the same size of the problem, but we observe that the time increases. Odd even sort is inherently parallelized to an extent, so it is likely that communcation between processors is a large overhead with many processors which could explain why the algorithm total time is increasing sharply

#### Radix MPI

Strong scaling:

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/radixSortMPI/Screen%20Shot%202023-11-29%20at%2011.54.14%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/radixSortMPI/Screen%20Shot%202023-11-29%20at%2011.54.26%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/radixSortMPI/Screen%20Shot%202023-11-29%20at%2011.54.38%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/radixSortMPI/Screen%20Shot%202023-11-29%20at%2011.54.53%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/radixSortMPI/Screen%20Shot%202023-11-29%20at%2011.55.04%20PM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/radixSortMPI/Screen%20Shot%202023-11-29%20at%2011.55.16%20PM.png">

The time for sorting the array remains relatively flat initially as the number of processors increases. This indicates good scaling performance when the number of processors is small. The sharp increase at higher processor counts could imply that the overhead of communication between processors, synchronization, or other parallelization overheads is becoming much larger compared to the computation time. Each step of the radix sort requires distribution of data across processors based on the current digit being considered. The overhead of this likely is getting very large, which would explain the trend

Weak scaling:

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/radixSortMPI/Screen%20Shot%202023-11-30%20at%203.13.58%20AM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/radixSortMPI/Screen%20Shot%202023-11-30%20at%203.17.02%20AM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/6cb0d461fb629ad2eb2f52ed6bb8057a453b31cb/ProjectSort/Graphs/radixSortMPI/Screen%20Shot%202023-11-30%20at%203.17.12%20AM.png">

The fact that the lines are not flat (which would indicate ideal weak scaling) suggests that the algorithm does not perfectly maintain efficiency as the number of processors increases. The increase in time could be due to the overhead associated with managing radix sort's internal workings which can lead to much higher execution times


### Speed Up

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/e3c40819cc37558d67d0f4b5478cff49da7e7362/ProjectSort/Graphs/speedup_mergeCuda_random.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/e3c40819cc37558d67d0f4b5478cff49da7e7362/ProjectSort/Graphs/speedup_mergeCuda_random_compLarge.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/e3c40819cc37558d67d0f4b5478cff49da7e7362/ProjectSort/Graphs/speedup_mergeCuda_random_main.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/e3c40819cc37558d67d0f4b5478cff49da7e7362/ProjectSort/Graphs/speedup_mergeCuda_reverse_compLarge.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/e3c40819cc37558d67d0f4b5478cff49da7e7362/ProjectSort/Graphs/speedup_mergeCuda_reverse_compLarge.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/e3c40819cc37558d67d0f4b5478cff49da7e7362/ProjectSort/Graphs/speedup_mergeCuda_sorted_compLarge.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/e3c40819cc37558d67d0f4b5478cff49da7e7362/ProjectSort/Graphs/speedup_mergeCuda_sorted_main.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/e3c40819cc37558d67d0f4b5478cff49da7e7362/ProjectSort/Graphs/speedup_mergeCuda_onep_compLarge.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/e3c40819cc37558d67d0f4b5478cff49da7e7362/ProjectSort/Graphs/speedup_mergeCuda_onep_main.png">




### Algorithm Comparison Graphs (MPI).

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/855b2bf3576d5d15b81971752c70f346981feb9a/ProjectSort/Graphs/ComparisonGraphs/Screen%20Shot%202023-11-30%20at%202.33.50%20AM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/855b2bf3576d5d15b81971752c70f346981feb9a/ProjectSort/Graphs/ComparisonGraphs/Screen%20Shot%202023-11-30%20at%202.33.58%20AM.png">

<img width="600" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/blob/855b2bf3576d5d15b81971752c70f346981feb9a/ProjectSort/Graphs/ComparisonGraphs/Screen%20Shot%202023-11-30%20at%202.34.13%20AM.png">


Above are the strong scaling comparison graphs for a constant input size of 2^18 for Radix, Odd Even, and Merge sort. We see a similar trend for all three regions across the board for the algorithms. Odd Even seems to be the one that is most greatly effected by increasing the number of processors. Total time goes up drastically, an explanation for which could be the inherent parallelism associated with Odd Even sort. Resource contention could be a limiting factor with increasing processor counts. Merge sort is a divide and conquer algorithm which divides the array into multiple sub-problems, which likely explains why the total time is relatively constant across the processor counts. At one point, the thread count is sufficient for the problem size, so additional processors really have no effect on the total time. Radix sort requires data to be exchanged between processes, especially during the redistribution of keys based on their digits. As the number of processes increases, the communication overhead can increase due to more messages being passed around. However, up to a certain point, the increased parallelism may compensate for this overhead, which could explain why the time first increases until 24 processes. Contention problems at higher processor counts may explain why the total time seems to go up after 64 or so processors

With bucketsort, we discovered that it was difficult to parallelize each “bucket” when sorting and did not provide much of a performance increase over the sequential algorithm, so we decided to omit it from our algorithm implementations
