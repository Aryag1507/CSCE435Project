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

## 4a. Vary the following parameters

We generated 2 caliper files with each algorithm while testing different metrics and keeping the rest constant when doing this. 

For MPI Merge Sort, we changed the data size and saw the difference it makes on the time and saw that with increasing data size, the time it took to complete the whole computation increased. The computation and communication time was also relative to the whole computation time and increased with the increased data size.

<img width="490" alt="Screen Shot 2023-11-15 at 11 55 24 PM" src="https://github.com/Aryag1507/CSCE435Project/assets/62392738/9e0d87c8-faa4-4b72-9644-d860bd387ff8">

For MPI Radix Sort, we changed the number of processors. We noticed that increasing the number of processors helped decrease the runtime due to the workload being distributed throughout multiple processors. The computation and communication time were relative to the overall runtime, and were also decreased with the decreased overall runtime.

<img width="495" alt="Screen Shot 2023-11-15 at 11 54 20 PM" src="https://github.com/Aryag1507/CSCE435Project/assets/62392738/b7a19caa-c865-4efd-8e26-2c251700768f">

## 4b. Hints for performance analysis

Similarly, we did this for all the algorithm implementations and noticed that increasing the number of processors/threads reduced runtime, increasing the input sizes increased runtime, changing the inputTypes changed the runtime depending on the inputType. The sorted arrays were the fastest since the arrays were already sorted and nothing else was required to be done. The reverse sorted were generally the slowest since the algorithm would for sure have to change the placement of the numbers since everything was basically sorted backwards. The random and 1% perturbed were in the middle with runtime since there are random elements involved with it. The communication and computation time were relative to the whole computation time, so if the whole computation time was decreased, so was the communication and computation time.


## Note:

We were unable to get thicket to work to graph our plots, therefore we graphed what we could manually. A lot of our nodes were killed and that prevented us from completing the project with thicket.

