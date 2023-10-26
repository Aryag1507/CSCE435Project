# CSCE 435 Group project

## 1. Group members:
1. Arya Gupta
2. Zia Uddin
3. Rijul Ranjan
4. JP Pham

---

## 2. _due 10/25_ Project topic
Our project topic is going to be sorting algorithms. We'll work on implementing several common sorting algorithms using MPI and Cuda.

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

We plan to do two sorting algorithms: Merge Sort and Radix Sort. We plan to use MPI for merge sort and CUDA for radix sort.

Merge Sort (https://pseudoeditor.com/guides/merge-sort): 

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

We plan to use the algorithm in the link below to parallelize the merge sort: 
https://cse.buffalo.edu/faculty/miller/Courses/CSE702/Swati.Nair-Fall-2018.pdf

Radix Sort (https://www.codingeek.com/algorithms/radix-sort-explanation-pseudocode-and-implementation/):

Takes in two parameters A, d. A is the array to sort and each value in A is a d-digit integer.

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
        for i = n-1 downto 0 do
         result[ count[key of(A[i])] ] = A[j]
         count[key of(A[i])]--
        //Now main array A[] contains sorted numbers
        //according to current digit place
        for i=0 to n do
          A[i] = result[i]
    end for(j)
end func
```

These are a few links we plan to refer to when we create our parallel algorithm for radix sort:
* https://github.com/ufukomer/cuda-radix-sort/blob/master/radix-sort/kernel.cu
* https://github.com/mark-poscablo/gpu-radix-sort

What we plan to do to show the changes with the parallel algorithm:
* We plan to compare several different variable between the parallel algorithms for both merge and radix sort with there sequential counterparts. We plan to calculate the speedup for both algorithms and compare. We also plan to figure out a way to analyze our performance of computation and communication for our algorithms.
