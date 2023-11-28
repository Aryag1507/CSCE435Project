#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

void initialize_array(int local_A[], int local_n, int my_rank, int input_type)
{
	srandom(1 + my_rank);

	switch (input_type)
	{
	case 1:
		for (int i = 0; i < local_n; i++)
		{
			local_A[i] = random() % 100;
		}
		break;

	case 2:
		for (int i = 0; i < local_n; i++)
		{
			local_A[i] = i;
		}
		break;

	case 3:
		for (int i = 0; i < local_n; i++)
		{
			local_A[i] = local_n - i - 1;
		}
		break;

	case 4:
		for (int i = 0; i < local_n; i++)
		{
			local_A[i] = i;
		}
		for (int i = 0; i < local_n / 100; i++)
		{
			int idx1 = random() % local_n;
			int idx2 = random() % local_n;
			int temp = local_A[idx1];
			local_A[idx1] = local_A[idx2];
			local_A[idx2] = temp;
		}
		break;

	default:
		fprintf(stderr, "Invalid input type: %d\n", input_type);
		break;
	}
}

void Print_global_list(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm)
{
	int *A;
	int i, n;

	if (my_rank == 0)
	{

		n = p * local_n;
		A = (int *)malloc(n * sizeof(int));

		MPI_Gather(local_A, local_n, MPI_INT, A, local_n, MPI_INT, 0,
				   comm);
		printf("Global list:\n");
		for (i = 0; i < n; i++)
			printf("%d ", A[i]);
		printf("\n\n");
		free(A);
	}
	else
	{

		MPI_Gather(local_A, local_n, MPI_INT, A, local_n, MPI_INT, 0,
				   comm);
	}
}

void Merge_low(int local_A[], int temp_B[], int temp_C[], int local_n)
{
	int m = 0;
	int n = 0;
	int i;

	for (i = 0; i < local_n; i++)
	{
		if (m < local_n && (n >= local_n || local_A[m] <= temp_B[n]))
		{
			temp_C[i] = local_A[m];
			m++;
		}
		else
		{
			temp_C[i] = temp_B[n];
			n++;
		}
	}

	for (i = 0; i < local_n; i++)
		local_A[i] = temp_C[i];
}

void Merge_high(int local_A[], int temp_B[], int temp_C[], int local_n)
{
	int m = local_n - 1;
	int n = local_n - 1;
	int i;

	for (i = local_n - 1; i >= 0; i--)
	{
		if (m >= 0 && (n < 0 || local_A[m] > temp_B[n]))
		{
			temp_C[i] = local_A[m];
			m--;
		}
		else
		{
			temp_C[i] = temp_B[n];
			n--;
		}
	}

	for (i = 0; i < local_n; i++)
		local_A[i] = temp_C[i];
}

void gather_global_array(int global_A[], int local_A[], int local_n, int my_rank, int p, MPI_Comm comm)
{
	if (my_rank == 0)
	{
		// Allocate memory for the global array on the root process
		global_A = (int *)malloc(p * local_n * sizeof(int));
	}

	// Use MPI_Gather to collect local arrays from all processes
	MPI_Gather(local_A, local_n, MPI_INT, global_A, local_n, MPI_INT, 0, comm);

	// Note: The global array will be filled only on the root process (my_rank == 0)
}

// Function to check if the array is sorted
bool isSorted(int arr[], int n)
{
	for (int i = 0; i < n - 1; i++)
	{
		if (arr[i] > arr[i + 1])
		{
			return false; // Array is not sorted
		}
	}
	return true; // Array is sorted
}

int Compare(const void *a_p, const void *b_p)
{
	int a = *((int *)a_p);
	int b = *((int *)b_p);

	if (a < b)
		return -1;
	else if (a == b)
		return 0;
	else /* a > b */
		return 1;
} /* Compare */

void Odd_even_iter(int local_A[], int temp_B[], int temp_C[],
				   int local_n, int phase, int even_partner, int odd_partner,
				   int my_rank, int p, MPI_Comm comm)
{
	MPI_Status status;
	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_large");
	CALI_MARK_BEGIN("MPI_Send/recieve");
	if (phase % 2 == 0)
	{ /* even phase */
		if (even_partner >= 0)
		{ /* check for even partner */
			//
			//
			//
			MPI_Sendrecv(local_A, local_n, MPI_INT, even_partner, 0,
						 temp_B, local_n, MPI_INT, even_partner, 0, comm,
						 &status);
			//
			//
			//
			if (my_rank % 2 != 0) /* odd rank */
				// local_A have largest local_n ints from local_A and even_partner
				Merge_high(local_A, temp_B, temp_C, local_n);
			else /* even rank */
				// local_A have smallest local_n ints from local_A and even_partner
				Merge_low(local_A, temp_B, temp_C, local_n);
		}
	}
	else
	{ /* odd phase */
		if (odd_partner >= 0)
		{ /* check for odd partner */
			//
			//
			//
			MPI_Sendrecv(local_A, local_n, MPI_INT, odd_partner, 0,
						 temp_B, local_n, MPI_INT, odd_partner, 0, comm,
						 &status);
			//
			//
			//
			if (my_rank % 2 != 0) /* odd rank */
				Merge_low(local_A, temp_B, temp_C, local_n);
			else /* even rank */
				Merge_high(local_A, temp_B, temp_C, local_n);
		}
	}
	CALI_MARK_END("MPI_Send/recieve");
	CALI_MARK_END("comm_large");
	CALI_MARK_END("comm");
} /* Odd_even_iter */

void Sort(int local_A[], int local_n, int my_rank,
		  int p, MPI_Comm comm)
{
	int phase;
	int *temp_B, *temp_C;
	int even_partner; /* phase is even or left-looking */
	int odd_partner;  /* phase is odd or right-looking */

	/* Temporary storage used in merge-split */
	temp_B = (int *)malloc(local_n * sizeof(int));
	temp_C = (int *)malloc(local_n * sizeof(int));

	/* Find partners:  negative rank => do nothing during phase */
	if (my_rank % 2 != 0)
	{ /* odd rank */
		even_partner = my_rank - 1;
		odd_partner = my_rank + 1;
		if (odd_partner == p)
			odd_partner = MPI_PROC_NULL; // Idle during odd phase
	}
	else
	{ /* even rank */
		even_partner = my_rank + 1;
		if (even_partner == p)
			even_partner = MPI_PROC_NULL; // Idle during even phase
		odd_partner = my_rank - 1;
	}

	/* Sort local list using built-in quick sort */
	qsort(local_A, local_n, sizeof(int), Compare);

	for (phase = 0; phase < p; phase++)
		Odd_even_iter(local_A, temp_B, temp_C, local_n, phase,
					  even_partner, odd_partner, my_rank, p, comm);

	// deallocate memory
	free(temp_B);
	free(temp_C);
} /* Sort */

void Print_list(int local_A[], int local_n, int rank)
{
	int i;
	printf("%d: ", rank);
	for (i = 0; i < local_n; i++)
		printf("%d ", local_A[i]);
	printf("\n");
} /* Print_list */

void Print_local_lists(int local_A[], int local_n,
					   int my_rank, int p, MPI_Comm comm)
{
	int *A;
	int q;
	MPI_Status status;

	if (my_rank == 0)
	{
		A = (int *)malloc(local_n * sizeof(int));
		Print_list(local_A, local_n, my_rank);
		for (q = 1; q < p; q++)
		{
			MPI_Recv(A, local_n, MPI_INT, q, 0, comm, &status);
			Print_list(A, local_n, q);
		}
		free(A);
	}
	else
	{
		MPI_Send(local_A, local_n, MPI_INT, 0, 0, comm);
	}
} /* Print_local_lists */

void isSorted(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm)
{

	int global_n = local_n * p;
	int *global_A = NULL;
	bool sorted = true;

	if (my_rank == 0)
		global_A = (int *)malloc(global_n * sizeof(int));

	MPI_Gather(local_A, local_n, MPI_INT, global_A, local_n, MPI_INT, 0, comm);

	if (my_rank == 0)
	{

		for (int i = 0; i < global_n - 1; i++)
		{
			if (global_A[i] > global_A[i + 1])
			{
				sorted = false;
				break;
			}
		}

		free(global_A);
	}
}

int main(int argc, char *argv[])
{

	CALI_CXX_MARK_FUNCTION;
	CALI_MARK_BEGIN("main");

	int my_rank, p; // rank, number processes
	int *local_A;	// local list: size of local number of elements * size of int
	int global_n;	// number of elements in global list
	int local_n;	// number of elements in local list (process list)
	MPI_Comm comm;
	int input_type;

	MPI_Init(&argc, &argv);
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &my_rank);

	cali::ConfigManager mgr;
	mgr.start();

	if (argc != 3)
	{
		if (my_rank == 0)
		{
			// Usage(argv[0]);
		}
		MPI_Finalize();
		return 1; // Exit if the incorrect number of arguments are provided
	}

	global_n = atoi(argv[1]);	// Get the total number of elements
	input_type = atoi(argv[2]); // Get the input type

	// Ensure global_n is divisible by p
	if (global_n % p != 0)
	{
		if (my_rank == 0)
		{
			fprintf(stderr, "Global number of elements must be divisible by number of processes.\n");
			// Usage(argv[0]);
		}
		MPI_Finalize();
		return 1;
	}

	local_n = global_n / p; // Compute the number of elements per process

	local_A = (int *)malloc(local_n * sizeof(int));
	if (local_A == NULL)
	{
		fprintf(stderr, "Process %d: Unable to allocate memory for local array.\n", my_rank);
		MPI_Finalize();
		return 1;
	}

	CALI_MARK_BEGIN("data_init");
	initialize_array(local_A, local_n, my_rank, input_type);
	CALI_MARK_END("data_init");

	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
	Sort(local_A, local_n, my_rank, p, comm);
	CALI_MARK_END("comp_large");
	CALI_MARK_END("comp");

	CALI_MARK_BEGIN("correctness_check");
	isSorted(local_A, local_n, my_rank, p, comm);
	CALI_MARK_END("correctness_check");

	Print_global_list(local_A, local_n, my_rank, p, comm);

	free(local_A);

	adiak::init(NULL);
	adiak::launchdate();
	adiak::libraries();
	adiak::cmdline();
	adiak::clustername();
	adiak::value("Algorithm", "OddEvenSort");
	adiak::value("ProgrammingModel", "MPI");
	adiak::value("Datatype", "int");
	adiak::value("SizeOfDatatype", sizeof(int));
	adiak::value("InputSize", global_n);
	adiak::value("InputType", input_type);
	adiak::value("num_procs", p);
	adiak::value("group_num", 2);
	adiak::value("implementation_source", "Online/Handwritten");

	mgr.stop();
	mgr.flush();
	MPI_Finalize();
	CALI_MARK_END("main");

	return 0;
} /* main */
