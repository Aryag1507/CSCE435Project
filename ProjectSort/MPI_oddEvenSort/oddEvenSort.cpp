/*
 This code was referenced from: https://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/oddEvenSort/oddEven.html and the book stated on the website.
 book: An Introduction to Parallel Programming, Peter Pacheco, Morgan Kaufmann Publishers, 2011
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <limits.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const int RMAX = 100;
typedef enum
{
	LIST_RANDOM,
	LIST_SORTED,
	LIST_REVERSE_SORTED,
	LIST_PERTURBED
} inputList;

const char *inputListToString(inputList inputType)
{
	switch (inputType)
	{
	case LIST_RANDOM:
		return "Random";
	case LIST_SORTED:
		return "Sorted";
	case LIST_REVERSE_SORTED:
		return "Reverse Sorted";
	case LIST_PERTURBED:
		return "1% Perturbed";
	default:
		return "Unknown";
	}
}

void Generate_list(int local_A[], int local_n, int my_rank, inputList inputType)
{
	srandom(my_rank + 1); // Set seed for random generator
	switch (inputType)
	{
	case LIST_RANDOM:
		for (int i = 0; i < local_n; i++)
		{
			local_A[i] = random() % RMAX;
		}
		break;
	case LIST_SORTED:
		for (int i = 0; i < local_n; i++)
		{
			local_A[i] = i; 
		}
		break;
	case LIST_REVERSE_SORTED:
		for (int i = 0; i < local_n; i++)
		{
			local_A[i] = local_n - i - 1; 
		}
		break;
	case LIST_PERTURBED:
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
	}
} 


void Usage(char *program)
{
	fprintf(stderr, "usage:  mpirun -np <p> %s <global_n> <inputType>\n", program);
	fprintf(stderr, "   - p: the number of processes\n");
	fprintf(stderr, "   - global_n: number of elements in global list (must be evenly divisible by p)\n");
	fprintf(stderr, "   - inputType: type of list to generate ('r' for Random, 's' for Sorted, 'rs' for Reverse Sorted, 'p' for Perturbed)\n");
	fflush(stderr);
} 
void Get_args(int argc, char *argv[], int *global_n_p, int *local_n_p,
			  inputList *inputType_p, int my_rank, int p, MPI_Comm comm)
{
	if (my_rank == 0)
	{
		
		if (argc != 3)
		{
			Usage(argv[0]);
			*global_n_p = -1; 
		}
		else
		{
			*global_n_p = atoi(argv[1]);
			if (strcmp(argv[2], "r") == 0)
			{
				*inputType_p = LIST_RANDOM;
			}
			else if (strcmp(argv[2], "s") == 0)
			{
				*inputType_p = LIST_SORTED;
			}
			else if (strcmp(argv[2], "rs") == 0)
			{
				*inputType_p = LIST_REVERSE_SORTED;
			}
			else if (strcmp(argv[2], "p") == 0)
			{
				*inputType_p = LIST_PERTURBED;
			}
			if (*global_n_p % p != 0)
			{
				Usage(argv[0]);
				*global_n_p = -1;
			}
		}
	} 
	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_small");
	CALI_MARK_BEGIN("MPI_Bcast");
	MPI_Bcast(global_n_p, 1, MPI_INT, 0, comm);
	CALI_MARK_END("MPI_Bcast");
	CALI_MARK_END("comm_small");
	CALI_MARK_END("comm");
	if (*global_n_p <= 0)
	{
		MPI_Finalize();
		exit(-1);
	}
	*local_n_p = *global_n_p / p;
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
} 
void Sort(int local_A[], int local_n, int my_rank,
		  int p, MPI_Comm comm)
{
	int phase;
	int *temp_B, *temp_C;
	int even_partner;
	int odd_partner;  
	temp_B = (int *)malloc(local_n * sizeof(int));
	temp_C = (int *)malloc(local_n * sizeof(int));

	if (my_rank % 2 != 0)
	{ 
		even_partner = my_rank - 1;
		odd_partner = my_rank + 1;
		if (odd_partner == p)
			odd_partner = MPI_PROC_NULL; 
	}
	else
	{ 
		even_partner = my_rank + 1;
		if (even_partner == p)
			even_partner = MPI_PROC_NULL; 
		odd_partner = my_rank - 1;
	}

	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
	qsort(local_A, local_n, sizeof(int), Compare);
	CALI_MARK_END("comp_large");
	CALI_MARK_END("comp");

	for (phase = 0; phase < p; phase++)
		Odd_even_iter(local_A, temp_B, temp_C, local_n, phase,
					  even_partner, odd_partner, my_rank, p, comm);

	
	free(temp_B);
	free(temp_C);
} 

void Odd_even_iter(int local_A[], int temp_B[], int temp_C[], int local_n, int phase, int even_partner, int odd_partner, int my_rank, int p, MPI_Comm comm)
{
	MPI_Status status;

	if (phase % 2 == 0)
	{ 
		if (even_partner >= 0)
		{ 
			CALI_MARK_BEGIN("comm");
			CALI_MARK_BEGIN("comm_small");
			CALI_MARK_BEGIN("MPI_Sendrecv");
			MPI_Sendrecv(local_A, local_n, MPI_INT, even_partner, 0,
						 temp_B, local_n, MPI_INT, even_partner, 0, comm,
						 &status);
			CALI_MARK_END("MPI_Sendrecv");
			CALI_MARK_END("comm_small");
			CALI_MARK_END("comm");
			if (my_rank % 2 != 0) 
				Merge_high(local_A, temp_B, temp_C, local_n);
			else
				Merge_low(local_A, temp_B, temp_C, local_n);
		}
	}
	else
	{ 
		if (odd_partner >= 0)
		{ 
			CALI_MARK_BEGIN("comm");
			CALI_MARK_BEGIN("comm_small");
			CALI_MARK_BEGIN("MPI_Sendrecv");
			MPI_Sendrecv(local_A, local_n, MPI_INT, odd_partner, 0, temp_B, local_n, MPI_INT, odd_partner, 0, comm, &status);
			CALI_MARK_END("MPI_Sendrecv");
			CALI_MARK_END("comm_small");
			CALI_MARK_END("comm");
			if (my_rank % 2 != 0) 
				Merge_low(local_A, temp_B, temp_C, local_n);
			else 
				Merge_high(local_A, temp_B, temp_C, local_n);
		}
	}
} 
void Merge_low(int local_A[], int temp_B[], int temp_C[], int local_n)
{
	int ai, bi, ci;

	ai = bi = ci = 0;
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_small");
	while (ci < local_n)
	{
		if (local_A[ai] <= temp_B[bi])
		{
			temp_C[ci] = local_A[ai];
			ci++;
			ai++;
		}
		else
		{
			temp_C[ci] = temp_B[bi];
			ci++;
			bi++;
		}
	}
	memcpy(local_A, temp_C, local_n * sizeof(int));
	CALI_MARK_END("comp_small");
	CALI_MARK_END("comp");
} 

void Merge_high(int local_A[], int temp_B[], int temp_C[], int local_n)
{
	int ai, bi, ci;

	ai = local_n - 1;
	bi = local_n - 1;
	ci = local_n - 1;
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_small");
	while (ci >= 0)
	{
		if (local_A[ai] >= temp_B[bi])
		{
			temp_C[ci] = local_A[ai];
			ci--;
			ai--;
		}
		else
		{
			temp_C[ci] = temp_B[bi];
			ci--;
			bi--;
		}
	}
	memcpy(local_A, temp_C, local_n * sizeof(int));
	CALI_MARK_END("comp_small");
	CALI_MARK_END("comp");
} 

void Print_list(int local_A[], int local_n, int rank)
{
	int i;
	printf("%d: ", rank);
	for (i = 0; i < local_n; i++)
		printf("%d ", local_A[i]);
	printf("\n");
} 

void Print_local_lists(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm)
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
}

void Check_sorted(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm)
{
	int global_n = local_n * p;
	int *global_A = NULL;
	int is_sorted = 1;

	if (my_rank == 0)
	{
		global_A = (int *)malloc(global_n * sizeof(int));
	}

	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_large");
	CALI_MARK_BEGIN("MPI_Gather");
	MPI_Gather(local_A, local_n, MPI_INT, global_A, local_n, MPI_INT, 0, comm);
	CALI_MARK_END("MPI_Gather");
	CALI_MARK_END("comm_large");
	CALI_MARK_END("comm");

	if (my_rank == 0)
	{
		CALI_MARK_BEGIN("correctness_check");
		for (int i = 0; i < global_n - 1; i++)
		{
			if (global_A[i] > global_A[i + 1])
			{
				printf("Error: List is not sorted at index %d\n", i);
				is_sorted = 0; // Mark as not sorted
				break;
			}
		}

		if (is_sorted)
		{
			printf("List is correctly sorted.\n");
		}
		CALI_MARK_END("correctness_check");

		free(global_A);
	}
}

int main(int argc, char *argv[])
{
	CALI_CXX_MARK_FUNCTION;
	int my_rank, p;
	int *local_A;	
	int global_n;	
	int local_n;	
	MPI_Comm comm;
	inputList inputType;

	MPI_Init(&argc, &argv);
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &my_rank);

	if (my_rank == 0)
	{
		printf("Input Arguments:\n");
		printf("processes:%d\n", p);
		for (int i = 0; i < argc; i++)
		{
			printf("argv[%d] = %s\n", i, argv[i]);
		}
	}

	CALI_MARK_BEGIN("main");
	Get_args(argc, argv, &global_n, &local_n, &inputType, my_rank, p, comm);
	local_A = (int *)malloc(local_n * sizeof(int));
	CALI_MARK_BEGIN("data_init");
	Generate_list(local_A, local_n, my_rank, inputType); // generate random list
	CALI_MARK_END("data_init");
	free(local_A); 
	CALI_MARK_END("main");

	cali::ConfigManager mgr;
	mgr.start();

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
	adiak::value("InputType", inputListToString(inputType));
	adiak::value("num_procs", p);
	adiak::value("group_num", "group");
	adiak::value("implementation_source", "online");

	mgr.stop();
	mgr.flush();
	MPI_Finalize();

	return 0;
} 