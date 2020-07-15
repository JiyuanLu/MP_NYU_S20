#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

// Global
int num;
int **weights;
int min_cost = 2147483647;
int *min_path;

void initialize_cost(int *cost){
    if (num == 1)
	*cost = 0;
    else if(num == 2)
	*cost = weights[0][1];
    else
    	*cost = 2147483647;
}

int *initialize_min_path(){
    int *path = (int *)malloc(num * sizeof(int));
    memset(path, 0, num * sizeof(int));
    return path;
}

int *initialize_path(){
    int *path = (int *)malloc(num * sizeof(int));
    int i;
    for (i = 0; i < num; ++i)
	path[i] = i;
    return path;
}

int compute_cost(int *path){
    int cost = 0;
    int i;
    for (i = 0; i < num - 1; ++i)
	cost += weights[path[i]][path[i+1]];
    return cost;
}

void swap(int *x, int *y){
    int temp = *x;
    *x = *y;
    *y = temp;
}

void permute(int *path, int l, int r, int *min_cost, int *min_path){
    int i;
    if (l > r)
	return;
    else if (l == r){
	int cost = compute_cost(path);
	if (cost < *min_cost){
	    *min_cost = cost;
	    memcpy(min_path, path, num * sizeof(int));
	}
    }
    else{
	for (i = l; i <= r; ++i){
	    swap(path+l, path+i);
	    permute(path, l+1, r, min_cost, min_path);
	    swap(path+l, path+i);
	}
    }
}

void update_global(int local_min_cost, int *local_min_path){
    #pragma omp critical
    {
        if (local_min_cost < min_cost){
	    min_cost = local_min_cost;
	    memcpy(min_path, local_min_path, num * sizeof(int));
	}
    }
}

int main(int argc, char *argv[]){
    /****** Start of Sequential Processing ******/
    clock_t program_start = clock();
    double program_start_time = omp_get_wtime();
    /*** Preprocessing ***/
    // Check command line arguments
    
    if (argc != 4){
	printf("usage: ptsm x t filename.txt\n");
	printf("x is the number of cities\n");
	printf("t is the number of threads\n");
	printf("filename.txt is the file that contains the distance matrix\n");
	return 1;
    }
    // Read in weights
    num = atoi(argv[1]);
    int num_of_threads = atoi(argv[2]);
    char *file = argv[3];
    FILE *fp = fopen(file, "r");

    if (!fp){
	printf("File cannot be opened!\n");
	return 2;
    }

    weights = (int**) malloc(num * sizeof(int*));
    if (!weights){
	printf("Cannot allocate weights matrix!\n");
	return 3;
    }

    int i, j;
    for (i = 0; i < num; ++i){
	weights[i] = (int*) malloc(num * sizeof(int));
	if (!weights[i]){
	    printf("Cannot allocate weights matrix on %dth row!\n", i+1);
	    return 4;
	}
    }

    for (i = 0; i < num; ++i){
	for (j = 0; j < num; ++j){
	    fscanf(fp, "%d ", &weights[i][j]);
	}
    }

    fclose(fp);

    /*** Find Min Path ***/
    // Initialize global variables
    initialize_cost(&min_cost);
    min_path = initialize_path();
     
    // Create private variables
    int local_min_cost;
    int *local_min_path;
    int *local_path;   
    clock_t parallel_start;
    clock_t parallel_end;
    double parallel_start_time;
    double parallel_end_time;

    /** Distribute the work among threads by setting the second city to visit **/
    #pragma omp parallel private(i, j, local_min_cost, local_min_path, local_path) num_threads(num_of_threads)
    {
    // Initialize private variables
    initialize_cost(&local_min_cost);
    local_min_path = initialize_min_path();
    local_path = initialize_path();

    /****** End of Sequential Processing ******/    
    /****** Start of Parallel Processing ******/
    #pragma omp single nowait
    {
    parallel_start = clock();
    parallel_start_time = omp_get_wtime();
    }

    #pragma omp for
    for (i = 1; i < num; ++i){
	local_path[1] = i;
	for (j = 1; j < i; ++j)
	    local_path[1+j] = j;
        for (j = i+1; j < num; ++j)
	    local_path[j] = j;
	permute(local_path, 2, num-1, &local_min_cost, local_min_path);
    }
    /****** End of Parallel Processing ******/
    #pragma omp single nowait
    {
    parallel_end = clock();
    parallel_end_time = omp_get_wtime();
    }

    /****** Start of Sequential Processing ******/

    update_global(local_min_cost, local_min_path);
    free(local_min_path);
    free(local_path);
    }

    /*** Output results ***/
    printf("Best path: ");
    for (i = 0; i < num; ++i)
	printf("%d ", min_path[i]);
    printf("\n");
    printf("Distance: %d\n", min_cost);
    free(min_path);
    /****** End of Sequential Processing ******/
    // Output total processing time VS sequential processing time VS parallel processing time
    clock_t program_end = clock();
    double program_end_time = omp_get_wtime();
    clock_t program_ticks = program_end - program_start;
    clock_t parallel_ticks = parallel_end - parallel_start;
    clock_t sequential_ticks = program_ticks - parallel_ticks;
    printf("It took %ld ticks (%f seconds) to execute the program.\n", program_ticks, ((float)program_ticks)/CLOCKS_PER_SEC);
    printf("It took %ld ticks (%f seconds) to execute the sequential part of the program.\n", sequential_ticks, ((float)sequential_ticks)/CLOCKS_PER_SEC);
    printf("It took %ld ticks (%f seconds) to execute the parallel part of the program.\n", parallel_ticks, ((float)parallel_ticks)/CLOCKS_PER_SEC);

    double program_time = program_end_time - program_start_time;
    double parallel_time = parallel_end_time - parallel_start_time;
    double sequential_time = program_time - parallel_time;
    printf("It took %f seconds to execute the program.\n", program_time);
    printf("It took %f seconds to execute the sequential part.\n", sequential_time);
    printf("It took %f seconds to execute the parallel part.\n", parallel_time);
    return 0;
}













