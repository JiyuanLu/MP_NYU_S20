// Pure mpi
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

int check_converge(float *old_x, float *x, int num, float err){
    int i;
    float absolute_relative_error;

    for(i = 0; i < num; ++i){
        absolute_relative_error = fabs((x[i] - old_x[i])/x[i]);
	
        if(absolute_relative_error > err){
            return 0;
        }
    }

    return 1;
}

int main(int argc, char *argv[]){ 
    double program_start_time = MPI_Wtime();
    int i, j, k;
    int nit = 0; // number of iterations
    char output[100] = "";

    if(argc != 2){
	printf("Usage: ./solveopenmp filename.txt\n");
	exit(1);
    }
    
    /* Master process gets input */
    int num = 0; // Number of unknowns
    float err; // The absolute relative error
    float *x; // The unknowns
    float *a; // The coefficients
    float *b; // The constants
    FILE *fp;

    
    fp = fopen(argv[1], "r");
    if(!fp){
        printf("Cannot open file %s\n", argv[1]);
        exit(1);
    }

    fscanf(fp, "%d ", &num);
    fscanf(fp, "%f ", &err);
    
    x = (float *)malloc(num * sizeof(float));
    if(!x){
        printf("Cannot allocate space for x!\n");
        exit(1);
    }
    
    a = (float *)malloc(num * num * sizeof(float));
    if(!a){
        printf("Cannot allocate space for a!\n");
        exit(1);
    }

    b = (float *)malloc(num * sizeof(float));
    if(!b){
        printf("Cannot allocate space for b!\n");
        exit(1);
    }

    for(i = 0; i < num; ++i){
        fscanf(fp, "%f ", &x[i]);
    }

    for(i = 0; i < num; ++i){
        for(j = 0; j < num; ++j){
            fscanf(fp, "%f ", &a[i * num + j]);
        }
        fscanf(fp, "%f ", &b[i]);
    }

    fclose(fp);
    
    /* End master process gets input */

    
    /* Initialize MPI */
    int comm_sz, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    /* Master process broadcast the initial values for x, a, b to each process */
    MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&err, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(a, num * num, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    /* Distribute the calculation of xi among processes */
    int send_counts[comm_sz];
    int displs[comm_sz];
    int min_work = num / comm_sz;
    int remainder = num % comm_sz;
    int displacement = 0;
    
    for(i = 0; i < comm_sz; ++i){
        displs[i] = displacement;
        if(i < remainder)
            send_counts[i] = min_work + 1;
        else
            send_counts[i] = min_work;
        
        displacement += send_counts[i];
    }
        
    /* Define local buffers */
    float *old_x; 
    float *local_x;
    
    old_x = (float *)malloc(num * sizeof(float));
    if(!old_x){
        printf("Cannot allocate space for old_x!\n");
        exit(1);
    }
    
    local_x = (float *)malloc((min_work + 1) * sizeof(float));
    if(!local_x){
        printf("Cannot allocate space for local_x!\n");
        exit(1);
    }    
    
    double parallel_start_time = MPI_Wtime();
    do{
        ++nit;
        
        // store old_x for comparsion later
        for(i = 0; i < num; ++i)
            old_x[i] = x[i];
        
        /* Each process compute its responsible portion of xis */
        int start_index = 0;
        for(i = 0; i < my_rank; ++i)
            start_index += send_counts[i];
        
        for(k = 0; k < send_counts[my_rank]; ++k){
            i = k + start_index;
            local_x[k] = b[i];
            //printf("local_x[%d] = %f in process %d\n", k, local_x[k], my_rank);
            
            for(j = 0; j < num; ++j){
                if (j != i){
                    local_x[k] -= a[i * num + j] * old_x[j];
                    //printf("local_x[%d] = %f in process %d\n", k, local_x[k], my_rank);
                }
            }
            
            local_x[k] /= a[i * num + i];
            //printf("local_x[%d] = %f in process %d\n", k, local_x[k], my_rank);
        }
        
        /* Print debug information 
        for(i = 0; i < comm_sz; ++i)
            printf("send_counts[%d] in process %d = %d\n", i, my_rank, send_counts[i]);
            
        for(i = 0; i < num; ++i)
            printf("old_x[%d] in process %d = %f\n", i, my_rank, old_x[i]);
        
        for(i = 0; i < send_counts[my_rank]; ++i)
            printf("local_x[%d] in process %d = %f\n", i, my_rank, local_x[i]);
        */
    
        /* Gather each part of computed xis into x. */
        //MPI_Allgather(local_x, send_counts[my_rank], MPI_FLOAT, x, num, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(local_x, send_counts[my_rank], MPI_FLOAT, x, send_counts, displs, MPI_FLOAT, MPI_COMM_WORLD);
    }while(!check_converge(old_x, x, num, err));
    
    double parallel_end_time = MPI_Wtime();

    /* Master process writes results to file */
    if(my_rank == 0){
        sprintf(output, "%d.sol", num);
        fp = fopen(output, "w");
        if(!fp){
            printf("Cannot create output file %s!\n", output);
            exit(1);
        }

        for(i = 0; i < num; ++i)
            fprintf(fp, "%f\n", x[i]);

        printf("total number of iterations: %d\n", nit);

        fclose(fp);
    }
        
    /* Clean up variables */
    free(x);
    free(a);
    free(b);
    
    free(old_x);
    free(local_x);
    
    //printf("Process %d has cleaned up variables and is ready to terminate\n", my_rank);
    double program_end_time = MPI_Wtime();
    double program_time = program_end_time - program_start_time;
    double parallel_time = parallel_end_time - parallel_start_time;
    double max_program_time;
    double max_parallel_time;
    MPI_Reduce(&program_time, &max_program_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&parallel_time, &max_parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double sequential_time = max_program_time - max_parallel_time;
    if(my_rank == 0){
        printf("It took %f seconds to execute the program.\n", max_program_time);
        printf("It took %f seconds to execute the sequential part.\n", sequential_time);
        printf("It took %f seconds to execute the parallel part.\n", max_parallel_time);
    }
    
    MPI_Finalize();
    return 0;
}
