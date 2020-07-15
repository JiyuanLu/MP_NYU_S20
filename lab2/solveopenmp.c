// Pure openmp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>

/*** Globals ***/
int num = 0; // Number of unknowns
float err; // The absolute relative error
float *x; // The unknowns
float **a; // The coefficients
float *b; // The constants

void get_input(char filename[]){
    FILE *fp;
    int i, j;

    fp = fopen(filename, "r");
    if(!fp){
	printf("Cannot open file %s\n", filename);
	exit(1);
    }

    fscanf(fp, "%d ", &num);
    fscanf(fp, "%f ", &err);
    
    x = (float *)malloc(num * sizeof(float));
    if(!x){
	printf("Cannot allocate space for x!\n");
	exit(1);
    }

    a = (float **)malloc(num * sizeof(float *));
    if(!a){
	printf("Cannot allocate space for a!\n");
	exit(1);
    }
    
    for(i = 0; i < num; ++i){
	a[i] = (float *)malloc(num * sizeof(float));
	if(!a[i]){
	    printf("Cannot allocate space for a[%d]!\n", i);
	    exit(1);
    	}
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
	    fscanf(fp, "%f ", &a[i][j]);
	}
	fscanf(fp, "%f ", &b[i]);
    }

    fclose(fp);
}

void check_matrix(){
    int bigger = 0; // set to 1 if at least one diagonal element > sum
    int i, j;
    float aii; // value at the diagonal
    float sum; // sum of all elements besides aii
    
    for(i = 0; i < num; ++i){
	sum = 0;
	aii = fabs(a[i][i]);

	for(j = 0; j < num; ++j){
	    if(j != i)
		sum += fabs(a[i][j]);
	}

        if(aii < sum){
	    printf("The matrix will not converge!\n");
	    exit(1);
	}

	if(aii > sum)
	    ++bigger;
    }

    if(!bigger){
	printf("The matrix will not converge!\n");
	exit(1);
    }
}	

int check_converge(float *old_x){
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

void update_x(float *old_x){
    int k;
    int i, j;

    for(k = 0; k < num; ++k){
	old_x[k] = x[k];
    } 

    #pragma omp parallel for 
    for(i = 0; i < num; ++i){
	x[i] = b[i];
	for(j = 0; j < num; ++j){
	    if(j != i) x[i] -= a[i][j] * old_x[j];
	}
	x[i] /= a[i][i];
    }
}

int main(int argc, char *argv[]){
    double program_start_time = omp_get_wtime();
    int nit = 0; // number of iterations
    int i;
    FILE *fp;
    char output[100] = "";

    if(argc != 2){
	printf("Usage: ./solveopenmp filename.txt\n");
	exit(1);
    }
    
    /* Read the input file and fill the global data structures */
    get_input(argv[1]);

    /* Check for convergence condition */    
    check_matrix();

    /* Compute x */
    float *old_x = (float *)malloc(num * sizeof(float));
    if(!old_x){
	printf("Cannot allcate space for old_x!\n");
	exit(1);
    }

    double parallel_start_time = omp_get_wtime();
    do{
	int k;
	int i, j;
	
	for(k = 0; k < num; ++k){
	    old_x[k] = x[k];
	}
	
	#pragma omp parallel for private(i, j) shared(x, a, b, num, old_x)
	for(i = 0; i < num; ++i){
	    x[i] = b[i];
	    for(j = 0; j < num; ++j){
		if(j != i) x[i] -= a[i][j] * old_x[j];
	    }
	    x[i] /= a[i][i];
        }

	++nit;
    }while(check_converge(old_x) != 1);
    double parallel_end_time = omp_get_wtime();

    free(old_x);

    /* Writing results to file */
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

    /* Clean up variable space */
    free(x);
    for(i = 0; i < num; ++i){
	free(a[i]);
    }
    free(a);
    free(b);

    /* Print out threads count */
    #pragma omp parallel
    #pragma omp single
    printf("Number of threads:%d\n", omp_get_num_threads());
    
    double program_end_time = omp_get_wtime();
    double program_time = program_end_time - program_start_time;
    double parallel_time = parallel_end_time - parallel_start_time;
    double sequential_time = program_time - parallel_time;
    printf("It took %f seconds to execute the program.\n", program_time);
    printf("It took %f seconds to execute the sequential part.\n", sequential_time);
    printf("It took %f seconds to execute the parallel part.\n", parallel_time);
    
    return 0;
}
