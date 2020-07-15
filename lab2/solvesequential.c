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
    int i, j;

    for(i = 0; i < num; ++i){
	old_x[i] = x[i];
    }

    for(i = 0; i < num; ++i){
	float xi = b[i];
	for(j = 0; j < num; ++j){
	    if(j != i) xi -= a[i][j] * old_x[j];
	}
	xi /= a[i][i];
	x[i] = xi;
    }
}

int main(int argc, char *argv[]){
    int i;
    int nit = 0; // number of iterations
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

    do{
	update_x(old_x);
	++nit;
    }while(check_converge(old_x) != 1);

    free(old_x);

    /* Writing results to file */
    sprintf(output, "%d_sequential.sol", num);
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
    //#pragma omp parallel
    //printf("Number of threads:%d\n", omp_get_num_threads());

    return 0;
}
