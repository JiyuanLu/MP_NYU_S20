#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int num;
int **weights;
int min_cost = 2147483647;
int *min_path;


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

void permute(int *path, int l, int r){
    int i;
    if (l > r)
	return;
    else if (l == r){
	int cost = compute_cost(path);
	if (cost < min_cost){
	    min_cost = cost;
	    memcpy(min_path, path, num * sizeof(int));
	}
    }
    else{
	for (i = l; i <= r; ++i){
	    swap(path+l, path+i);
	    permute(path, l+1, r);
	    swap(path+l, path+i);
	}
    }
}


int main(int argc, char *argv[]){
    // Check command line arguments
    if (argc != 3){
	printf("usage: stsm num file\n");
	printf("num is the number of cities\n");
	printf("file is the number of the file containing the weights\n");
	return 1;
    }
    // Read in weights
    num = atoi(argv[1]);
    char *file = argv[2];
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

    // Enumerate every path
    min_path = malloc(num * sizeof(int));
    memset(min_path, 0, num * sizeof(int));
    int path[num];
    for (i = 0; i < num; ++i)
	path[i] = i; 
    permute(path, 1, num-1);
    printf("Shortest path is:\n");
    for (i = 0; i < num; ++i)
	printf("%d ", min_path[i]);
    printf("\n");
    printf("total weight: %d\n", min_cost);
 
    return 0;
}
