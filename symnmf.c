#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"
#define BETA 0.5


int max_iter=300;
double epsilon=0.0001;
int d,N,k;
double** W;
double** A;
double* D;
double **data;

void print_matrix(double **matrix,int rows,int cols);

/*
* we represented a diagonal matrix as an array of its a[i][i] entries
* this function returns a full representation of the matrix, by padding it with 0's
*
* arr- an array of doubles representing a diagonal matrix
* len- representing the dimension of the matrix
*/
double** Diagonal_D(double* arr,int len){
    int i;
    double** res;
    double* p = calloc(len*len, sizeof(double));
    if (p == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    res = calloc(len,sizeof(double *));
    if (res == NULL) {
        printf("An Error Has Occurred");
        free(p);
        exit(1);
    }
    for(i=0; i<len; i++)
        res[i] = p+i*len;
    for(i=0; i<len; i++)
        res[i][i] = arr[i];
    return res;
}

/*return the squared euclidean distance of two vector given by the formula sqrt(Σ (v1 - v2)^2)
* vector1 - first vector to be calculated
* vector2 - second vector to calculated
* len- the dimension of the vector
*/
double squared_euclidean_distance(double* vector1,double* vector2,int len) {
    int i;
    double res = 0;
    for (i = 0; i < len; i++){
        res+= pow(vector1[i]-vector2[i],2);
    }
    return res;
}

/*Form the similarity matrix A from the given set of n points X by calculating the distance between each entry in X
* X - a matrix of vectors, calculating the nC2 distances between the vectors
*
*/
double** sym_c(double** X){
    int i,j;
    double* p;
    p = calloc(N*N, sizeof(double));
    /* handling errors */
    if (p == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    A = calloc(N,sizeof(double *));
    if (A == NULL) {
        printf("An Error Has Occurred");
        free(p);
        exit(1);
    }
    for(i=0; i<N; i++)
        A[i] = p+i*N;
    /* main algorithm */
    for(i=0;i<N;i++){
        for(j=0;j<i;j++){
            double res = squared_euclidean_distance(X[i],X[j],d);
            res = exp(-res/2);
            A[i][j] = res;
            A[j][i] = res;
        }
    }
    return A;
}

/*Compute the Diagonal Degree Matrix D with the similarity matrix A
* computing the sum of each row in the sym matrix by the formula Σa[i][j] = d[i]
* 
* X - the matrix to be 
* returns - a diagonal matrix of n*1 dimensions expanded with Diag_matrix method
*/
double* ddg_c(double** X)
{
    int i,j;
    D = calloc(N,sizeof(double *));
    if (D == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    A=sym_c(X);
    for(i=0;i<N;i++)
        for(j=0;j<N;j++)
            D[i]+=A[i][j];
    
    /*free the allocate space for A*/
    if(!A[0])
        free(A[0]);
    if(!A)
        free(A);
    
    return D;        
}

/*Compute the normalized similarity W with the Diagonal Degree Matrix D using two matrix multlipications
* X - the diagonal matrix that extracts the diagonal and the A matrix
*
*returns - a similarity matrix W
*/
double** norm_c(double** X){
    int i,j;
    double* p;
    p = calloc(N*N, sizeof(double));
    if (p == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    W = calloc(N,sizeof(double *));
    if (W == NULL) {
        printf("An Error Has Occurred");
        free(p);
        exit(1);
    }
    /*main function*/
    D=ddg_c(X); /*getting the diagonal matrix */
    for(i=0; i<N; i++)
        W[i] = p+i*N;
    for(i=0;i<N;i++)
        for(j=0;j<N;j++)
            W[i][j] = A[i][j]*(1/sqrt(D[i]))*(1/sqrt(D[j]));
    if(!D)
        free(D);
    return W;
}

/*calculate the squared Frobenius_norm between two matrix using the formula  ΣΣ(a-b)^2
*
* arr1 - a matrix to be first in the bilinear form
* arr2 - a matrix to be second in the bilinear form
*
* 
* returns - a non negative number denoting the squared frobenius norm of the two matrices

*/
double Frobenius_norm(double** arr1,double** arr2){
    int i,j;
    double res=0;
    /* main loop using  ΣΣ(a-b)^2 formula */
    for(i=0;i<N;i++){
        for(j=0;j<k;j++){
            res+= pow(arr1[i][j]-arr2[i][j],2);
        }
    }
    return res;
}

/*Function to free the allocated memory for a matrix
* matrix - a 2D array of doubles to be freed
* rows - the amount of arrays (rows) to be freed
*/
void freeMatrix(double** matrix) {
    if(matrix){
        return;
    }
    if(!matrix[0]){
        free(matrix[0]);
    }
    if(matrix){
        free(matrix);
    }
}

/*multiply two matrices by using the given two matrices and their sizes A*B =  ΣΣ(a*b) 
* firstMatrix - matrix A of dimension l * m
* secondMatrix - matrix B of dimension m * n
* rows_matrix1 - the size l
* cols_matrix_1 - the size m
* cols_matrix2 - the size n
*returns - a l * n matrix of doubles
*/
double** multiplyMatrices(double** firstMatrix, double** secondMatrix, int rows_matrix1, int cols_matrix1, int cols_matrix2) {
    int i, j, m;
    double** resultMatrix;
    double* p = calloc(rows_matrix1 * cols_matrix2, sizeof(double));
    /* error handling */
    if (p == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    resultMatrix = calloc(rows_matrix1, sizeof(double *));
    if (resultMatrix == NULL) {
        printf("An Error Has Occurred");
        free(p);
        exit(1);
    }
    for (i = 0; i < rows_matrix1; i++)
        resultMatrix[i] = p + i * cols_matrix2;

    /*O(n^3) standart algorithm*/
    for (i = 0; i < rows_matrix1; i++) {
        for (j = 0; j < cols_matrix2; j++) {
            for (m = 0; m < cols_matrix1; m++) {
                resultMatrix[i][j] += firstMatrix[i][m] * secondMatrix[m][j];
            }
        }
    }

    return resultMatrix;
}

/* assigns the transpose matrix of matrix to a given matrix "transposedMatrix" of size l*m
* rows - size l
* cols - size m
* transposedMatrix - the matrix assigned values to
*/
void transposeMatrix(int rows, int cols, double** matrix, double*** transposedMatrix) {
    int i, j;
    /* aij = aji */
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            (*transposedMatrix)[j][i] = matrix[i][j];
        }
    }
}

/* a step in the iterative algorithm to compute H - the affiliation matrix, swaps between resH and oldH
* 
* resH - the matrix to be assigned to
* oldH - the "old" matrix.
*/
void calc_new_H(double ***resH, double **oldH) {
    int i, j;
    double **HT;
    double **HHT;
    double **HHTH;
    double **WH;
    double *p, *p_trans;

    freeMatrix(*resH);
    /* handling errors */
    p = calloc(N * k, sizeof(double));
    if (p == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }
    *resH = calloc(N, sizeof(double *));
    if (*resH == NULL) {
        printf("An Error Has Occurred");
        free(p);
        exit(1);
    }
    /* allocating contiguous memory for resH*/
    for (i = 0; i < N; i++) {
        (*resH)[i] = p + i * k;
    }

    /* more error handling */
    p_trans = calloc(N * k, sizeof(double));
    if (p_trans == NULL) {
        printf("An Error Has Occurred");
        free(p);
        freeMatrix(*resH);
        exit(1);
    }
    HT = calloc(k, sizeof(double *));
    if (HT == NULL) {
        printf("An Error Has Occurred");
        free(p);
        freeMatrix(*resH);
        free(p_trans);
        exit(1);
    }
    for (i = 0; i < k; i++) {
        HT[i] = p_trans + i * N;
    }

    /* computing H*H^t*H (denominator) and W * H (numerator)*/
    WH = multiplyMatrices(W, oldH, N, N, k);
    transposeMatrix(N, k, oldH, &HT);
    HHT = multiplyMatrices(oldH, HT, N, k, N);
    HHTH = multiplyMatrices(HHT, oldH, N, N, k);

    for (i = 0; i < N; i++) {
        for (j = 0; j < k; j++) { /* Correct the loop to increment j instead of k*/
            if (HHTH[i][j] == 0) {
                freeMatrix(*resH);
                freeMatrix(WH);
                freeMatrix(HT);
                freeMatrix(HHT);
                freeMatrix(HHTH);
                printf("An Error Has Occurred");
                exit(1);
            }
            /* resH[i][j] = 1 - β + β( W * H )/(H*H^t*H) */
            (*resH)[i][j] = oldH[i][j] * (1 - BETA + (BETA * (WH[i][j] / HHTH[i][j])));
        }
    }

    freeMatrix(*resH);
    freeMatrix(WH);
    freeMatrix(HT);
    freeMatrix(HHT);
    freeMatrix(HHTH);
}

/*
* A <- B where A and B are matrices of same size j* l
*
* copyTo - the matrix to be overrided 
* copyFrom - the overrides matrix 
* rows - size j
* cols - size l
* returns - 0 if succesful 
*/
int copy_matrix(double **copyTo, double **copyFrom, int rows, int cols){
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            /* a[i][j] <- B[i][j] */
            copyTo[i][j] = copyFrom[i][j];
        }
    }
    return 0;
}

/*Perform full the symNMF and compute the Symmetric Nonnegative Matrix Factorization with iterative algorithm
* prev_H - the raw H matrix generated uniformally with python
* returns - the final affinity matrix H, each row's argmin() is the respective point's cluster affiliation
*/
double** symnmf_c(double **prev_H){
    int i;
    double **curr_H;
    
    double *p = calloc(N*k, sizeof(double));
    /* error handling */
    if (p == NULL) {
        printf("An Error Has Occurred");
        exit(1);
    }

    curr_H = calloc(N,sizeof(double *));
    if (curr_H == NULL) {
        printf("An Error Has Occurred");
        free(p);
        exit(1);
    }

    /* allocating contiguous matrix memory*/
    for(i=0; i<N; i++){
        curr_H[i] = p+i*k;
    }
    
    /* each iteration a new H matrix is generated via the formula resH[i][j] = 1 - β + β( W * H )/(H*H^t*H) */
    for(i=0; i<max_iter; i++){
        calc_new_H(&curr_H,prev_H);
        /* if arrived to convergance frobenius norm-wise - halt before max_iter */
        if(Frobenius_norm(prev_H,curr_H)<epsilon){
            break;
        }
        copy_matrix(prev_H,curr_H,N,k);
    }
    return curr_H;
}

/* calculates how many rows a file have, used for extracting N when being called from C's main
* file - the source file to count his rows number
*
* returns - the amount of rows in a given file
*/
int cols_in_file(FILE *file){
    int cnt=1;
    int c;
    while((c = fgetc(file))!=10 && c!=EOF){
        if(c==44) { /* if \n -> add to count */
            cnt++;
        }
    }
    rewind(file);
    return cnt;
}

int rows_in_file(FILE *file){
    int cnt = 0;
    char *line = NULL;
    size_t len = 0;

    while (getline(&line, &len, file) != -1) {
        cnt++;
    }
    free(line);
    rewind(file);
    return cnt;
}

void print_matrix(double **matrix, int rows , int cols) {
    int i,j;
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            if(j==cols-1){
                printf("%.4f", matrix[i][j]);
            }
            else
                printf("%.4f,", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]){
    if(argc ==3){
        int i,j;
        double *p;
        char *line = NULL;
        size_t len = 0;
        FILE *file = fopen(argv[2],"r");
        double **sym_mat, **ddg_mat, **norm_mat;
        if (file == NULL) {
            printf("An Error Has Occurred");
            return 1;
        }
        d=cols_in_file(file);
        N = rows_in_file(file);
        p = calloc(N*d, sizeof(double));
        if (p == NULL) {
            printf("An Error Has Occurred");
            return 1;
        }
        data = calloc(N,sizeof(double *));
        if (data == NULL) {
            printf("An Error Has Occurred");
            free(p);
            return 1;
        }
        for(i=0; i<N; i++)
            data[i] = p+i*d;
        for(i=0;i<N;i++){
            if (getline(&line, &len, file) != -1) {
                char *start = line;
                char *end;
                for(j=0;j<d;j++){
                    data[i][j] = strtod(start, &end);
                    if (start == end) {
                        printf( "An Error Has Occurred");
                        if (line)
                            free(line);
                        return 0;
                    }
                    start = end;
                    while (*start == ',') {
                        start++;
                    }
                }
                if(!start){
                    free(start);
                }
                if(!end){
                    free(end);
                }
            }
        }
        fclose(file); 
        if(strcmp(argv[1],"sym")==0){
            sym_mat = sym_c(data);
            print_matrix(sym_mat,N,N);
            if(sym_mat[0]!=NULL){
                free(sym_mat[0]);
            }
            if(sym_mat!=NULL){
                free(sym_mat);
            }
        }
        else if(strcmp(argv[1],"ddg")==0){
            ddg_mat = Diagonal_D(ddg_c(data),N);
            print_matrix(ddg_mat,N,N);
            if(ddg_mat[0]!=NULL){
                free(ddg_mat[0]);
            }
            if(ddg_mat!=NULL){
                free(ddg_mat);
            }
            if(A[0]!=NULL){
                free(A[0]);
            }
            if(A!=NULL){
                free(A);
            }
            if(D!=NULL){
                free(D);
            }
        }
        else if(strcmp(argv[1],"norm")==0){
            norm_mat = norm_c(data);
            print_matrix(norm_mat,N,N);
            if(norm_mat[0]!=NULL){
                free(norm_mat[0]);
            }
            if(norm_mat!=NULL){
                free(norm_mat);
            }
            if(A[0]!=NULL){
                free(A[0]);
            }
            if(A!=NULL){
                free(A);
            }
            if(D!=NULL){
                free(D);
            }
        }
        if(line!=NULL){
            free(line);
        }
        if(data[0]!=NULL){
            free(data[0]);
        }
        if(data !=NULL){
            free(data);
        }

        return 1;
    }
    return 0;
}