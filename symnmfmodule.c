#include "symnmf.h"
#include <stdio.h>
#include <stdlib.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>


int d,N,k;
double** W;
double** H;
double** A;
double* D;


/*function that convert the PyObject array into array in c*/
static double** convertPyFloatArrayToDoubleArray(PyObject* pyArray, int numRows, int numCols) {
    double** doubleArray = malloc(numRows * sizeof(double*));
    if (doubleArray == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
        return NULL;
    }

    for (int i = 0; i < numRows; ++i) {
        doubleArray[i] = malloc(numCols * sizeof(double));
        if (doubleArray[i] == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
            // Free memory allocated so far
            for (int j = 0; j < i; ++j) {
                free(doubleArray[j]);
            }
            free(doubleArray);
            return NULL;
        }

        PyObject* row = PyList_GetItem(pyArray, i);
        if (!PyList_Check(row) || PyList_Size(row) != numCols) {
            PyErr_SetString(PyExc_TypeError, "Input is not a 2D list of appropriate size");
            // Free memory allocated so far
            for (int j = 0; j <= i; ++j) {
                free(doubleArray[j]);
            }
            free(doubleArray);
            return NULL;
        }

        for (int j = 0; j < numCols; ++j) {
            int k;
            PyObject* item = PyList_GetItem(row, j);
            if (!PyFloat_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "Element is not a float");
                // Free memory allocated so far
                for (k = 0; k <= i; ++k) {
                    free(doubleArray[k]);
                }
                free(doubleArray);
                return NULL;
            }
            doubleArray[i][j] = PyFloat_AsDouble(item);
        }
    }

    return doubleArray;
}

/*convert the array in c back into PyObject array*/
static PyObject* convertDoubleArrayToPyFloatArray(double** doubleArray, int numRows, int numCols) {
    PyObject* pyArray = PyList_New(numRows);
    if (!pyArray) {
        return NULL;
    }

    for (int i = 0; i < numRows; ++i) {
        PyObject* row = PyList_New(numCols);
        if (!row) {
            Py_DECREF(pyArray);
            return NULL;
        }

        for (int j = 0; j < numCols; ++j) {
            PyObject* value = PyFloat_FromDouble(doubleArray[i][j]);
            if (!value) {
                Py_DECREF(pyArray);
                Py_DECREF(row);
                return NULL;
            }
            PyList_SET_ITEM(row, j, value);
        }

        PyList_SET_ITEM(pyArray, i, row);
    }
    return pyArray;
}

/**************************************/
/*  The python API for sym function   */
/**************************************/
static PyObject* sym(PyObject *self, PyObject *args){
    int i;
    PyObject* X_array;
    if(!PyArg_ParseTuple(args, "O",&X_array)) {
        return NULL;
    }
    N = PyList_Size(X_array);
    d=  PyList_Size(PyList_GetItem(X_array,0));
    double** X = convertPyFloatArrayToDoubleArray(X_array,N,d);
    double** A = sym_c(X);
    PyObject* py_result = convertDoubleArrayToPyFloatArray(A,N,N);
    
    for (i = 0; i<N;i++) {
        if(!X[i])
            free(X[i]);
    }
    if(!X)
        free(X);

    if(!A[0])
        free(A[0]);
    if(!A)
        free(A);
    return py_result;
}

/**************************************/
/*  The python API for ddg function   */
/**************************************/
static PyObject* ddg(PyObject *self, PyObject *args){
    int i;
    PyObject* X_array;
    if(!PyArg_ParseTuple(args, "O",&X_array)) {
        return NULL;
    }
    N = PyList_Size(X_array);
    d=  PyList_Size(PyList_GetItem(X_array,0));
    double** X = convertPyFloatArrayToDoubleArray(X_array,N,d);
    D = ddg_c(X);
    double** diag_D = Diagonal_D(D,N);
    PyObject* py_result = convertDoubleArrayToPyFloatArray(diag_D,N,N);
    
    for (i = 0; i<N;i++) {
        if(!X[i])
            free(X[i]);
    }
    if(!X)
        free(X);
    if(!D)
        free(D);
    if(!diag_D[0])
        free(diag_D[0]);
    if(!diag_D)
        free(diag_D);
    return py_result;
}

/***************************************/
/*  The python API for norm function   */
/***************************************/
static PyObject* norm(PyObject *self, PyObject *args){
    int i;
    PyObject* X_array;
    if(!PyArg_ParseTuple(args,"O",&X_array)) {
        return NULL;
    }
    N = PyList_Size(X_array);
    d=  PyList_Size(PyList_GetItem(X_array,0));
    double** X = convertPyFloatArrayToDoubleArray(X_array,N,d);
    W = norm_c(X);
    PyObject* py_result = convertDoubleArrayToPyFloatArray(W,N,N);
    
    for (i = 0; i<N;i++) {
        if(!X[i])
            free(X[i]);
    }
    if(!X)
        free(X);

    if(!W[0])
        free(W[0]);
    if(!W)
        free(W);
    return py_result;
}

/***************************************/
/*  The python API for symnmf function */
/***************************************/
static PyObject* symnmf(PyObject *self, PyObject *args){
    int i;
    PyObject *H_array,*W_array;
    double **res;
    if(!PyArg_ParseTuple(args, "iOO",&k,&W_array,&H_array)) {
        return NULL;
    }
    W = convertPyFloatArrayToDoubleArray(W_array,N,N);
    H = convertPyFloatArrayToDoubleArray(H_array,N,k);
    res = symnmf_c(H);
    PyObject* py_result = convertDoubleArrayToPyFloatArray(res,N,k);
    for (i = 0; i<N;i++) {
        if(!W[i])
            free(W[i]);
    }
    if(!W)
        free(W);
    for (i = 0; i<N;i++) {
        if(!H[i])
            free(H[i]);
    }
    if(!H)
        free(H);
    if(!res[0])
        free(res[0]);
    if(res)
        free(res);
    return py_result;
}

static PyMethodDef symnmfMethods[] = {
        {"sym",
                (PyCFunction) sym,
                     METH_VARARGS,
                PyDoc_STR("Form the similarity matrix A from X")},

        {"norm",
                (PyCFunction) norm,
                     METH_VARARGS,
                PyDoc_STR("Compute the normalized similarity W")},

        {"ddg",
                (PyCFunction) ddg,
                     METH_VARARGS,
                PyDoc_STR("Compute the Diagonal Degree Matrix")},

        {"symnmf",
                (PyCFunction) symnmf,
                     METH_VARARGS,
                PyDoc_STR("Compute the Symmetric Nonnegative Matrix Factorization")},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
        PyModuleDef_HEAD_INIT,
        "symnmf_capi", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        symnmfMethods /* the PyMethodDef array from before containing the methods of the extension */
};


PyMODINIT_FUNC PyInit_symnmf_capi(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m) {
        return NULL;
    }
    return m;
}
