/* Header to test of C modules for arrays for Python: C_test.c */

/* ==== Prototypes =================================== */

// .... Python callable Vector functions ..................
static PyObject *vecfcn1(PyObject *self, PyObject *args);
static PyObject *vecsq(PyObject *self, PyObject *args);

/* .... C vector utility functions ..................*/
PyArrayObject *pyvector(PyObject *objin);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
int  not_doublevector(PyArrayObject *vec);


/* .... Python callable Matrix functions ..................*/
static PyObject *rowx2(PyObject *self, PyObject *args);
static PyObject *rowx2_v2(PyObject *self, PyObject *args);
static PyObject *matsq(PyObject *self, PyObject *args);
static PyObject *contigmat(PyObject *self, PyObject *args);

/* .... C matrix utility functions ..................*/
PyArrayObject *pymatrix(PyObject *objin);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
void free_Carrayptrs(double **v);
int  not_doublematrix(PyArrayObject *mat);

/* .... Python callable integer 2D array functions ..................*/
static PyObject *intfcn1(PyObject *self, PyObject *args);

/* .... C 2D int array utility functions ..................*/
PyArrayObject *pyint2Darray(PyObject *objin);
int **pyint2Darray_to_Carrayptrs(PyArrayObject *arrayin);
int **ptrintvector(long n);
void free_Cint2Darrayptrs(int **v);
int  not_int2Darray(PyArrayObject *mat);
