#define FFI_SCOPE "Rindow\\Matlib\\FFI"
//#define FFI_LIB "rindowmatlib.dll"

/////////////////////////////////////////////
typedef int8_t                      cl_char;
typedef uint8_t                     cl_uchar;
typedef int16_t                     cl_short;
typedef uint16_t                    cl_ushort;
typedef int32_t                     cl_int;
typedef uint32_t                    cl_uint;
typedef int64_t                     cl_long;
typedef uint64_t                    cl_ulong;
/////////////////////////////////////////////
typedef uint16_t                    bfloat16;
typedef int32_t                     blasint;
typedef int32_t                     lapack_int;
/////////////////////////////////////////////

enum rindow_matlib_dtype {
    rindow_matlib_dtype_unknown   = 0,
    rindow_matlib_dtype_bool      = 1,
    rindow_matlib_dtype_int8      = 2,
    rindow_matlib_dtype_int16     = 3,
    rindow_matlib_dtype_int32     = 4,
    rindow_matlib_dtype_int64     = 5,
    rindow_matlib_dtype_uint8     = 6,
    rindow_matlib_dtype_uint16    = 7,
    rindow_matlib_dtype_uint32    = 8,
    rindow_matlib_dtype_uint64    = 9,
    rindow_matlib_dtype_float8    = 10,
    rindow_matlib_dtype_float16   = 11,
    rindow_matlib_dtype_float32   = 12,
    rindow_matlib_dtype_float64   = 13,
    rindow_matlib_dtype_complex8  = 14,
    rindow_matlib_dtype_complex16 = 15,
    rindow_matlib_dtype_complex32 = 16,
    rindow_matlib_dtype_complex64 = 17
};



/*Set the number of threads on runtime.*/
int32_t rindow_matlib_common_get_nprocs(void);

/* Matlib is compiled for sequential use  */
#define MATLIB_SEQUENTIAL  0
/* Matlib is compiled using normal threading model */
#define MATLIB_THREAD  1
/* Matlib is compiled using OpenMP threading model */
#define MATLIB_OPENMP 2

//#define CBLAS_INDEX size_t
typedef size_t CBLAS_INDEX;

#define RINDOW_MATLIB_SUCCESS                 0
#define RINDOW_MATLIB_E_MEM_ALLOC_FAILURE     -101
#define RINDOW_MATLIB_E_PERM_OUT_OF_RANGE     -102
#define RINDOW_MATLIB_E_DUP_AXIS              -103
#define RINDOW_MATLIB_E_UNSUPPORTED_DATA_TYPE -104
#define RINDOW_MATLIB_E_UNMATCH_IMAGE_BUFFER_SIZE -105
#define RINDOW_MATLIB_E_UNMATCH_COLS_BUFFER_SIZE -106
#define RINDOW_MATLIB_E_INVALID_SHAPE_OR_PARAM -107
#define RINDOW_MATLIB_E_IMAGES_OUT_OF_RANGE   -108
#define RINDOW_MATLIB_E_COLS_OUT_OF_RANGE     -109

// Matlib is compiled for sequential use
#define RINDOW_MATLIB_SEQUENTIAL 0;
// Matlib is compiled using normal threading model
#define RINDOW_MATLIB_THREAD     1;
// Matlib is compiled using OpenMP threading model
#define RINDOW_MATLIB_OPENMP     2;

//#define RINDOW_MATLIB_NO_TRANS       111
//#define RINDOW_MATLIB_TRANS          112
//#define RINDOW_MATLIB_CONJ_TRANS     113
//#define RINDOW_MATLIB_CONJ_NO_TRANS  114
typedef enum RINDOW_MATLIB_TRANSPOSE {
    RINDOW_MATLIB_NO_TRANS=111,
    RINDOW_MATLIB_TRANS=112,
    RINDOW_MATLIB_CONJ_TRANSTrans=113,
    RINDOW_MATLIB_CONJ_NO_TRANSNoTrans=114
};

int32_t rindow_matlib_common_get_nprocs(void);
int32_t rindow_matlib_common_get_num_threads(void);
int32_t rindow_matlib_common_get_parallel(void);
char* rindow_matlib_common_get_version(void);

float rindow_matlib_s_sum(int32_t n,float *x,int32_t incX);
double rindow_matlib_d_sum(int32_t n,double *x,int32_t incX);
void rindow_matlib_s_add(int32_t trans,int32_t m,int32_t n,float alpha,float *x, int32_t incX,float *a, int32_t ldA);
void rindow_matlib_d_add(int32_t trans,int32_t m,int32_t n,double alpha,double *x, int32_t incX,double *a, int32_t ldA);
