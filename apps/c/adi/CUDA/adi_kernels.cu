//
// auto-generated by ops.py//

// header
#define OPS_ACC_MD_MACROS
#define OPS_3D
#include "ops_lib_cpp.h"

#include "ops_cuda_reduction.h"
#include "ops_cuda_rt_support.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
// global constants
__constant__ int nx;
__constant__ int ny;
__constant__ int nz;
__constant__ double lambda;

void ops_decl_const_char(int dim, char const *type, int size, char *dat,
                         char const *name) {
  if (!strcmp(name, "nx")) {
    cutilSafeCall(cudaMemcpyToSymbol(nx, dat, dim * size));
  } else if (!strcmp(name, "ny")) {
    cutilSafeCall(cudaMemcpyToSymbol(ny, dat, dim * size));
  } else if (!strcmp(name, "nz")) {
    cutilSafeCall(cudaMemcpyToSymbol(nz, dat, dim * size));
  } else if (!strcmp(name, "lambda")) {
    cutilSafeCall(cudaMemcpyToSymbol(lambda, dat, dim * size));
  } else {
    printf("error: unknown const name\n");
    exit(1);
  }
}

// user kernel files
#include "init_kernel_cuda_kernel.cu"
#include "preproc_kernel_cuda_kernel.cu"
