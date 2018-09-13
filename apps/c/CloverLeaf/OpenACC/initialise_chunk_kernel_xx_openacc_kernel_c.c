//
// auto-generated by ops.py
//
#include "./OpenACC/clover_leaf_common.h"

#define OPS_GPU

int xdim0_initialise_chunk_kernel_xx;


#undef OPS_ACC0


#define OPS_ACC0(x,y) (x+xdim0_initialise_chunk_kernel_xx*(y))

//user function
inline 
void initialise_chunk_kernel_xx(int *xx, int *idx) {
  xx[OPS_ACC0(0,0)] = idx[0]-2;
}


#undef OPS_ACC0



void initialise_chunk_kernel_xx_c_wrapper(
  int *p_a0,
  int *p_a1,
  int arg_idx0, int arg_idx1,
  int x_size, int y_size) {
  #ifdef OPS_GPU
  #pragma acc parallel deviceptr(p_a0)
  #pragma acc loop
  #endif
  for ( int n_y=0; n_y<y_size; n_y++ ){
    #ifdef OPS_GPU
    #pragma acc loop
    #endif
    for ( int n_x=0; n_x<x_size; n_x++ ){
      int arg_idx[] = {arg_idx0+n_x, arg_idx1+n_y};
      initialise_chunk_kernel_xx(  p_a0 + n_x*1*1 + n_y*xdim0_initialise_chunk_kernel_xx*0*1,
          arg_idx );

    }
  }
}
