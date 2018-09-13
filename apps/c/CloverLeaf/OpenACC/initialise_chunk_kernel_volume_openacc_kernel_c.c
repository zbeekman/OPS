//
// auto-generated by ops.py
//
#include "./OpenACC/clover_leaf_common.h"

#define OPS_GPU

int xdim0_initialise_chunk_kernel_volume;
int xdim1_initialise_chunk_kernel_volume;
int xdim2_initialise_chunk_kernel_volume;
int xdim3_initialise_chunk_kernel_volume;
int xdim4_initialise_chunk_kernel_volume;


#undef OPS_ACC0
#undef OPS_ACC1
#undef OPS_ACC2
#undef OPS_ACC3
#undef OPS_ACC4


#define OPS_ACC0(x,y) (x+xdim0_initialise_chunk_kernel_volume*(y))
#define OPS_ACC1(x,y) (x+xdim1_initialise_chunk_kernel_volume*(y))
#define OPS_ACC2(x,y) (x+xdim2_initialise_chunk_kernel_volume*(y))
#define OPS_ACC3(x,y) (x+xdim3_initialise_chunk_kernel_volume*(y))
#define OPS_ACC4(x,y) (x+xdim4_initialise_chunk_kernel_volume*(y))

//user function
inline 
void initialise_chunk_kernel_volume(double *volume, const double *celldy, double *xarea,
                                         const double *celldx, double *yarea) {

  double d_x, d_y;

  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;

  volume[OPS_ACC0(0,0)] = d_x*d_y;
  xarea[OPS_ACC2(0,0)] = celldy[OPS_ACC1(0,0)];
  yarea[OPS_ACC4(0,0)] = celldx[OPS_ACC3(0,0)];
}


#undef OPS_ACC0
#undef OPS_ACC1
#undef OPS_ACC2
#undef OPS_ACC3
#undef OPS_ACC4



void initialise_chunk_kernel_volume_c_wrapper(
  double *p_a0,
  double *p_a1,
  double *p_a2,
  double *p_a3,
  double *p_a4,
  int x_size, int y_size) {
  #ifdef OPS_GPU
  #pragma acc parallel deviceptr(p_a0,p_a1,p_a2,p_a3,p_a4)
  #pragma acc loop
  #endif
  for ( int n_y=0; n_y<y_size; n_y++ ){
    #ifdef OPS_GPU
    #pragma acc loop
    #endif
    for ( int n_x=0; n_x<x_size; n_x++ ){
      initialise_chunk_kernel_volume(  p_a0 + n_x*1*1 + n_y*xdim0_initialise_chunk_kernel_volume*1*1,
           p_a1 + n_x*0*1 + n_y*xdim1_initialise_chunk_kernel_volume*1*1, p_a2 + n_x*1*1 + n_y*xdim2_initialise_chunk_kernel_volume*1*1,
           p_a3 + n_x*1*1 + n_y*xdim3_initialise_chunk_kernel_volume*0*1, p_a4 + n_x*1*1 + n_y*xdim4_initialise_chunk_kernel_volume*1*1 );

    }
  }
}
