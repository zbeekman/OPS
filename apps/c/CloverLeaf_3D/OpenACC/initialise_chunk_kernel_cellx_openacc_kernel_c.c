//
// auto-generated by ops.py
//
#include "./OpenACC/clover_leaf_common.h"

#define OPS_GPU

int xdim0_initialise_chunk_kernel_cellx;
int ydim0_initialise_chunk_kernel_cellx;
int xdim1_initialise_chunk_kernel_cellx;
int ydim1_initialise_chunk_kernel_cellx;
int xdim2_initialise_chunk_kernel_cellx;
int ydim2_initialise_chunk_kernel_cellx;


#undef OPS_ACC0
#undef OPS_ACC1
#undef OPS_ACC2


#define OPS_ACC0(x,y,z) (x+xdim0_initialise_chunk_kernel_cellx*(y)+xdim0_initialise_chunk_kernel_cellx*ydim0_initialise_chunk_kernel_cellx*(z))
#define OPS_ACC1(x,y,z) (x+xdim1_initialise_chunk_kernel_cellx*(y)+xdim1_initialise_chunk_kernel_cellx*ydim1_initialise_chunk_kernel_cellx*(z))
#define OPS_ACC2(x,y,z) (x+xdim2_initialise_chunk_kernel_cellx*(y)+xdim2_initialise_chunk_kernel_cellx*ydim2_initialise_chunk_kernel_cellx*(z))

//user function
inline 
void initialise_chunk_kernel_cellx(const double *vertexx, double* cellx, double *celldx) {
  double d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  cellx[OPS_ACC1(0,0,0)]  = 0.5*( vertexx[OPS_ACC0(0,0,0)] + vertexx[OPS_ACC0(1,0,0)] );
  celldx[OPS_ACC2(0,0,0)]  = d_x;




}


#undef OPS_ACC0
#undef OPS_ACC1
#undef OPS_ACC2



void initialise_chunk_kernel_cellx_c_wrapper(
  double *p_a0,
  double *p_a1,
  double *p_a2,
  int x_size, int y_size, int z_size) {
  #ifdef OPS_GPU
  #pragma acc parallel deviceptr(p_a0,p_a1,p_a2)
  #pragma acc loop
  #endif
  for ( int n_z=0; n_z<z_size; n_z++ ){
    #ifdef OPS_GPU
    #pragma acc loop
    #endif
    for ( int n_y=0; n_y<y_size; n_y++ ){
      #ifdef OPS_GPU
      #pragma acc loop
      #endif
      for ( int n_x=0; n_x<x_size; n_x++ ){
        initialise_chunk_kernel_cellx(  p_a0 + n_x*1*1 + n_y*xdim0_initialise_chunk_kernel_cellx*0*1 + n_z*xdim0_initialise_chunk_kernel_cellx*ydim0_initialise_chunk_kernel_cellx*0*1,
           p_a1 + n_x*1*1 + n_y*xdim1_initialise_chunk_kernel_cellx*0*1 + n_z*xdim1_initialise_chunk_kernel_cellx*ydim1_initialise_chunk_kernel_cellx*0*1,
           p_a2 + n_x*1*1 + n_y*xdim2_initialise_chunk_kernel_cellx*0*1 + n_z*xdim2_initialise_chunk_kernel_cellx*ydim2_initialise_chunk_kernel_cellx*0*1 );

      }
    }
  }
}
