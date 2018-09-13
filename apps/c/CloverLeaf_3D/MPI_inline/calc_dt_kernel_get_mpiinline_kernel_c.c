//
// auto-generated by ops.py
//
#include "./MPI_inline/clover_leaf_common.h"

int xdim0_calc_dt_kernel_get;
int ydim0_calc_dt_kernel_get;
int xdim1_calc_dt_kernel_get;
int ydim1_calc_dt_kernel_get;
int xdim4_calc_dt_kernel_get;
int ydim4_calc_dt_kernel_get;


#define OPS_ACC0(x,y,z) (n_x*1+n_y*xdim0_calc_dt_kernel_get*0+n_z*xdim0_calc_dt_kernel_get*ydim0_calc_dt_kernel_get*0+x+xdim0_calc_dt_kernel_get*(y)+xdim0_calc_dt_kernel_get*ydim0_calc_dt_kernel_get*(z))
#define OPS_ACC1(x,y,z) (n_x*0+n_y*xdim1_calc_dt_kernel_get*1+n_z*xdim1_calc_dt_kernel_get*ydim1_calc_dt_kernel_get*0+x+xdim1_calc_dt_kernel_get*(y)+xdim1_calc_dt_kernel_get*ydim1_calc_dt_kernel_get*(z))
#define OPS_ACC4(x,y,z) (n_x*0+n_y*xdim4_calc_dt_kernel_get*0+n_z*xdim4_calc_dt_kernel_get*ydim4_calc_dt_kernel_get*1+x+xdim4_calc_dt_kernel_get*(y)+xdim4_calc_dt_kernel_get*ydim4_calc_dt_kernel_get*(z))

//user function



void calc_dt_kernel_get_c_wrapper(
  const double * restrict cellx,
  const double * restrict celly,
  double * restrict xl_pos_g,
  double * restrict yl_pos_g,
  const double * restrict cellz,
  double * restrict zl_pos_g,
  int x_size, int y_size, int z_size) {
  double xl_pos_v = *xl_pos_g;
  double yl_pos_v = *yl_pos_g;
  double zl_pos_v = *zl_pos_g;
  #pragma omp parallel for reduction(+:xl_pos_v) reduction(+:yl_pos_v) reduction(+:zl_pos_v)
  for ( int n_z=0; n_z<z_size; n_z++ ){
    for ( int n_y=0; n_y<y_size; n_y++ ){
      for ( int n_x=0; n_x<x_size; n_x++ ){
        double * restrict xl_pos = &xl_pos_v;
        double * restrict yl_pos = &yl_pos_v;
        double * restrict zl_pos = &zl_pos_v;
        
  *xl_pos = cellx[OPS_ACC0(0,0,0)];
  *yl_pos = celly[OPS_ACC1(0,0,0)];
  *zl_pos = cellz[OPS_ACC4(0,0,0)];

      }
    }
  }
  *xl_pos_g = xl_pos_v;
  *yl_pos_g = yl_pos_v;
  *zl_pos_g = zl_pos_v;
}
#undef OPS_ACC0
#undef OPS_ACC1
#undef OPS_ACC4

