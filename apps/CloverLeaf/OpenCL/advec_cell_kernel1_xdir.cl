//
// auto-generated by ops.py on 2014-05-14 16:53
//


#ifndef MIN
#define MIN(a,b) ((a<b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a>b) ? (a) : (b))
#endif
#ifndef SIGN
#define SIGN(a,b) ((b<0.0) ? (a*(-1)) : (a))
#endif

#pragma OPENCL EXTENSION cl_khr_fp64:enable

#define OPS_ACC0(x,y) (x+xdim0_advec_cell_kernel1_xdir*(y))
#define OPS_ACC1(x,y) (x+xdim1_advec_cell_kernel1_xdir*(y))
#define OPS_ACC2(x,y) (x+xdim2_advec_cell_kernel1_xdir*(y))
#define OPS_ACC3(x,y) (x+xdim3_advec_cell_kernel1_xdir*(y))
#define OPS_ACC4(x,y) (x+xdim4_advec_cell_kernel1_xdir*(y))


//user function

inline void advec_cell_kernel1_xdir( __global double *pre_vol, __global double *post_vol,  __global double *volume,
                         __global double *vol_flux_x,  __global double *vol_flux_y,
  int xdim0_advec_cell_kernel1_xdir,
  int xdim1_advec_cell_kernel1_xdir,
  int xdim2_advec_cell_kernel1_xdir,
  int xdim3_advec_cell_kernel1_xdir,
  int xdim4_advec_cell_kernel1_xdir)
  {

  pre_vol[OPS_ACC0(0,0)] = volume[OPS_ACC2(0,0)] + ( vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)] +
                           vol_flux_y[OPS_ACC4(0,1)] - vol_flux_y[OPS_ACC4(0,0)]);
  post_vol[OPS_ACC1(0,0)] = pre_vol[OPS_ACC0(0,0)] - ( vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)]);

}



 #undef OPS_ACC0
 #undef OPS_ACC1
 #undef OPS_ACC2
 #undef OPS_ACC3
 #undef OPS_ACC4


 __kernel void ops_advec_cell_kernel1_xdir(
 __global double* arg0,
 __global double* arg1,
 __global double* arg2,
 __global double* arg3,
 __global double* arg4,
 int xdim0_advec_cell_kernel1_xdir,
 int xdim1_advec_cell_kernel1_xdir,
 int xdim2_advec_cell_kernel1_xdir,
 int xdim3_advec_cell_kernel1_xdir,
 int xdim4_advec_cell_kernel1_xdir,
 const int base0,
 const int base1,
 const int base2,
 const int base3,
 const int base4,
 int size0,
 int size1 ){


   int idx_y = get_global_id(1);
   int idx_x = get_global_id(0);

   if (idx_x < size0 && idx_y < size1) {
     advec_cell_kernel1_xdir(&arg0[base0 + idx_x * 1 + idx_y * 1 * xdim0_advec_cell_kernel1_xdir],
                       &arg1[base1 + idx_x * 1 + idx_y * 1 * xdim1_advec_cell_kernel1_xdir],
                       &arg2[base2 + idx_x * 1 + idx_y * 1 * xdim2_advec_cell_kernel1_xdir],
                       &arg3[base3 + idx_x * 1 + idx_y * 1 * xdim3_advec_cell_kernel1_xdir],
                       &arg4[base4 + idx_x * 1 + idx_y * 1 * xdim4_advec_cell_kernel1_xdir],
                       
                       xdim0_advec_cell_kernel1_xdir,
                       xdim1_advec_cell_kernel1_xdir,
                       xdim2_advec_cell_kernel1_xdir,
                       xdim3_advec_cell_kernel1_xdir,
                       xdim4_advec_cell_kernel1_xdir);
   }

 }
