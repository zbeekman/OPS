//
// auto-generated by op2.py on 2013-10-25 15:16
//

//header
#include "ops_lib_cpp.h"

__constant__ int xdim0_accel;
__constant__ int xdim1_accel;
__constant__ int xdim2_accel;
__constant__ int xdim3_accel;
__constant__ int xdim4_accel;
__constant__ int xdim5_accel;
__constant__ int xdim6_accel;
__constant__ int xdim7_accel;
__constant__ int xdim8_accel;
__constant__ int xdim9_accel;
__constant__ int xdim10_accel;
__constant__ int xdim11_accel;
__constant__ int xdim12_accel;
__constant__ int xdim13_accel;

__constant__ double dt_device;
__constant__ double x_max_device;
__constant__ double y_max_device;

//can have a function to copy device constants here .. see OP2 airfoil_kernels.cu

#define OPS_ACC_MACROS
#define OPS_ACC0(x,y) (x+xdim0_accel*(y))
#define OPS_ACC1(x,y) (x+xdim1_accel*(y))
#define OPS_ACC2(x,y) (x+xdim2_accel*(y))
#define OPS_ACC3(x,y) (x+xdim3_accel*(y))
#define OPS_ACC4(x,y) (x+xdim4_accel*(y))
#define OPS_ACC5(x,y) (x+xdim5_accel*(y))
#define OPS_ACC6(x,y) (x+xdim6_accel*(y))
#define OPS_ACC7(x,y) (x+xdim7_accel*(y))
#define OPS_ACC8(x,y) (x+xdim8_accel*(y))
#define OPS_ACC9(x,y) (x+xdim9_accel*(y))
#define OPS_ACC10(x,y) (x+xdim10_accel*(y))
#define OPS_ACC11(x,y) (x+xdim11_accel*(y))
#define OPS_ACC12(x,y) (x+xdim12_accel*(y))
#define OPS_ACC13(x,y) (x+xdim13_accel*(y))


//user kernel files

#include "accelerate_kernel_stepbymass_cuda_kernel.cu"
#include "accelerate_kernelx1_cuda_kernel.cu"
#include "accelerate_kernely1_cuda_kernel.cu"
#include "accelerate_kernelx2_cuda_kernel.cu"
#include "accelerate_kernely2_cuda_kernel.cu"
#include "viscosity_kernel_cuda_kernel.cu"

#include "PdV_kernel_predict_cuda_kernel.cu"
#include "PdV_kernel_nopredict_cuda_kernel.cu"
#include "revert_kernel_cuda_kernel.cu"
#include "reset_field_kernel1_cuda_kernel.cu"
#include "reset_field_kernel2_cuda_kernel.cu"

#include "advec_mom_kernel_x1_cuda_kernel.cu"
#include "advec_mom_kernel_y1_cuda_kernel.cu"
#include "advec_mom_kernel_x2_cuda_kernel.cu"
#include "advec_mom_kernel_y2_cuda_kernel.cu"
#include "advec_mom_kernel_mass_flux_x_cuda_kernel.cu"
#include "advec_mom_kernel_post_advec_x_cuda_kernel.cu"
#include "advec_mom_kernel_pre_advec_x_cuda_kernel.cu"
#include "advec_mom_kernel1_x_cuda_kernel.cu"
#include "advec_mom_kernel2_x_cuda_kernel.cu"
#include "advec_mom_kernel_mass_flux_y_cuda_kernel.cu"
#include "advec_mom_kernel_post_advec_y_cuda_kernel.cu"
#include "advec_mom_kernel_pre_advec_y_cuda_kernel.cu"
#include "advec_mom_kernel1_y_cuda_kernel.cu"
#include "advec_mom_kernel2_y_cuda_kernel.cu"

#include "advec_cell_kernel1_xdir_cuda_kernel.cu"
#include "advec_cell_kernel2_xdir_cuda_kernel.cu"
#include "advec_cell_kernel3_xdir_cuda_kernel.cu"
#include "advec_cell_kernel4_xdir_cuda_kernel.cu"
#include "advec_cell_kernel1_ydir_cuda_kernel.cu"
#include "advec_cell_kernel2_ydir_cuda_kernel.cu"
#include "advec_cell_kernel3_ydir_cuda_kernel.cu"
#include "advec_cell_kernel4_ydir_cuda_kernel.cu"

#include "ideal_gas_kernel_cuda_kernel.cu"
