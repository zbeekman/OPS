//
// auto-generated by ops.py on 2014-05-23 12:13
//

#include "./OpenACC/clover_leaf_common.h"

#define OPS_GPU

int xdim0_advec_cell_kernel2_ydir;
int xdim1_advec_cell_kernel2_ydir;
int xdim2_advec_cell_kernel2_ydir;
int xdim3_advec_cell_kernel2_ydir;

#define OPS_ACC0(x,y) (x+xdim0_advec_cell_kernel2_ydir*(y))
#define OPS_ACC1(x,y) (x+xdim1_advec_cell_kernel2_ydir*(y))
#define OPS_ACC2(x,y) (x+xdim2_advec_cell_kernel2_ydir*(y))
#define OPS_ACC3(x,y) (x+xdim3_advec_cell_kernel2_ydir*(y))

//user function

inline void advec_cell_kernel2_ydir( double *pre_vol, double *post_vol, const double *volume,
                        const double *vol_flux_y) {

  pre_vol[OPS_ACC0(0,0)] = volume[OPS_ACC2(0,0)] + vol_flux_y[OPS_ACC3(0,1)] - vol_flux_y[OPS_ACC3(0,0)];
  post_vol[OPS_ACC1(0,0)] = volume[OPS_ACC2(0,0)];

}



#undef OPS_ACC0
#undef OPS_ACC1
#undef OPS_ACC2
#undef OPS_ACC3



// host stub function
void ops_par_loop_advec_cell_kernel2_ydir(char const *name, ops_block Block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3) {

  ops_arg args[4] = { arg0, arg1, arg2, arg3};

  sub_block_list sb = OPS_sub_block_list[Block->index];
  //compute localy allocated range for the sub-block
  int start_add[2];
  int end_add[2];
  for ( int n=0; n<2; n++ ){
    start_add[n] = sb->istart[n];end_add[n] = sb->iend[n]+1;
    if (start_add[n] >= range[2*n]) {
      start_add[n] = 0;
    }
    else {
      start_add[n] = range[2*n] - start_add[n];
    }
    if (end_add[n] >= range[2*n+1]) {
      end_add[n] = range[2*n+1] - sb->istart[n];
    }
    else {
      end_add[n] = sb->sizes[n];
    }
  }


  int x_size = MAX(0,end_add[0]-start_add[0]);
  int y_size = MAX(0,end_add[1]-start_add[1]);


  //Timing
  double t1,t2,c1,c2;
  ops_timing_realloc(12,"advec_cell_kernel2_ydir");
  ops_timers_core(&c2,&t2);

  if (OPS_kernels[12].count == 0) {
    xdim0_advec_cell_kernel2_ydir = args[0].dat->block_size[0]*args[0].dat->dim;
    xdim1_advec_cell_kernel2_ydir = args[1].dat->block_size[0]*args[1].dat->dim;
    xdim2_advec_cell_kernel2_ydir = args[2].dat->block_size[0]*args[2].dat->dim;
    xdim3_advec_cell_kernel2_ydir = args[3].dat->block_size[0]*args[3].dat->dim;
  }

  int dat0 = args[0].dat->size;
  int dat1 = args[1].dat->size;
  int dat2 = args[2].dat->size;
  int dat3 = args[3].dat->size;


  //set up initial pointers
  int base0 = dat0 * 1 * 
  (start_add[0] * args[0].stencil->stride[0] - args[0].dat->offset[0]);
  base0 = base0  + dat0 * args[0].dat->block_size[0] * 
  (start_add[1] * args[0].stencil->stride[1] - args[0].dat->offset[1]);
  #ifdef OPS_GPU
  double *p_a0 = (double *)((char *)args[0].data_d + base0);
  #else
  double *p_a0 = (double *)((char *)args[0].data + base0);
  #endif

  //set up initial pointers
  int base1 = dat1 * 1 * 
  (start_add[0] * args[1].stencil->stride[0] - args[1].dat->offset[0]);
  base1 = base1  + dat1 * args[1].dat->block_size[0] * 
  (start_add[1] * args[1].stencil->stride[1] - args[1].dat->offset[1]);
  #ifdef OPS_GPU
  double *p_a1 = (double *)((char *)args[1].data_d + base1);
  #else
  double *p_a1 = (double *)((char *)args[1].data + base1);
  #endif

  //set up initial pointers
  int base2 = dat2 * 1 * 
  (start_add[0] * args[2].stencil->stride[0] - args[2].dat->offset[0]);
  base2 = base2  + dat2 * args[2].dat->block_size[0] * 
  (start_add[1] * args[2].stencil->stride[1] - args[2].dat->offset[1]);
  #ifdef OPS_GPU
  double *p_a2 = (double *)((char *)args[2].data_d + base2);
  #else
  double *p_a2 = (double *)((char *)args[2].data + base2);
  #endif

  //set up initial pointers
  int base3 = dat3 * 1 * 
  (start_add[0] * args[3].stencil->stride[0] - args[3].dat->offset[0]);
  base3 = base3  + dat3 * args[3].dat->block_size[0] * 
  (start_add[1] * args[3].stencil->stride[1] - args[3].dat->offset[1]);
  #ifdef OPS_GPU
  double *p_a3 = (double *)((char *)args[3].data_d + base3);
  #else
  double *p_a3 = (double *)((char *)args[3].data + base3);
  #endif


  #ifdef OPS_GPU
  ops_H_D_exchanges_cuda(args, 4);
  #else
  ops_H_D_exchanges(args, 4);
  #endif
  ops_halo_exchanges(args,4,range);

  ops_timers_core(&c1,&t1);
  OPS_kernels[12].mpi_time += t1-t2;

  #ifdef OPS_GPU
  #pragma acc parallel deviceptr(p_a0,p_a1,p_a2,p_a3)
  #pragma acc loop
  #endif
  for ( int n_y=0; n_y<y_size; n_y++ ){
    #ifdef OPS_GPU
    #pragma acc loop
    #endif
    for ( int n_x=0; n_x<x_size; n_x++ ){
      advec_cell_kernel2_ydir(  p_a0+ n_x*1+n_y*xdim0_advec_cell_kernel2_ydir*1, p_a1+ n_x*1+n_y*xdim1_advec_cell_kernel2_ydir*1,
           p_a2+ n_x*1+n_y*xdim2_advec_cell_kernel2_ydir*1, p_a3+ n_x*1+n_y*xdim3_advec_cell_kernel2_ydir*1 );

    }
  }


  ops_timers_core(&c2,&t2);
  OPS_kernels[12].time += t2-t1;
  #ifdef OPS_GPU
  ops_set_dirtybit_cuda(args, 4);
  #else
  ops_set_dirtybit_host(args, 4);
  #endif
  ops_set_halo_dirtybit3(&args[0],range);
  ops_set_halo_dirtybit3(&args[1],range);

  //Update kernel record
  OPS_kernels[12].count++;
  OPS_kernels[12].transfer += ops_compute_transfer(dim, range, &arg0);
  OPS_kernels[12].transfer += ops_compute_transfer(dim, range, &arg1);
  OPS_kernels[12].transfer += ops_compute_transfer(dim, range, &arg2);
  OPS_kernels[12].transfer += ops_compute_transfer(dim, range, &arg3);
}
