//
// auto-generated by ops.py on 2013-11-19 16:38
//

__constant__ int xdim0_advec_cell_kernel3_xdir;
__constant__ int xdim1_advec_cell_kernel3_xdir;
__constant__ int xdim2_advec_cell_kernel3_xdir;
__constant__ int xdim3_advec_cell_kernel3_xdir;
__constant__ int xdim4_advec_cell_kernel3_xdir;
__constant__ int xdim5_advec_cell_kernel3_xdir;
__constant__ int xdim6_advec_cell_kernel3_xdir;
__constant__ int xdim7_advec_cell_kernel3_xdir;

#define OPS_ACC0(x,y) (x+xdim0_advec_cell_kernel3_xdir*(y))
#define OPS_ACC1(x,y) (x+xdim1_advec_cell_kernel3_xdir*(y))
#define OPS_ACC2(x,y) (x+xdim2_advec_cell_kernel3_xdir*(y))
#define OPS_ACC3(x,y) (x+xdim3_advec_cell_kernel3_xdir*(y))
#define OPS_ACC4(x,y) (x+xdim4_advec_cell_kernel3_xdir*(y))
#define OPS_ACC5(x,y) (x+xdim5_advec_cell_kernel3_xdir*(y))
#define OPS_ACC6(x,y) (x+xdim6_advec_cell_kernel3_xdir*(y))
#define OPS_ACC7(x,y) (x+xdim7_advec_cell_kernel3_xdir*(y))

//user function
__device__

inline void advec_cell_kernel3_xdir( const double *vol_flux_x, const double *pre_vol, const int *xx,
                              const double *vertexdx,
                              const double *density1, const double *energy1 ,
                              double *mass_flux_x, double *ener_flux) {

  double sigma, sigmat, sigmav, sigmam, sigma3, sigma4;
  double diffuw, diffdw, limiter;
  double one_by_six = 1.0/6.0;

  //int x_max=field->x_max;

  int upwind,donor,downwind,dif;

  //pre_vol accessed with: {0,0, 1,0, -1,0, -2,0};
  //vertexdx accessed with: {0,0, 1,0, -1,0, -2,0};
  //density1, energy1 accessed with: {0,0, 1,0, -1,0, -2,0};
  //xx accessed with: {0,0 ,1,0}

  if(vol_flux_x[OPS_ACC0(0,0)] > 0.0) {
    upwind   = -2; //j-2
    donor    = -1; //j-1
    downwind = 0; //j
    dif      = donor;
  }
  else if (xx[OPS_ACC2(1,0)] < x_max+2) {
    upwind   = 1; //j+1
    donor    = 0; //j
    downwind = -1; //j-1
    dif      = upwind;
  } else { //*xx[OPS_ACC2(1,0)] >= x_max+2 , then need 0
    upwind   = 0; //xmax+2
    donor    = 0; //j
    downwind = -1; //j-1
    dif      = upwind;
  }
  //return;

  sigmat = fabs(vol_flux_x[OPS_ACC0(0,0)])/pre_vol[OPS_ACC1(donor,0)];
  sigma3 = (1.0 + sigmat)*(vertexdx[OPS_ACC3(0,0)]/vertexdx[OPS_ACC3(dif,0)]);
  sigma4 = 2.0 - sigmat;

  sigma = sigmat;
  sigmav = sigmat;

  diffuw = density1[OPS_ACC4(donor,0)] - density1[OPS_ACC4(upwind,0)];
  diffdw = density1[OPS_ACC4(downwind,0)] - density1[OPS_ACC4(donor,0)];

  if( (diffuw*diffdw) > 0.0)
    limiter=(1.0 - sigmav) * SIGN(1.0 , diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  mass_flux_x[OPS_ACC6(0,0)] = (vol_flux_x[OPS_ACC0(0,0)]) * ( density1[OPS_ACC4(donor,0)] + limiter );

  sigmam = fabs(mass_flux_x[OPS_ACC6(0,0)])/( density1[OPS_ACC4(donor,0)] * pre_vol[OPS_ACC1(donor,0)]);
  diffuw = energy1[OPS_ACC5(donor,0)] - energy1[OPS_ACC5(upwind,0)];
  diffdw = energy1[OPS_ACC5(downwind,0)] - energy1[OPS_ACC5(donor,0)];

  if( (diffuw*diffdw) > 0.0)
    limiter = (1.0 - sigmam) * SIGN(1.0,diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  ener_flux[OPS_ACC7(0,0)] = mass_flux_x[OPS_ACC6(0,0)] * ( energy1[OPS_ACC0(donor,0)] + limiter );
}

#undef OPS_ACC0
#undef OPS_ACC1
#undef OPS_ACC2
#undef OPS_ACC3
#undef OPS_ACC4
#undef OPS_ACC5
#undef OPS_ACC6
#undef OPS_ACC7


__global__ void ops_advec_cell_kernel3_xdir(
const double* __restrict arg0,
const double* __restrict arg1,
const int* __restrict arg2,
const double* __restrict arg3,
const double* __restrict arg4,
const double* __restrict arg5,
double* __restrict arg6,
double* __restrict arg7,
int size0,
int size1 ){


  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

  arg0 += idx_x * 1 + idx_y * 1 * xdim0_advec_cell_kernel3_xdir;
  arg1 += idx_x * 1 + idx_y * 1 * xdim1_advec_cell_kernel3_xdir;
  arg2 += idx_x * 1 + idx_y * 0 * xdim2_advec_cell_kernel3_xdir;
  arg3 += idx_x * 1 + idx_y * 0 * xdim3_advec_cell_kernel3_xdir;
  arg4 += idx_x * 1 + idx_y * 1 * xdim4_advec_cell_kernel3_xdir;
  arg5 += idx_x * 1 + idx_y * 1 * xdim5_advec_cell_kernel3_xdir;
  arg6 += idx_x * 1 + idx_y * 1 * xdim6_advec_cell_kernel3_xdir;
  arg7 += idx_x * 1 + idx_y * 1 * xdim7_advec_cell_kernel3_xdir;

  if (idx_x < size0 && idx_y < size1) {
    advec_cell_kernel3_xdir(arg0, arg1, arg2, arg3,
                   arg4, arg5, arg6, arg7);
  }

}

// host stub function
void ops_par_loop_advec_cell_kernel3_xdir(char const *name, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7) {

  ops_arg args[8] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7};


  int x_size = range[1]-range[0];
  int y_size = range[3]-range[2];

  int xdim0 = args[0].dat->block_size[0];
  int xdim1 = args[1].dat->block_size[0];
  int xdim2 = args[2].dat->block_size[0];
  int xdim3 = args[3].dat->block_size[0];
  int xdim4 = args[4].dat->block_size[0];
  int xdim5 = args[5].dat->block_size[0];
  int xdim6 = args[6].dat->block_size[0];
  int xdim7 = args[7].dat->block_size[0];

  ops_timing_realloc(7);
  if (OPS_kernels[7].count == 0) {
    cudaMemcpyToSymbol( xdim0_advec_cell_kernel3_xdir, &xdim0, sizeof(int) );
    cudaMemcpyToSymbol( xdim1_advec_cell_kernel3_xdir, &xdim1, sizeof(int) );
    cudaMemcpyToSymbol( xdim2_advec_cell_kernel3_xdir, &xdim2, sizeof(int) );
    cudaMemcpyToSymbol( xdim3_advec_cell_kernel3_xdir, &xdim3, sizeof(int) );
    cudaMemcpyToSymbol( xdim4_advec_cell_kernel3_xdir, &xdim4, sizeof(int) );
    cudaMemcpyToSymbol( xdim5_advec_cell_kernel3_xdir, &xdim5, sizeof(int) );
    cudaMemcpyToSymbol( xdim6_advec_cell_kernel3_xdir, &xdim6, sizeof(int) );
    cudaMemcpyToSymbol( xdim7_advec_cell_kernel3_xdir, &xdim7, sizeof(int) );
  }



  dim3 grid( (x_size-1)/OPS_block_size_x+ 1, (y_size-1)/OPS_block_size_y + 1, 1);
  dim3 block(OPS_block_size_x,OPS_block_size_y,1);




  char *p_a[8];


  //set up initial pointers
  p_a[0] = &args[0].data_d[
  + args[0].dat->size * args[0].dat->block_size[0] * ( range[2] * 1 - args[0].dat->offset[1] )
  + args[0].dat->size * ( range[0] * 1 - args[0].dat->offset[0] ) ];

  p_a[1] = &args[1].data_d[
  + args[1].dat->size * args[1].dat->block_size[0] * ( range[2] * 1 - args[1].dat->offset[1] )
  + args[1].dat->size * ( range[0] * 1 - args[1].dat->offset[0] ) ];

  p_a[2] = &args[2].data_d[
  + args[2].dat->size * args[2].dat->block_size[0] * ( range[2] * 0 - args[2].dat->offset[1] )
  + args[2].dat->size * ( range[0] * 1 - args[2].dat->offset[0] ) ];

  p_a[3] = &args[3].data_d[
  + args[3].dat->size * args[3].dat->block_size[0] * ( range[2] * 0 - args[3].dat->offset[1] )
  + args[3].dat->size * ( range[0] * 1 - args[3].dat->offset[0] ) ];

  p_a[4] = &args[4].data_d[
  + args[4].dat->size * args[4].dat->block_size[0] * ( range[2] * 1 - args[4].dat->offset[1] )
  + args[4].dat->size * ( range[0] * 1 - args[4].dat->offset[0] ) ];

  p_a[5] = &args[5].data_d[
  + args[5].dat->size * args[5].dat->block_size[0] * ( range[2] * 1 - args[5].dat->offset[1] )
  + args[5].dat->size * ( range[0] * 1 - args[5].dat->offset[0] ) ];

  p_a[6] = &args[6].data_d[
  + args[6].dat->size * args[6].dat->block_size[0] * ( range[2] * 1 - args[6].dat->offset[1] )
  + args[6].dat->size * ( range[0] * 1 - args[6].dat->offset[0] ) ];

  p_a[7] = &args[7].data_d[
  + args[7].dat->size * args[7].dat->block_size[0] * ( range[2] * 1 - args[7].dat->offset[1] )
  + args[7].dat->size * ( range[0] * 1 - args[7].dat->offset[0] ) ];


  ops_halo_exchanges_cuda(args, 8);


  //call kernel wrapper function, passing in pointers to data
  ops_advec_cell_kernel3_xdir<<<grid, block >>> (  (double *)p_a[0], (double *)p_a[1],
           (int *)p_a[2], (double *)p_a[3],
           (double *)p_a[4], (double *)p_a[5],
           (double *)p_a[6], (double *)p_a[7],x_size, y_size);

  ops_set_dirtybit_cuda(args, 8);
  OPS_kernels[7].count++;
}
