//
// auto-generated by ops.py
//
__constant__ int xdim0_write_kernel;
int xdim0_write_kernel_h = -1;
__constant__ int ydim0_write_kernel;
int ydim0_write_kernel_h = -1;
__constant__ int xdim1_write_kernel;
int xdim1_write_kernel_h = -1;
__constant__ int ydim1_write_kernel;
int ydim1_write_kernel_h = -1;
__constant__ int xdim2_write_kernel;
int xdim2_write_kernel_h = -1;
__constant__ int ydim2_write_kernel;
int ydim2_write_kernel_h = -1;

#undef OPS_ACC1
#undef OPS_ACC2

#undef OPS_ACC_MD0

#define OPS_ACC1(x,y,z) (x+xdim1_write_kernel*(y)+xdim1_write_kernel*ydim1_write_kernel*(z))
#define OPS_ACC2(x,y,z) (x+xdim2_write_kernel*(y)+xdim2_write_kernel*ydim2_write_kernel*(z))

#define OPS_ACC_MD0(d,x,y,z) ((x)*2+(d)+(xdim0_write_kernel*(y)*2)+(xdim0_write_kernel*ydim0_write_kernel*(z)*2))
//user function
__device__

void write_kernel_gpu(double *mult, double *single, int *digit, const int *idx) {

  mult[OPS_ACC_MD0(0, 0, 0, 0)] = 1;

  mult[OPS_ACC_MD0(1, 0, 0, 0)] = 2;

  single[OPS_ACC1(0, 0, 0)] = 3;

  digit[OPS_ACC2(0, 0, 0)] = idx[0] * 100 + idx[1] * 10 + idx[2];
}



#undef OPS_ACC1
#undef OPS_ACC2

#undef OPS_ACC_MD0

__global__ void ops_write_kernel(
double* __restrict arg0,
double* __restrict arg1,
int* __restrict arg2,
int arg_idx0, int arg_idx1, int arg_idx2,
int size0,
int size1,
int size2 ){


  int idx_z = blockDim.z * blockIdx.z + threadIdx.z;
  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

  int arg_idx[3];
  arg_idx[0] = arg_idx0+idx_x;
  arg_idx[1] = arg_idx1+idx_y;
  arg_idx[2] = arg_idx2+idx_z;
  arg0 += idx_x * 1*2 + idx_y * 1*2 * xdim0_write_kernel + idx_z * 1*2 * xdim0_write_kernel * ydim0_write_kernel;
  arg1 += idx_x * 1*1 + idx_y * 1*1 * xdim1_write_kernel + idx_z * 1*1 * xdim1_write_kernel * ydim1_write_kernel;
  arg2 += idx_x * 1*1 + idx_y * 1*1 * xdim2_write_kernel + idx_z * 1*1 * xdim2_write_kernel * ydim2_write_kernel;

  if (idx_x < size0 && idx_y < size1 && idx_z < size2) {
    write_kernel_gpu(arg0, arg1, arg2, arg_idx);
  }

}

// host stub function
#ifndef OPS_LAZY
void ops_par_loop_write_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3) {
#else
void ops_par_loop_write_kernel_execute(ops_kernel_descriptor *desc) {
  int dim = desc->dim;
  int *range = desc->range;
  ops_arg arg0 = desc->args[0];
  ops_arg arg1 = desc->args[1];
  ops_arg arg2 = desc->args[2];
  ops_arg arg3 = desc->args[3];
  #endif

  //Timing
  double t1,t2,c1,c2;

  ops_arg args[4] = { arg0, arg1, arg2, arg3};


  #if CHECKPOINTING && !OPS_LAZY
  if (!ops_checkpointing_before(args,4,range,0)) return;
  #endif

  if (OPS_diags > 1) {
    ops_timing_realloc(0,"write_kernel");
    OPS_kernels[0].count++;
    ops_timers_core(&c1,&t1);
  }

  //compute locally allocated range for the sub-block
  int start[3];
  int end[3];
  #if OPS_MPI && !OPS_LAZY
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  for ( int n=0; n<3; n++ ){
    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];
    if (start[n] >= range[2*n]) {
      start[n] = 0;
    }
    else {
      start[n] = range[2*n] - start[n];
    }
    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];
    if (end[n] >= range[2*n+1]) {
      end[n] = range[2*n+1] - sb->decomp_disp[n];
    }
    else {
      end[n] = sb->decomp_size[n];
    }
    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))
      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);
  }
  #else
  for ( int n=0; n<3; n++ ){
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif

  int x_size = MAX(0,end[0]-start[0]);
  int y_size = MAX(0,end[1]-start[1]);
  int z_size = MAX(0,end[2]-start[2]);

  int arg_idx[3];
  #ifdef OPS_MPI
  #ifdef OPS_LAZY
  ops_block block = desc->block;
  sub_block_list sb = OPS_sub_block_list[block->index];
  #endif
  arg_idx[0] = sb->decomp_disp[0]+start[0];
  arg_idx[1] = sb->decomp_disp[1]+start[1];
  arg_idx[2] = sb->decomp_disp[2]+start[2];
  #else
  arg_idx[0] = start[0];
  arg_idx[1] = start[1];
  arg_idx[2] = start[2];
  #endif
  int xdim0 = args[0].dat->size[0];
  int ydim0 = args[0].dat->size[1];
  int xdim1 = args[1].dat->size[0];
  int ydim1 = args[1].dat->size[1];
  int xdim2 = args[2].dat->size[0];
  int ydim2 = args[2].dat->size[1];

  if (xdim0 != xdim0_write_kernel_h || ydim0 != ydim0_write_kernel_h || xdim1 != xdim1_write_kernel_h || ydim1 != ydim1_write_kernel_h || xdim2 != xdim2_write_kernel_h || ydim2 != ydim2_write_kernel_h) {
    cudaMemcpyToSymbol( xdim0_write_kernel, &xdim0, sizeof(int) );
    xdim0_write_kernel_h = xdim0;
    cudaMemcpyToSymbol( ydim0_write_kernel, &ydim0, sizeof(int) );
    ydim0_write_kernel_h = ydim0;
    cudaMemcpyToSymbol( xdim1_write_kernel, &xdim1, sizeof(int) );
    xdim1_write_kernel_h = xdim1;
    cudaMemcpyToSymbol( ydim1_write_kernel, &ydim1, sizeof(int) );
    ydim1_write_kernel_h = ydim1;
    cudaMemcpyToSymbol( xdim2_write_kernel, &xdim2, sizeof(int) );
    xdim2_write_kernel_h = xdim2;
    cudaMemcpyToSymbol( ydim2_write_kernel, &ydim2, sizeof(int) );
    ydim2_write_kernel_h = ydim2;
  }



  dim3 grid( (x_size-1)/OPS_block_size_x+ 1, (y_size-1)/OPS_block_size_y + 1, (z_size-1)/OPS_block_size_z +1);
  dim3 tblock(OPS_block_size_x,OPS_block_size_y,OPS_block_size_z);



  int dat0 = (OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);
  int dat1 = (OPS_soa ? args[1].dat->type_size : args[1].dat->elem_size);
  int dat2 = (OPS_soa ? args[2].dat->type_size : args[2].dat->elem_size);

  char *p_a[4];

  //set up initial pointers
  int base0 = args[0].dat->base_offset + 
           dat0 * 1 * (start[0] * args[0].stencil->stride[0]);
  base0 = base0+ dat0 *
    args[0].dat->size[0] *
    (start[1] * args[0].stencil->stride[1]);
  base0 = base0+ dat0 *
    args[0].dat->size[0] *
    args[0].dat->size[1] *
    (start[2] * args[0].stencil->stride[2]);
  p_a[0] = (char *)args[0].data_d + base0;

  int base1 = args[1].dat->base_offset + 
           dat1 * 1 * (start[0] * args[1].stencil->stride[0]);
  base1 = base1+ dat1 *
    args[1].dat->size[0] *
    (start[1] * args[1].stencil->stride[1]);
  base1 = base1+ dat1 *
    args[1].dat->size[0] *
    args[1].dat->size[1] *
    (start[2] * args[1].stencil->stride[2]);
  p_a[1] = (char *)args[1].data_d + base1;

  int base2 = args[2].dat->base_offset + 
           dat2 * 1 * (start[0] * args[2].stencil->stride[0]);
  base2 = base2+ dat2 *
    args[2].dat->size[0] *
    (start[1] * args[2].stencil->stride[1]);
  base2 = base2+ dat2 *
    args[2].dat->size[0] *
    args[2].dat->size[1] *
    (start[2] * args[2].stencil->stride[2]);
  p_a[2] = (char *)args[2].data_d + base2;


  #ifndef OPS_LAZY
  ops_H_D_exchanges_device(args, 4);
  ops_halo_exchanges(args,4,range);
  #endif

  if (OPS_diags > 1) {
    ops_timers_core(&c2,&t2);
    OPS_kernels[0].mpi_time += t2-t1;
  }


  //call kernel wrapper function, passing in pointers to data
  if (x_size > 0 && y_size > 0 && z_size > 0)
    ops_write_kernel<<<grid, tblock >>> (  (double *)p_a[0], (double *)p_a[1],
           (int *)p_a[2], arg_idx[0], arg_idx[1], arg_idx[2],x_size, y_size, z_size);

  cutilSafeCall(cudaGetLastError());

  if (OPS_diags>1) {
    cutilSafeCall(cudaDeviceSynchronize());
    ops_timers_core(&c1,&t1);
    OPS_kernels[0].time += t1-t2;
  }

  #ifndef OPS_LAZY
  ops_set_dirtybit_device(args, 4);
  ops_set_halo_dirtybit3(&args[0],range);
  ops_set_halo_dirtybit3(&args[1],range);
  ops_set_halo_dirtybit3(&args[2],range);
  #endif

  if (OPS_diags > 1) {
    //Update kernel record
    ops_timers_core(&c2,&t2);
    OPS_kernels[0].mpi_time += t2-t1;
    OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg0);
    OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg1);
    OPS_kernels[0].transfer += ops_compute_transfer(dim, start, end, &arg2);
  }
}

#ifdef OPS_LAZY
void ops_par_loop_write_kernel(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3) {
  ops_kernel_descriptor *desc = (ops_kernel_descriptor *)malloc(sizeof(ops_kernel_descriptor));
  desc->name = name;
  desc->block = block;
  desc->dim = dim;
  desc->device = 1;
  desc->index = 0;
  desc->hash = 5381;
  desc->hash = ((desc->hash << 5) + desc->hash) + 0;
  for ( int i=0; i<6; i++ ){
    desc->range[i] = range[i];
    desc->orig_range[i] = range[i];
    desc->hash = ((desc->hash << 5) + desc->hash) + range[i];
  }
  desc->nargs = 4;
  desc->args = (ops_arg*)malloc(4*sizeof(ops_arg));
  desc->args[0] = arg0;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg0.dat->index;
  desc->args[1] = arg1;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg1.dat->index;
  desc->args[2] = arg2;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg2.dat->index;
  desc->args[3] = arg3;
  desc->function = ops_par_loop_write_kernel_execute;
  if (OPS_diags > 1) {
    ops_timing_realloc(0,"write_kernel");
  }
  ops_enqueue_kernel(desc);
}
#endif
