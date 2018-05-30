//
// auto-generated by ops.py
//
__constant__ int xdim0_initialise_chunk_kernel_xx;
int xdim0_initialise_chunk_kernel_xx_h = -1;

#undef OPS_ACC0


#define OPS_ACC0(x,y) (x+xdim0_initialise_chunk_kernel_xx*(y))

//user function
__device__

void initialise_chunk_kernel_xx_gpu(int *xx, int *idx) {
  xx[OPS_ACC0(0,0)] = idx[0]-2;
}



#undef OPS_ACC0


__global__ void ops_initialise_chunk_kernel_xx(
int* __restrict arg0,
int arg_idx0, int arg_idx1,
int size0,
int size1 ){


  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

  int arg_idx[2];
  arg_idx[0] = arg_idx0+idx_x;
  arg_idx[1] = arg_idx1+idx_y;
  arg0 += idx_x * 1*1 + idx_y * 0*1 * xdim0_initialise_chunk_kernel_xx;

  if (idx_x < size0 && idx_y < size1) {
    initialise_chunk_kernel_xx_gpu(arg0, arg_idx);
  }

}

// host stub function
#ifndef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_xx(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1) {
#else
#ifdef OPS_HYBRID
void ops_par_loop_initialise_chunk_kernel_xx_execute_gpu(ops_kernel_descriptor *desc) {
#else
void ops_par_loop_initialise_chunk_kernel_xx_execute(ops_kernel_descriptor *desc) {
#endif
  int dim = desc->dim;
  int *range = desc->range;
  ops_arg arg0 = desc->args[0];
  ops_arg arg1 = desc->args[1];
  #endif

  //Timing
  double t1,t2,c1,c2;

  ops_arg args[2] = { arg0, arg1};


  #if CHECKPOINTING && !OPS_LAZY
  if (!ops_checkpointing_before(args,2,range,35)) return;
  #endif

  if (OPS_diags > 1) {
    ops_timing_realloc(35,"initialise_chunk_kernel_xx");
    OPS_kernels[35].count++;
    ops_timers_core(&c1,&t1);
  }

  //compute locally allocated range for the sub-block
  int start[2];
  int end[2];
  #if OPS_MPI && !OPS_LAZY
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned) return;
  for ( int n=0; n<2; n++ ){
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
  for ( int n=0; n<2; n++ ){
    start[n] = range[2*n];end[n] = range[2*n+1];
  }
  #endif

  int x_size = MAX(0,end[0]-start[0]);
  int y_size = MAX(0,end[1]-start[1]);

  int arg_idx[2];
  #ifdef OPS_MPI
  #ifdef OPS_LAZY
  ops_block block = desc->block;
  sub_block_list sb = OPS_sub_block_list[block->index];
  #endif
  arg_idx[0] = sb->decomp_disp[0]+start[0];
  arg_idx[1] = sb->decomp_disp[1]+start[1];
  #else
  arg_idx[0] = start[0];
  arg_idx[1] = start[1];
  #endif
  int xdim0 = args[0].dat->size[0];

  if (xdim0 != xdim0_initialise_chunk_kernel_xx_h) {
    cudaMemcpyToSymbolAsync( xdim0_initialise_chunk_kernel_xx, &xdim0, sizeof(int),0 );
    xdim0_initialise_chunk_kernel_xx_h = xdim0;
  }



  dim3 grid( (x_size-1)/OPS_block_size_x+ 1, (y_size-1)/OPS_block_size_y + 1, 1);
  dim3 tblock(OPS_block_size_x,OPS_block_size_y,1);



  int dat0 = (OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);

  char *p_a[2];

  //set up initial pointers
  int base0 = args[0].dat->base_offset + 
           dat0 * 1 * (start[0] * args[0].stencil->stride[0]);
  base0 = base0+ dat0 *
    args[0].dat->size[0] *
    (start[1] * args[0].stencil->stride[1]);
  p_a[0] = (char *)args[0].data_d + base0;


  #ifndef OPS_LAZY
  ops_H_D_exchanges_device(args, 2);
  ops_halo_exchanges(args,2,range);
  #endif

  if (OPS_diags > 1) {
    ops_timers_core(&c2,&t2);
    OPS_kernels[35].mpi_time += t2-t1;
  }


  //call kernel wrapper function, passing in pointers to data
  ops_initialise_chunk_kernel_xx<<<grid, tblock >>> (  (int *)p_a[0], arg_idx[0], arg_idx[1],x_size, y_size);

  if (OPS_diags>1) {
    cutilSafeCall(cudaDeviceSynchronize());
    ops_timers_core(&c1,&t1);
    OPS_kernels[35].time += t1-t2;
  }

  #ifndef OPS_LAZY
  ops_set_dirtybit_device(args, 2);
  ops_set_halo_dirtybit3(&args[0],range);
  #endif

  if (OPS_diags > 1) {
    //Update kernel record
    ops_timers_core(&c2,&t2);
    OPS_kernels[35].mpi_time += t2-t1;
    OPS_kernels[35].transfer += ops_compute_transfer(dim, start, end, &arg0);
  }
}

#ifdef OPS_HYBRID
void ops_par_loop_initialise_chunk_kernel_xx_execute_cpu(ops_kernel_descriptor *desc);
void ops_par_loop_initialise_chunk_kernel_xx_execute(ops_kernel_descriptor *desc) {
  if (desc->device == 1) {
    ops_par_loop_initialise_chunk_kernel_xx_execute_gpu(desc);
  }
  else {
    ops_par_loop_initialise_chunk_kernel_xx_execute_cpu(desc);
  }
}
#endif


#ifdef OPS_LAZY
void ops_par_loop_initialise_chunk_kernel_xx(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1) {
  ops_kernel_descriptor *desc = (ops_kernel_descriptor *)malloc(sizeof(ops_kernel_descriptor));
  desc->name = name;
  desc->block = block;
  desc->dim = dim;
  desc->device = 1;
  desc->index = 35;
  desc->hash = 5381;
  desc->hash = ((desc->hash << 5) + desc->hash) + 35;
  for ( int i=0; i<4; i++ ){
    desc->range[i] = range[i];
    desc->orig_range[i] = range[i];
    desc->hash = ((desc->hash << 5) + desc->hash) + range[i];
  }
  desc->nargs = 2;
  desc->args = (ops_arg*)malloc(2*sizeof(ops_arg));
  desc->args[0] = arg0;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg0.dat->index;
  desc->args[1] = arg1;
  desc->function = ops_par_loop_initialise_chunk_kernel_xx_execute;
  if (OPS_diags > 1) {
    ops_timing_realloc(35,"initialise_chunk_kernel_xx");
  }
  ops_enqueue_kernel(desc);
}
#endif
