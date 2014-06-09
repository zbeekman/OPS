//
// auto-generated by ops.py on 2014-06-09 10:49
//


// host stub function
void ops_par_loop_calc_dt_kernel_min(char const *name, ops_block Block, int dim, int* range,
 ops_arg arg0, ops_arg arg1) {

  buildOpenCLKernels();
  ops_arg args[2] = { arg0, arg1};

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


  int x_size = end_add[0]-start_add[0];
  int y_size = end_add[1]-start_add[1];

  int xdim0 = args[0].dat->block_size[0]*args[0].dat->dim;


  //Timing
  double t1,t2,c1,c2;
  ops_timing_realloc(73,"calc_dt_kernel_min");
  ops_timers_core(&c1,&t1);

  //set up OpenCL thread blocks
  size_t globalWorkSize[3] = {((x_size-1)/OPS_block_size_x+ 1)*OPS_block_size_x, ((y_size-1)/OPS_block_size_y + 1)*OPS_block_size_y, 1};
  size_t localWorkSize[3] =  {OPS_block_size_x,OPS_block_size_y,1};


  double *arg1h = (double *)arg1.data;
  int nblocks = ((x_size-1)/OPS_block_size_x+ 1)*((y_size-1)/OPS_block_size_y + 1);
  int maxblocks = nblocks;
  int reduct_bytes = 0;

  reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double)*64);

  reallocReductArrays(reduct_bytes);
  reduct_bytes = 0;

  int r_bytes1 = reduct_bytes/sizeof(double);
  arg1.data = OPS_reduct_h + reduct_bytes;
  arg1.data_d = OPS_reduct_d;// + reduct_bytes;
  for (int b=0; b<maxblocks; b++)
  for (int d=0; d<1; d++) ((double *)arg1.data)[d+b*1] = INFINITY_double;
  reduct_bytes += ROUND_UP(maxblocks*1*sizeof(double));


  mvReductArraysToDevice(reduct_bytes);
  int dat0 = args[0].dat->size;

  //set up initial pointers
  int base0 = dat0 * 1 * 
  (start_add[0] * args[0].stencil->stride[0] - args[0].dat->offset[0]);
  base0 = base0  + dat0 * args[0].dat->block_size[0] * 
  (start_add[1] * args[0].stencil->stride[1] - args[0].dat->offset[1]);
  base0 = base0/dat0;


  ops_H_D_exchanges_cuda(args, 2);

  int nthread = OPS_block_size_x*OPS_block_size_y;


  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[73], 0, sizeof(cl_mem), (void*) &arg0.data_d ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[73], 1, sizeof(cl_mem), (void*) &arg1.data_d ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[73], 2, nthread*sizeof(double), NULL));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[73], 3, sizeof(cl_int), (void*) &r_bytes1 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[73], 4, sizeof(cl_int), (void*) &xdim0 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[73], 5, sizeof(cl_int), (void*) &base0 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[73], 6, sizeof(cl_int), (void*) &x_size ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[73], 7, sizeof(cl_int), (void*) &y_size ));

  //call/enque opencl kernel wrapper function
  clSafeCall( clEnqueueNDRangeKernel(OPS_opencl_core.command_queue, OPS_opencl_core.kernel[73], 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
  clSafeCall( clFinish(OPS_opencl_core.command_queue) );

  mvReductArraysToHost(reduct_bytes);
  for ( int b=0; b<maxblocks; b++ ){
    for ( int d=0; d<1; d++ ){
      arg1h[d] = MIN(arg1h[d],((double *)arg1.data)[d+b*1]);
    }
  }
  arg1.data = (char *)arg1h;

  ops_set_dirtybit_cuda(args, 2);

  //Update kernel record
  ops_timers_core(&c2,&t2);
  OPS_kernels[73].count++;
  OPS_kernels[73].time += t2-t1;
  OPS_kernels[73].transfer += ops_compute_transfer(dim, range, &arg0);
}
