//
// auto-generated by ops.py
//
#define OPS_ACC0(x,y,z) (n_x*1+n_y*xdim0_calc_dt_kernel_print*1+n_z*xdim0_calc_dt_kernel_print*ydim0_calc_dt_kernel_print*1+x+xdim0_calc_dt_kernel_print*(y)+xdim0_calc_dt_kernel_print*ydim0_calc_dt_kernel_print*(z))
#define OPS_ACC1(x,y,z) (n_x*1+n_y*xdim1_calc_dt_kernel_print*1+n_z*xdim1_calc_dt_kernel_print*ydim1_calc_dt_kernel_print*1+x+xdim1_calc_dt_kernel_print*(y)+xdim1_calc_dt_kernel_print*ydim1_calc_dt_kernel_print*(z))
#define OPS_ACC2(x,y,z) (n_x*1+n_y*xdim2_calc_dt_kernel_print*1+n_z*xdim2_calc_dt_kernel_print*ydim2_calc_dt_kernel_print*1+x+xdim2_calc_dt_kernel_print*(y)+xdim2_calc_dt_kernel_print*ydim2_calc_dt_kernel_print*(z))
#define OPS_ACC3(x,y,z) (n_x*1+n_y*xdim3_calc_dt_kernel_print*1+n_z*xdim3_calc_dt_kernel_print*ydim3_calc_dt_kernel_print*1+x+xdim3_calc_dt_kernel_print*(y)+xdim3_calc_dt_kernel_print*ydim3_calc_dt_kernel_print*(z))
#define OPS_ACC4(x,y,z) (n_x*1+n_y*xdim4_calc_dt_kernel_print*1+n_z*xdim4_calc_dt_kernel_print*ydim4_calc_dt_kernel_print*1+x+xdim4_calc_dt_kernel_print*(y)+xdim4_calc_dt_kernel_print*ydim4_calc_dt_kernel_print*(z))
#define OPS_ACC5(x,y,z) (n_x*1+n_y*xdim5_calc_dt_kernel_print*1+n_z*xdim5_calc_dt_kernel_print*ydim5_calc_dt_kernel_print*1+x+xdim5_calc_dt_kernel_print*(y)+xdim5_calc_dt_kernel_print*ydim5_calc_dt_kernel_print*(z))
#define OPS_ACC6(x,y,z) (n_x*1+n_y*xdim6_calc_dt_kernel_print*1+n_z*xdim6_calc_dt_kernel_print*ydim6_calc_dt_kernel_print*1+x+xdim6_calc_dt_kernel_print*(y)+xdim6_calc_dt_kernel_print*ydim6_calc_dt_kernel_print*(z))


//user function

// host stub function
void ops_par_loop_calc_dt_kernel_print_execute(ops_kernel_descriptor *desc) {
  ops_block block = desc->block;
  int dim = desc->dim;
  int *range = desc->range;
  ops_arg arg0 = desc->args[0];
  ops_arg arg1 = desc->args[1];
  ops_arg arg2 = desc->args[2];
  ops_arg arg3 = desc->args[3];
  ops_arg arg4 = desc->args[4];
  ops_arg arg5 = desc->args[5];
  ops_arg arg6 = desc->args[6];
  ops_arg arg7 = desc->args[7];

  //Timing
  double t1,t2,c1,c2;

  ops_arg args[8] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7};



  #ifdef CHECKPOINTING
  if (!ops_checkpointing_before(args,8,range,101)) return;
  #endif

  if (OPS_diags > 1) {
    OPS_kernels[101].count++;
    ops_timers_core(&c2,&t2);
  }

  //compute locally allocated range for the sub-block
  int start[3];
  int end[3];

  for ( int n=0; n<3; n++ ){
    start[n] = range[2*n];end[n] = range[2*n+1];
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, "calc_dt_kernel_print");
  #endif



  //set up initial pointers and exchange halos if necessary
  int base0 = args[0].dat->base_offset;
  const double * __restrict__ xvel0 = (double *)(args[0].data + base0);

  int base1 = args[1].dat->base_offset;
  const double * __restrict__ yvel0 = (double *)(args[1].data + base1);

  int base2 = args[2].dat->base_offset;
  const double * __restrict__ zvel0 = (double *)(args[2].data + base2);

  int base3 = args[3].dat->base_offset;
  const double * __restrict__ density0 = (double *)(args[3].data + base3);

  int base4 = args[4].dat->base_offset;
  const double * __restrict__ energy0 = (double *)(args[4].data + base4);

  int base5 = args[5].dat->base_offset;
  const double * __restrict__ pressure = (double *)(args[5].data + base5);

  int base6 = args[6].dat->base_offset;
  const double * __restrict__ soundspeed = (double *)(args[6].data + base6);

  #ifdef OPS_MPI
  double * __restrict__ p_a7 = (double *)(((ops_reduction)args[7].data)->data + ((ops_reduction)args[7].data)->size * block->index);
  #else //OPS_MPI
  double * __restrict__ p_a7 = (double *)((ops_reduction)args[7].data)->data;
  #endif //OPS_MPI



  //initialize global variable with the dimension of dats
  int xdim0_calc_dt_kernel_print = args[0].dat->size[0];
  int ydim0_calc_dt_kernel_print = args[0].dat->size[1];
  int xdim1_calc_dt_kernel_print = args[1].dat->size[0];
  int ydim1_calc_dt_kernel_print = args[1].dat->size[1];
  int xdim2_calc_dt_kernel_print = args[2].dat->size[0];
  int ydim2_calc_dt_kernel_print = args[2].dat->size[1];
  int xdim3_calc_dt_kernel_print = args[3].dat->size[0];
  int ydim3_calc_dt_kernel_print = args[3].dat->size[1];
  int xdim4_calc_dt_kernel_print = args[4].dat->size[0];
  int ydim4_calc_dt_kernel_print = args[4].dat->size[1];
  int xdim5_calc_dt_kernel_print = args[5].dat->size[0];
  int ydim5_calc_dt_kernel_print = args[5].dat->size[1];
  int xdim6_calc_dt_kernel_print = args[6].dat->size[0];
  int ydim6_calc_dt_kernel_print = args[6].dat->size[1];

  if (OPS_diags > 1) {
    ops_timers_core(&c1,&t1);
    OPS_kernels[101].mpi_time += t1-t2;
  }

  double p_a7_0 = p_a7[0];
  double p_a7_1 = p_a7[1];
  double p_a7_2 = p_a7[2];
  double p_a7_3 = p_a7[3];
  double p_a7_4 = p_a7[4];
  double p_a7_5 = p_a7[5];
  double p_a7_6 = p_a7[6];
  double p_a7_7 = p_a7[7];
  double p_a7_8 = p_a7[8];
  double p_a7_9 = p_a7[9];
  double p_a7_10 = p_a7[10];
  double p_a7_11 = p_a7[11];
  double p_a7_12 = p_a7[12];
  double p_a7_13 = p_a7[13];
  double p_a7_14 = p_a7[14];
  double p_a7_15 = p_a7[15];
  double p_a7_16 = p_a7[16];
  double p_a7_17 = p_a7[17];
  double p_a7_18 = p_a7[18];
  double p_a7_19 = p_a7[19];
  double p_a7_20 = p_a7[20];
  double p_a7_21 = p_a7[21];
  double p_a7_22 = p_a7[22];
  double p_a7_23 = p_a7[23];
  double p_a7_24 = p_a7[24];
  double p_a7_25 = p_a7[25];
  double p_a7_26 = p_a7[26];
  double p_a7_27 = p_a7[27];
  #pragma omp parallel for reduction(+:p_a7_0) reduction(+:p_a7_1) reduction(+:p_a7_2) reduction(+:p_a7_3) reduction(+:p_a7_4) reduction(+:p_a7_5) reduction(+:p_a7_6) reduction(+:p_a7_7) reduction(+:p_a7_8) reduction(+:p_a7_9) reduction(+:p_a7_10) reduction(+:p_a7_11) reduction(+:p_a7_12) reduction(+:p_a7_13) reduction(+:p_a7_14) reduction(+:p_a7_15) reduction(+:p_a7_16) reduction(+:p_a7_17) reduction(+:p_a7_18) reduction(+:p_a7_19) reduction(+:p_a7_20) reduction(+:p_a7_21) reduction(+:p_a7_22) reduction(+:p_a7_23) reduction(+:p_a7_24) reduction(+:p_a7_25) reduction(+:p_a7_26) reduction(+:p_a7_27)
  for ( int n_z=start[2]; n_z<end[2]; n_z++ ){
    for ( int n_y=start[1]; n_y<end[1]; n_y++ ){
      #ifdef intel
      #pragma loop_count(10000)
      #pragma omp simd reduction(+:p_a7_0) reduction(+:p_a7_1) reduction(+:p_a7_2) reduction(+:p_a7_3) reduction(+:p_a7_4) reduction(+:p_a7_5) reduction(+:p_a7_6) reduction(+:p_a7_7) reduction(+:p_a7_8) reduction(+:p_a7_9) reduction(+:p_a7_10) reduction(+:p_a7_11) reduction(+:p_a7_12) reduction(+:p_a7_13) reduction(+:p_a7_14) reduction(+:p_a7_15) reduction(+:p_a7_16) reduction(+:p_a7_17) reduction(+:p_a7_18) reduction(+:p_a7_19) reduction(+:p_a7_20) reduction(+:p_a7_21) reduction(+:p_a7_22) reduction(+:p_a7_23) reduction(+:p_a7_24) reduction(+:p_a7_25) reduction(+:p_a7_26) reduction(+:p_a7_27) aligned(xvel0,yvel0,zvel0,density0,energy0,pressure,soundspeed)
      #else
      #pragma simd reduction(+:p_a7_0) reduction(+:p_a7_1) reduction(+:p_a7_2) reduction(+:p_a7_3) reduction(+:p_a7_4) reduction(+:p_a7_5) reduction(+:p_a7_6) reduction(+:p_a7_7) reduction(+:p_a7_8) reduction(+:p_a7_9) reduction(+:p_a7_10) reduction(+:p_a7_11) reduction(+:p_a7_12) reduction(+:p_a7_13) reduction(+:p_a7_14) reduction(+:p_a7_15) reduction(+:p_a7_16) reduction(+:p_a7_17) reduction(+:p_a7_18) reduction(+:p_a7_19) reduction(+:p_a7_20) reduction(+:p_a7_21) reduction(+:p_a7_22) reduction(+:p_a7_23) reduction(+:p_a7_24) reduction(+:p_a7_25) reduction(+:p_a7_26) reduction(+:p_a7_27)
      #endif
      for ( int n_x=start[0]; n_x<end[0]; n_x++ ){
        double output[28];
        output[0] = ZERO_double;
        output[1] = ZERO_double;
        output[2] = ZERO_double;
        output[3] = ZERO_double;
        output[4] = ZERO_double;
        output[5] = ZERO_double;
        output[6] = ZERO_double;
        output[7] = ZERO_double;
        output[8] = ZERO_double;
        output[9] = ZERO_double;
        output[10] = ZERO_double;
        output[11] = ZERO_double;
        output[12] = ZERO_double;
        output[13] = ZERO_double;
        output[14] = ZERO_double;
        output[15] = ZERO_double;
        output[16] = ZERO_double;
        output[17] = ZERO_double;
        output[18] = ZERO_double;
        output[19] = ZERO_double;
        output[20] = ZERO_double;
        output[21] = ZERO_double;
        output[22] = ZERO_double;
        output[23] = ZERO_double;
        output[24] = ZERO_double;
        output[25] = ZERO_double;
        output[26] = ZERO_double;
        output[27] = ZERO_double;
        
  output[0] = xvel0[OPS_ACC0(0,0,0)];
  output[1] = yvel0[OPS_ACC1(0,0,0)];
  output[2] = zvel0[OPS_ACC2(0,0,0)];
  output[3] = xvel0[OPS_ACC0(1,0,0)];
  output[4] = yvel0[OPS_ACC1(1,0,0)];
  output[5] = zvel0[OPS_ACC2(0,0,0)];
  output[6] = xvel0[OPS_ACC0(1,1,0)];
  output[7] = yvel0[OPS_ACC1(1,1,0)];
  output[8] = zvel0[OPS_ACC2(0,0,0)];
  output[9] = xvel0[OPS_ACC0(0,1,0)];
  output[10] = yvel0[OPS_ACC1(0,1,0)];
  output[11] = zvel0[OPS_ACC2(0,0,0)];
  output[12] = xvel0[OPS_ACC0(0,0,1)];
  output[13] = yvel0[OPS_ACC1(0,0,1)];
  output[14] = zvel0[OPS_ACC2(0,0,1)];
  output[15] = xvel0[OPS_ACC0(1,0,1)];
  output[16] = yvel0[OPS_ACC1(1,0,1)];
  output[17] = zvel0[OPS_ACC2(0,0,1)];
  output[18] = xvel0[OPS_ACC0(1,1,1)];
  output[19] = yvel0[OPS_ACC1(1,1,1)];
  output[20] = zvel0[OPS_ACC2(0,0,1)];
  output[21] = xvel0[OPS_ACC0(0,1,1)];
  output[22] = yvel0[OPS_ACC1(0,1,1)];
  output[23] = zvel0[OPS_ACC2(0,0,1)];
  output[24] = density0[OPS_ACC3(0,0,0)];
  output[25] = energy0[OPS_ACC4(0,0,0)];
  output[26] = pressure[OPS_ACC5(0,0,0)];
  output[27] = soundspeed[OPS_ACC6(0,0,0)];


        p_a7_0 +=output[0];
        p_a7_1 +=output[1];
        p_a7_2 +=output[2];
        p_a7_3 +=output[3];
        p_a7_4 +=output[4];
        p_a7_5 +=output[5];
        p_a7_6 +=output[6];
        p_a7_7 +=output[7];
        p_a7_8 +=output[8];
        p_a7_9 +=output[9];
        p_a7_10 +=output[10];
        p_a7_11 +=output[11];
        p_a7_12 +=output[12];
        p_a7_13 +=output[13];
        p_a7_14 +=output[14];
        p_a7_15 +=output[15];
        p_a7_16 +=output[16];
        p_a7_17 +=output[17];
        p_a7_18 +=output[18];
        p_a7_19 +=output[19];
        p_a7_20 +=output[20];
        p_a7_21 +=output[21];
        p_a7_22 +=output[22];
        p_a7_23 +=output[23];
        p_a7_24 +=output[24];
        p_a7_25 +=output[25];
        p_a7_26 +=output[26];
        p_a7_27 +=output[27];
      }
    }
  }
  p_a7[0] = p_a7_0;
  p_a7[1] = p_a7_1;
  p_a7[2] = p_a7_2;
  p_a7[3] = p_a7_3;
  p_a7[4] = p_a7_4;
  p_a7[5] = p_a7_5;
  p_a7[6] = p_a7_6;
  p_a7[7] = p_a7_7;
  p_a7[8] = p_a7_8;
  p_a7[9] = p_a7_9;
  p_a7[10] = p_a7_10;
  p_a7[11] = p_a7_11;
  p_a7[12] = p_a7_12;
  p_a7[13] = p_a7_13;
  p_a7[14] = p_a7_14;
  p_a7[15] = p_a7_15;
  p_a7[16] = p_a7_16;
  p_a7[17] = p_a7_17;
  p_a7[18] = p_a7_18;
  p_a7[19] = p_a7_19;
  p_a7[20] = p_a7_20;
  p_a7[21] = p_a7_21;
  p_a7[22] = p_a7_22;
  p_a7[23] = p_a7_23;
  p_a7[24] = p_a7_24;
  p_a7[25] = p_a7_25;
  p_a7[26] = p_a7_26;
  p_a7[27] = p_a7_27;
  if (OPS_diags > 1) {
    ops_timers_core(&c2,&t2);
    OPS_kernels[101].time += t2-t1;
  }

  if (OPS_diags > 1) {
    //Update kernel record
    ops_timers_core(&c1,&t1);
    OPS_kernels[101].mpi_time += t1-t2;
    OPS_kernels[101].transfer += ops_compute_transfer(dim, start, end, &arg0);
    OPS_kernels[101].transfer += ops_compute_transfer(dim, start, end, &arg1);
    OPS_kernels[101].transfer += ops_compute_transfer(dim, start, end, &arg2);
    OPS_kernels[101].transfer += ops_compute_transfer(dim, start, end, &arg3);
    OPS_kernels[101].transfer += ops_compute_transfer(dim, start, end, &arg4);
    OPS_kernels[101].transfer += ops_compute_transfer(dim, start, end, &arg5);
    OPS_kernels[101].transfer += ops_compute_transfer(dim, start, end, &arg6);
  }
}
#undef OPS_ACC0
#undef OPS_ACC1
#undef OPS_ACC2
#undef OPS_ACC3
#undef OPS_ACC4
#undef OPS_ACC5
#undef OPS_ACC6


void ops_par_loop_calc_dt_kernel_print(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7) {
  ops_kernel_descriptor *desc = (ops_kernel_descriptor *)malloc(sizeof(ops_kernel_descriptor));
  desc->name = name;
  desc->block = block;
  desc->dim = dim;
  desc->device = 1;
  desc->index = 101;
  desc->hash = 5381;
  desc->hash = ((desc->hash << 5) + desc->hash) + 101;
  for ( int i=0; i<6; i++ ){
    desc->range[i] = range[i];
    desc->orig_range[i] = range[i];
    desc->hash = ((desc->hash << 5) + desc->hash) + range[i];
  }
  desc->nargs = 8;
  desc->args = (ops_arg*)malloc(8*sizeof(ops_arg));
  desc->args[0] = arg0;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg0.dat->index;
  desc->args[1] = arg1;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg1.dat->index;
  desc->args[2] = arg2;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg2.dat->index;
  desc->args[3] = arg3;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg3.dat->index;
  desc->args[4] = arg4;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg4.dat->index;
  desc->args[5] = arg5;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg5.dat->index;
  desc->args[6] = arg6;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg6.dat->index;
  desc->args[7] = arg7;
  desc->function = ops_par_loop_calc_dt_kernel_print_execute;
  if (OPS_diags > 1) {
    ops_timing_realloc(101,"calc_dt_kernel_print");
  }
  ops_enqueue_kernel(desc);
  }
