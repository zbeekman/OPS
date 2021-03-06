//
// auto-generated by ops.py
//

int xdim0_update_halo_kernel2_yvel_plus_2_b;
int xdim1_update_halo_kernel2_yvel_plus_2_b;


//user function



void update_halo_kernel2_yvel_plus_2_b_c_wrapper(
  double * restrict yvel0_p,
  double * restrict yvel1_p,
  const int * restrict fields,
  int x_size, int y_size) {
  #pragma omp parallel for
  for ( int n_y=0; n_y<y_size; n_y++ ){
    for ( int n_x=0; n_x<x_size; n_x++ ){
      ptr_double yvel0 = { yvel0_p + n_x*1 + n_y * xdim0_update_halo_kernel2_yvel_plus_2_b*1, xdim0_update_halo_kernel2_yvel_plus_2_b};
      ptr_double yvel1 = { yvel1_p + n_x*1 + n_y * xdim1_update_halo_kernel2_yvel_plus_2_b*1, xdim1_update_halo_kernel2_yvel_plus_2_b};

      if (fields[FIELD_YVEL0] == 1)
        OPS_ACC(yvel0, 0, 0) = OPS_ACC(yvel0, -2, 0);
      if (fields[FIELD_YVEL1] == 1)
        OPS_ACC(yvel1, 0, 0) = OPS_ACC(yvel1, -2, 0);
    }
  }
}
