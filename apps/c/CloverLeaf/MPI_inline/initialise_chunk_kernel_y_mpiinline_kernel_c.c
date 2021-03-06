//
// auto-generated by ops.py
//

int xdim0_initialise_chunk_kernel_y;
int xdim1_initialise_chunk_kernel_y;
int xdim2_initialise_chunk_kernel_y;


//user function



void initialise_chunk_kernel_y_c_wrapper(
  double * restrict vertexy_p,
  int * restrict yy_p,
  double * restrict vertexdy_p,
  int x_size, int y_size) {
  #pragma omp parallel for
  for ( int n_y=0; n_y<y_size; n_y++ ){
    for ( int n_x=0; n_x<x_size; n_x++ ){
      ptr_double vertexy = { vertexy_p + n_x*0 + n_y * xdim0_initialise_chunk_kernel_y*1, xdim0_initialise_chunk_kernel_y};
      const ptr_int yy = { yy_p + n_x*0 + n_y * xdim1_initialise_chunk_kernel_y*1, xdim1_initialise_chunk_kernel_y};
      ptr_double vertexdy = { vertexdy_p + n_x*0 + n_y * xdim2_initialise_chunk_kernel_y*1, xdim2_initialise_chunk_kernel_y};

      int y_min = field.y_min - 2;
      double min_y, d_y;

      d_y = (grid.ymax - grid.ymin) / (double)grid.y_cells;
      min_y = grid.ymin + d_y * field.bottom;

      OPS_ACC(vertexy, 0, 0) = min_y + d_y * (OPS_ACC(yy, 0, 0) - y_min);
      OPS_ACC(vertexdy, 0, 0) = (double)d_y;
    }
  }
}
