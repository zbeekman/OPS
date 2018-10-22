//
// auto-generated by ops.py
//
#include "./MPI_inline/clover_leaf_common.h"

int xdim0_advec_mom_kernel_mass_flux_y;
int xdim1_advec_mom_kernel_mass_flux_y;

#define OPS_ACC0(x, y)                                                         \
  (n_x * 1 + x + (n_y * 1 + (y)) * xdim0_advec_mom_kernel_mass_flux_y)
#define OPS_ACC1(x, y)                                                         \
  (n_x * 1 + x + (n_y * 1 + (y)) * xdim1_advec_mom_kernel_mass_flux_y)
// user function

void advec_mom_kernel_mass_flux_y_c_wrapper(double *restrict node_flux,
                                            const double *restrict mass_flux_y,
                                            int x_size, int y_size) {
#pragma omp parallel for
  for (int n_y = 0; n_y < y_size; n_y++) {
    for (int n_x = 0; n_x < x_size; n_x++) {

      node_flux[OPS_ACC0(0, 0)] =
          0.25 * (mass_flux_y[OPS_ACC1(-1, 0)] + mass_flux_y[OPS_ACC1(0, 0)] +
                  mass_flux_y[OPS_ACC1(-1, 1)] + mass_flux_y[OPS_ACC1(0, 1)]);
    }
  }
}
#undef OPS_ACC0
#undef OPS_ACC1
