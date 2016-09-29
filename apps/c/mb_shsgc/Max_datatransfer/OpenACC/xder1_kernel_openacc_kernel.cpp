//
// auto-generated by ops.py
//
#include "./OpenACC/shsgc_common.h"

#undef OPS_GPU

extern int xdim0_xder1_kernel;
int xdim0_xder1_kernel_h = -1;
extern int xdim1_xder1_kernel;
int xdim1_xder1_kernel_h = -1;

#ifdef __cplusplus
extern "C" {
#endif
void xder1_kernel_c_wrapper(double *p_a0, double *p_a1, int x_size);

#ifdef __cplusplus
}
#endif

// host stub function
void ops_par_loop_xder1_kernel(char const *name, ops_block block, int dim,
                               int *range, ops_arg arg0, ops_arg arg1) {

  // Timing
  double t1, t2, c1, c2;
  ops_arg args[2] = {arg0, arg1};

#ifdef CHECKPOINTING
  if (!ops_checkpointing_before(args, 2, range, 4))
    return;
#endif

  if (OPS_diags > 1) {
    ops_timing_realloc(4, "xder1_kernel");
    OPS_kernels[4].count++;
    ops_timers_core(&c1, &t1);
  }

  // compute localy allocated range for the sub-block

  int start[1];
  int end[1];
#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
  if (!sb->owned)
    return;
  for (int n = 0; n < 1; n++) {
    start[n] = sb->decomp_disp[n];
    end[n] = sb->decomp_disp[n] + sb->decomp_size[n];
    if (start[n] >= range[2 * n]) {
      start[n] = 0;
    } else {
      start[n] = range[2 * n] - start[n];
    }
    if (sb->id_m[n] == MPI_PROC_NULL && range[2 * n] < 0)
      start[n] = range[2 * n];
    if (end[n] >= range[2 * n + 1]) {
      end[n] = range[2 * n + 1] - sb->decomp_disp[n];
    } else {
      end[n] = sb->decomp_size[n];
    }
    if (sb->id_p[n] == MPI_PROC_NULL &&
        (range[2 * n + 1] > sb->decomp_disp[n] + sb->decomp_size[n]))
      end[n] += (range[2 * n + 1] - sb->decomp_disp[n] - sb->decomp_size[n]);
  }
#else
  for (int n = 0; n < 1; n++) {
    start[n] = range[2 * n];
    end[n] = range[2 * n + 1];
  }
#endif

  int x_size = MAX(0, end[0] - start[0]);

  xdim0 = args[0].dat->size[0];
  xdim1 = args[1].dat->size[0];
  if (xdim0 != xdim0_xder1_kernel_h || xdim1 != xdim1_xder1_kernel_h) {
    xdim0_xder1_kernel = xdim0;
    xdim0_xder1_kernel_h = xdim0;
    xdim1_xder1_kernel = xdim1;
    xdim1_xder1_kernel_h = xdim1;
  }

  // set up initial pointers
  int base0 = args[0].dat->base_offset +
              args[0].dat->elem_size * start[0] * args[0].stencil->stride[0];
#ifdef OPS_GPU
  double *p_a0 = (double *)((char *)args[0].data_d + base0);
#else
  double *p_a0 = (double *)((char *)args[0].data + base0);
#endif

  int base1 = args[1].dat->base_offset +
              args[1].dat->elem_size * start[0] * args[1].stencil->stride[0];
#ifdef OPS_GPU
  double *p_a1 = (double *)((char *)args[1].data_d + base1);
#else
  double *p_a1 = (double *)((char *)args[1].data + base1);
#endif

#ifdef OPS_GPU
  ops_H_D_exchanges_device(args, 2);
#else
  ops_H_D_exchanges_host(args, 2);
#endif
  ops_halo_exchanges(args, 2, range);

#ifdef OPS_GPU
  ops_H_D_exchanges_device(args, 2);
#else
  ops_H_D_exchanges_host(args, 2);
#endif
  if (OPS_diags > 1) {
    ops_timers_core(&c2, &t2);
    OPS_kernels[4].mpi_time += t2 - t1;
  }

  xder1_kernel_c_wrapper(p_a0, p_a1, x_size);

  if (OPS_diags > 1) {
    ops_timers_core(&c1, &t1);
    OPS_kernels[4].time += t1 - t2;
  }
#ifdef OPS_GPU
  ops_set_dirtybit_device(args, 2);
#else
  ops_set_dirtybit_host(args, 2);
#endif
  ops_set_halo_dirtybit3(&args[1], range);

  if (OPS_diags > 1) {
    // Update kernel record
    ops_timers_core(&c2, &t2);
    OPS_kernels[4].mpi_time += t2 - t1;
    OPS_kernels[4].transfer += ops_compute_transfer(dim, start, end, &arg0);
    OPS_kernels[4].transfer += ops_compute_transfer(dim, start, end, &arg1);
  }
}
