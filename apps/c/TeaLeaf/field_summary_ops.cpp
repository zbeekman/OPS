//
// auto-generated by ops.py
//


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

 #define OPS_2D
#include  "ops_lib_cpp.h"

//
// ops_par_loop declarations
//

void ops_par_loop_field_summary_kernel(char const *, ops_block, int , int*,
  ops_arg,
  ops_arg,
  ops_arg,
  ops_arg,
  ops_arg,
  ops_arg,
  ops_arg,
  ops_arg );



#include "data.h"
#include "definitions.h"

//#include "field_summary_kernel.h"

void field_summary()
{
  double qa_diff;

  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max};

  double vol= 0.0 , mass = 0.0, ie = 0.0, temp = 0.0;

  ops_par_loop_field_summary_kernel("field_summary_kernel", tea_grid, 2, rangexy_inner,
               ops_arg_dat(volume, 1, S2D_00, "double", OPS_READ),
               ops_arg_dat(density, 1, S2D_00, "double", OPS_READ),
               ops_arg_dat(energy1, 1, S2D_00, "double", OPS_READ),
               ops_arg_dat(u, 1, S2D_00, "double", OPS_READ),
               ops_arg_reduce(red_vol, 1, "double", OPS_INC),
               ops_arg_reduce(red_mass, 1, "double", OPS_INC),
               ops_arg_reduce(red_ie, 1, "double", OPS_INC),
               ops_arg_reduce(red_temp, 1, "double", OPS_INC));

  ops_reduction_result(red_vol,&vol);
  ops_reduction_result(red_mass,&mass);
  ops_reduction_result(red_ie,&ie);
  ops_reduction_result(red_temp,&temp);

  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out,"\n Time %lf\n",clover_time);
  ops_fprintf(g_out,"              %-10s  %-10s  %-15s  %-10s  %-s\n",
  " Volume"," Mass"," Density"," Internal Energy","Temperature");
  ops_fprintf(g_out," step:   %3d   %-10.3E  %-10.3E  %-15.3E  %-10.3E  %-.3E",
          step, vol, mass, mass/vol, ie, temp);

  if(complete == 1) {
    if(test_problem>0) {
      if (test_problem == 1)
        qa_diff = fabs((100.0 * (temp / 157.55084183279294)) - 100.0);
      if (test_problem == 2)
        qa_diff = fabs((100.0 * (temp / 106.27221178646569)) - 100.0);
      if (test_problem == 3)
        qa_diff = fabs((100.0 * (temp / 99.955877498324000)) - 100.0);
      if (test_problem == 4)
        qa_diff = fabs((100.0 * (temp / 97.277332050749976)) - 100.0);
      if (test_problem == 5)
        qa_diff = fabs((100.0 * (temp / 95.462351583362249)) - 100.0);
      ops_printf("Test problem %3d is within   %-10.7E%% of the expected solution\n",test_problem, qa_diff);
      ops_fprintf(g_out,"\nTest problem %3d is within   %10.7E%% of the expected solution\n",test_problem, qa_diff);
      if(qa_diff < 0.001) {
        ops_printf(" This test is considered PASSED\n");
        ops_fprintf(g_out," This test is considered PASSED\n");
      }
      else
      {
        ops_printf(" This test is considered FAILED\n");
        ops_fprintf(g_out," This test is considered FAILED\n");
      }
    }
  }
  fflush(g_out);



}
