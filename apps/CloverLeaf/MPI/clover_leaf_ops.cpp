//
// auto-generated by ops.py on 2014-03-17 11:57
//




#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include  "ops_lib_cpp.h"




#include "data.h"

#include "definitions.h"


void initialise();
void field_summary();
void timestep();
void PdV(int predict);
void accelerate();
void flux_calc();
void advection(int);
void reset_field();




float   g_version = 1.0;
int     g_ibig = 640000;
double  g_small = 1.0e-16;
double  g_big  = 1.0e+21;
int     g_name_len_max = 255 ,
        g_xdir = 1,
        g_ydir = 2;

int     number_of_states;

int     CHUNK_LEFT    = 1,
        CHUNK_RIGHT   = 2,
        CHUNK_BOTTOM  = 3,
        CHUNK_TOP     = 4,
        EXTERNAL_FACE = -1;

int     FIELD_DENSITY0   = 0,
        FIELD_DENSITY1   = 1,
        FIELD_ENERGY0    = 2,
        FIELD_ENERGY1    = 3,
        FIELD_PRESSURE   = 4,
        FIELD_VISCOSITY  = 5,
        FIELD_SOUNDSPEED = 6,
        FIELD_XVEL0      = 7,
        FIELD_XVEL1      = 8,
        FIELD_YVEL0      = 9,
        FIELD_YVEL1      =10,
        FIELD_VOL_FLUX_X =11,
        FIELD_VOL_FLUX_Y =12,
        FIELD_MASS_FLUX_X=13,
        FIELD_MASS_FLUX_Y=14,
        NUM_FIELDS       =15;

FILE    *g_out, *g_in;

int     g_rect=1,
        g_circ=2,
        g_point=3;

state_type * states;

grid_type grid;

field_type field;

int step ;
int advect_x;
int error_condition;
int test_problem;
int state_max;
int complete;

int fields[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

double dtold, dt, clover_time, dtinit, dtmin, dtmax, dtrise, dtu_safe, dtv_safe, dtc_safe,
       dtdiv_safe, dtc, dtu, dtv, dtdiv;

int x_min, y_min, x_max, y_max, x_cells, y_cells;

double end_time;
int end_step;
int visit_frequency;
int summary_frequency;
int use_vector_loops;

int jdt, kdt;

#include "cloverleaf_ops_vars.h"


int main(int argc, char **argv)
{


  ops_init(argc,argv,5);
  ops_printf(" Clover version %f\n", g_version);



  initialise();


  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  ops_decl_const2( "g_small",1,"double",&g_small);
  ops_decl_const2( "g_big",1,"double",&g_big);
  ops_decl_const2( "dtc_safe",1,"double",&dtc_safe);
  ops_decl_const2( "dtu_safe",1,"double",&dtu_safe);
  ops_decl_const2( "dtv_safe",1,"double",&dtv_safe);
  ops_decl_const2( "dtdiv_safe",1,"double",&dtdiv_safe);
  ops_decl_const2( "x_max",1,"int",&x_max);
  ops_decl_const2( "y_max",1,"int",&y_max);

  double ct0, ct1, et0, et1;
  ops_timers_core(&ct0, &et0);

  while(1) {

    step = step + 1;

    timestep();


      ops_decl_const2( "dt",1,"double",&dt);

    PdV(TRUE);

    accelerate();

    PdV(FALSE);

    flux_calc();

    advection(step);

    reset_field();

    if (advect_x == TRUE) advect_x = FALSE;
    else advect_x = TRUE;

    clover_time = clover_time + dt;

    if(summary_frequency != 0)
      if((step%summary_frequency) == 0)
        field_summary();

    if((clover_time+g_small) > end_time || (step >= end_step)) {
      complete=TRUE;
      field_summary();
      ops_fprintf(g_out,"\n\n Calculation complete\n");
      ops_fprintf(g_out,"\n Clover is finishing\n");
      break;
    }

     if(step == 70) {




     }

  }

  ops_timers_core(&ct1, &et1);
  ops_timing_output();

  ops_printf("\nTotal Wall time %lf\n",et1-et0);
  ops_fprintf(g_out,"\nTotal Wall time %lf\n",et1-et0);

  fclose(g_out);
  ops_exit();
}
