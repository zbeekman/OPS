#include <stdlib.h>
#include <string.h>
#include <math.h>
// Global constants in the equations are
int nx2;
int nx1;
double deltai2;
double rinv8;
double rinv9;
double Minf;
double rinv1;
double rinv4;
double rinv5;
double rinv6;
double Pr;
double rinv12;
double rinv13;
double deltat;
double rinv14;
double rknew[3];
double rinv15;
double rc0;
double rc2;
double rc3;
int nx0;
double deltai1;
double rc10;
double deltai0;
double Re;
double rinv7;
double gama;
double rkold[3];
double rc11;
// OPS header file
#define OPS_3D
#include "ops_seq.h"
#include "taylor_green_vortex_block_0_kernel.h"

// main program start
int main (int argc, char **argv) 
{

   gama = 1.40000000000000;
   Pr = 0.710000000000000;
   nx2 = atoi(argv[3]);
   nx0 = atoi(argv[1]);
   Re = 1600;
   deltat = 0.000846250000000000;
   nx1 = atoi(argv[2]);
   Minf = 0.100000000000000;
   rc0 = 1.0/2.0;
   rc2 = 1.0/12.0;
   rc3 = 2.0/3.0;
   rc10 = 5.0/2.0;
   rc11 = 4.0/3.0;
   rkold[0] = 1.0/4.0;
   rkold[1] = 3.0/20.0;
   rkold[2] = 3.0/5.0;
   rknew[0] = 2.0/3.0;
   rknew[1] = 5.0/12.0;
   rknew[2] = 3.0/5.0;
   rinv13 = 1.0/Pr;
   rinv14 = pow(Minf, -2);
   rinv15 = 1.0/(gama*pow(Minf, 2));
   rinv12 = 1.0/(gama - 1);
   deltai2 = (1.0/256.0)*M_PI;
   rinv8 = 1.0/Re;
   deltai1 = (1.0/256.0)*M_PI;
   deltai0 = (1.0/256.0)*M_PI;
   rinv9 = pow(deltai2, -2);
   rinv1 = 1.0/deltai2;
   rinv4 = 1.0/deltai1;
   rinv5 = 1.0/deltai0;
   rinv6 = pow(deltai1, -2);
   rinv7 = pow(deltai0, -2);

   // Initializing OPS 
   ops_init(argc,argv,1);

   ops_decl_const("nx2" , 1, "int", &nx2);
   ops_decl_const("nx1" , 1, "int", &nx1);
   ops_decl_const("deltai2" , 1, "double", &deltai2);
   ops_decl_const("rinv8" , 1, "double", &rinv8);
   ops_decl_const("rinv9" , 1, "double", &rinv9);
   ops_decl_const("Minf" , 1, "double", &Minf);
   ops_decl_const("rinv1" , 1, "double", &rinv1);
   ops_decl_const("rinv4" , 1, "double", &rinv4);
   ops_decl_const("rinv5" , 1, "double", &rinv5);
   ops_decl_const("rinv6" , 1, "double", &rinv6);
   ops_decl_const("Pr" , 1, "double", &Pr);
   ops_decl_const("rinv12" , 1, "double", &rinv12);
   ops_decl_const("rinv13" , 1, "double", &rinv13);
   ops_decl_const("deltat" , 1, "double", &deltat);
   ops_decl_const("rinv14" , 1, "double", &rinv14);
   ops_decl_const("rinv15" , 1, "double", &rinv15);
   ops_decl_const("rc0" , 1, "double", &rc0);
   ops_decl_const("rc2" , 1, "double", &rc2);
   ops_decl_const("rc3" , 1, "double", &rc3);
   ops_decl_const("nx0" , 1, "int", &nx0);
   ops_decl_const("deltai1" , 1, "double", &deltai1);
   ops_decl_const("rc10" , 1, "double", &rc10);
   ops_decl_const("deltai0" , 1, "double", &deltai0);
   ops_decl_const("Re" , 1, "double", &Re);
   ops_decl_const("rinv7" , 1, "double", &rinv7);
   ops_decl_const("gama" , 1, "double", &gama);
   ops_decl_const("rc11" , 1, "double", &rc11);

   // Defining block in OPS Format
   ops_block taylor_green_vortex_block;

   // Initialising block in OPS Format
   taylor_green_vortex_block = ops_decl_block(3, "taylor_green_vortex_block");

   // Define dataset
   ops_dat wk3;
   ops_dat wk7;
   ops_dat p;
   ops_dat wk1;
   ops_dat wk2;
   ops_dat wk6;
   ops_dat rho;
   ops_dat rhou2;
   ops_dat wk11;
   ops_dat u1;
   ops_dat wk0;
   ops_dat rho_old;
   ops_dat wk8;
   ops_dat u0;
   ops_dat rhoE_old;
   ops_dat wk12;
   ops_dat rhoE;
   ops_dat rhou1_old;
   ops_dat wk10;
   ops_dat wk5;
   ops_dat u2;
   ops_dat wk9;
   ops_dat rhou1;
   ops_dat wk4;
   ops_dat rhou0;
   ops_dat T;
   ops_dat rhou2_old;
   ops_dat wk13;
   ops_dat rhou0_old;

   // Initialise/allocate OPS dataset.
   int halo_p[] = {2, 2, 2};
   int halo_m[] = {-2, -2, -2};
   int size[] = {nx0, nx1, nx2};
   int base[] = {0, 0, 0};
   double* val = NULL;
   wk3 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk3");
   wk7 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk7");
   p = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "p");
   wk1 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk1");
   wk2 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk2");
   wk6 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk6");
   rho = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "rho");
   rhou2 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "rhou2");
   wk11 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk11");
   u1 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "u1");
   wk0 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk0");
   rho_old = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "rho_old");
   wk8 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk8");
   u0 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "u0");
   rhoE_old = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "rhoE_old");
   wk12 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk12");
   rhoE = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "rhoE");
   rhou1_old = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "rhou1_old");
   wk10 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk10");
   wk5 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk5");
   u2 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "u2");
   wk9 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk9");
   rhou1 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "rhou1");
   wk4 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk4");
   rhou0 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "rhou0");
   T = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "T");
   rhou2_old = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "rhou2_old");
   wk13 = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "wk13");
   rhou0_old = ops_decl_dat(taylor_green_vortex_block, 1, size, base, halo_m, halo_p, val, "double", "rhou0_old");

   // Declare all the stencils used 
   int stencil7_temp[] = {0,0,-2,0,0,-1,0,-2,0,0,-1,0,0,0,0,0,1,0,0,2,0,0,0,1,0,0,2};
   ops_stencil stencil7 = ops_decl_stencil(3,9,stencil7_temp,"0,0,-2,0,0,-1,0,-2,0,0,-1,0,0,0,0,0,1,0,0,2,0,0,0,1,0,0,2");
   int stencil2_temp[] = {0,-2,0,0,-1,0,0,1,0,0,2,0};
   ops_stencil stencil2 = ops_decl_stencil(3,4,stencil2_temp,"0,-2,0,0,-1,0,0,1,0,0,2,0");
   int stencil5_temp[] = {0,0,-2,0,0,-1,0,-2,0,0,-1,0,-2,0,0,-1,0,0,0,0,0,1,0,0,2,0,0,0,1,0,0,2,0,0,0,1,0,0,2};
   ops_stencil stencil5 = ops_decl_stencil(3,13,stencil5_temp,"0,0,-2,0,0,-1,0,-2,0,0,-1,0,-2,0,0,-1,0,0,0,0,0,1,0,0,2,0,0,0,1,0,0,2,0,0,0,1,0,0,2");
   int stencil6_temp[] = {0,0,-2,0,0,-1,0,-2,0,0,-1,0,-2,0,0,-1,0,0,1,0,0,2,0,0,0,1,0,0,2,0,0,0,1,0,0,2};
   ops_stencil stencil6 = ops_decl_stencil(3,12,stencil6_temp,"0,0,-2,0,0,-1,0,-2,0,0,-1,0,-2,0,0,-1,0,0,1,0,0,2,0,0,0,1,0,0,2,0,0,0,1,0,0,2");
   int stencil8_temp[] = {0,-2,0,0,-1,0,0,0,0,0,1,0,0,2,0};
   ops_stencil stencil8 = ops_decl_stencil(3,5,stencil8_temp,"0,-2,0,0,-1,0,0,0,0,0,1,0,0,2,0");
   int stencil4_temp[] = {0,0,-2,0,0,-1,0,0,0,0,0,1,0,0,2};
   ops_stencil stencil4 = ops_decl_stencil(3,5,stencil4_temp,"0,0,-2,0,0,-1,0,0,0,0,0,1,0,0,2");
   int stencil1_temp[] = {0,0,-2,0,0,-1,0,0,1,0,0,2};
   ops_stencil stencil1 = ops_decl_stencil(3,4,stencil1_temp,"0,0,-2,0,0,-1,0,0,1,0,0,2");
   int stencil0_temp[] = {0,0,0};
   ops_stencil stencil0 = ops_decl_stencil(3,1,stencil0_temp,"0,0,0");
   int stencil3_temp[] = {-2,0,0,-1,0,0,1,0,0,2,0,0};
   ops_stencil stencil3 = ops_decl_stencil(3,4,stencil3_temp,"-2,0,0,-1,0,0,1,0,0,2,0,0");





   // Init OPS partition
   ops_partition("");
  ops_diagnostic_output();

   int iter_range14[] = {-2, nx0 + 2, -2, nx1 + 2, -2, nx2 + 2};
   ops_par_loop(taylor_green_vortex_block0_14_kernel, "Initialisation", taylor_green_vortex_block, 3, iter_range14,
   ops_arg_dat(rhou1, 1, stencil0, "double", OPS_WRITE),
   ops_arg_dat(rhoE, 1, stencil0, "double", OPS_WRITE),
   ops_arg_dat(rho, 1, stencil0, "double", OPS_WRITE),
   ops_arg_dat(rhou2, 1, stencil0, "double", OPS_WRITE),
   ops_arg_dat(rhou0, 1, stencil0, "double", OPS_WRITE),
   ops_arg_idx());



   int iter_range15[] = {0, 1, -2, nx1 + 2, -2, nx2 + 2};
   ops_par_loop(taylor_green_vortex_block0_15_kernel, "Symmetry bc 0 Left", taylor_green_vortex_block, 3, iter_range15,
   ops_arg_dat(rhou1, 1, stencil3, "double", OPS_RW),
   ops_arg_dat(rhoE, 1, stencil3, "double", OPS_RW),
   ops_arg_dat(rho, 1, stencil3, "double", OPS_RW),
   ops_arg_dat(rhou2, 1, stencil3, "double", OPS_RW),
   ops_arg_dat(rhou0, 1, stencil3, "double", OPS_RW));


   int iter_range16[] = {nx0 - 1, nx0, -2, nx1 + 2, -2, nx2 + 2};
   ops_par_loop(taylor_green_vortex_block0_16_kernel, "Symmetry bc 0 Right", taylor_green_vortex_block, 3, iter_range16,
   ops_arg_dat(rhou1, 1, stencil3, "double", OPS_RW),
   ops_arg_dat(rhoE, 1, stencil3, "double", OPS_RW),
   ops_arg_dat(rho, 1, stencil3, "double", OPS_RW),
   ops_arg_dat(rhou2, 1, stencil3, "double", OPS_RW),
   ops_arg_dat(rhou0, 1, stencil3, "double", OPS_RW));


   int iter_range17[] = {-2, nx0 + 2, 0, 1, -2, nx2 + 2};
   ops_par_loop(taylor_green_vortex_block0_17_kernel, "Symmetry bc 1 Left", taylor_green_vortex_block, 3, iter_range17,
   ops_arg_dat(rhou1, 1, stencil2, "double", OPS_RW),
   ops_arg_dat(rhoE, 1, stencil2, "double", OPS_RW),
   ops_arg_dat(rho, 1, stencil2, "double", OPS_RW),
   ops_arg_dat(rhou2, 1, stencil2, "double", OPS_RW),
   ops_arg_dat(rhou0, 1, stencil2, "double", OPS_RW));


   int iter_range18[] = {-2, nx0 + 2, nx1 - 1, nx1, -2, nx2 + 2};
   ops_par_loop(taylor_green_vortex_block0_18_kernel, "Symmetry bc 1 Right", taylor_green_vortex_block, 3, iter_range18,
   ops_arg_dat(rhou1, 1, stencil2, "double", OPS_RW),
   ops_arg_dat(rhoE, 1, stencil2, "double", OPS_RW),
   ops_arg_dat(rho, 1, stencil2, "double", OPS_RW),
   ops_arg_dat(rhou2, 1, stencil2, "double", OPS_RW),
   ops_arg_dat(rhou0, 1, stencil2, "double", OPS_RW));


   int iter_range19[] = {-2, nx0 + 2, -2, nx1 + 2, 0, 1};
   ops_par_loop(taylor_green_vortex_block0_19_kernel, "Symmetry bc 2 Left", taylor_green_vortex_block, 3, iter_range19,
   ops_arg_dat(rhou1, 1, stencil1, "double", OPS_RW),
   ops_arg_dat(rhoE, 1, stencil1, "double", OPS_RW),
   ops_arg_dat(rho, 1, stencil1, "double", OPS_RW),
   ops_arg_dat(rhou2, 1, stencil1, "double", OPS_RW),
   ops_arg_dat(rhou0, 1, stencil1, "double", OPS_RW));


   int iter_range20[] = {-2, nx0 + 2, -2, nx1 + 2, nx2 - 1, nx2};
   ops_par_loop(taylor_green_vortex_block0_20_kernel, "Symmetry bc 2 Right", taylor_green_vortex_block, 3, iter_range20,
   ops_arg_dat(rhou1, 1, stencil1, "double", OPS_RW),
   ops_arg_dat(rhoE, 1, stencil1, "double", OPS_RW),
   ops_arg_dat(rho, 1, stencil1, "double", OPS_RW),
   ops_arg_dat(rhou2, 1, stencil1, "double", OPS_RW),
   ops_arg_dat(rhou0, 1, stencil1, "double", OPS_RW));



   double cpu_start, elapsed_start;
   ops_timers(&cpu_start, &elapsed_start);

   for (int iteration=0; iteration<30; iteration++){
      printf("Iteration %d\n", iteration);

      int iter_range13[] = {-2, nx0 + 2, -2, nx1 + 2, -2, nx2 + 2};
      ops_par_loop(taylor_green_vortex_block0_13_kernel, "Save equations", taylor_green_vortex_block, 3, iter_range13,
      ops_arg_dat(rhou1, 1, stencil0, "double", OPS_READ),
      ops_arg_dat(rhoE, 1, stencil0, "double", OPS_READ),
      ops_arg_dat(rho, 1, stencil0, "double", OPS_READ),
      ops_arg_dat(rhou2, 1, stencil0, "double", OPS_READ),
      ops_arg_dat(rhou0, 1, stencil0, "double", OPS_READ),
      ops_arg_dat(rhou1_old, 1, stencil0, "double", OPS_WRITE),
      ops_arg_dat(rhou2_old, 1, stencil0, "double", OPS_WRITE),
      ops_arg_dat(rhou0_old, 1, stencil0, "double", OPS_WRITE),
      ops_arg_dat(rho_old, 1, stencil0, "double", OPS_WRITE),
      ops_arg_dat(rhoE_old, 1, stencil0, "double", OPS_WRITE));



      for (int stage=0; stage<3; stage++){


         int iter_range0[] = {-2, nx0 + 2, -2, nx1 + 2, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_0_kernel, "Grouped Formula Evaluation", taylor_green_vortex_block, 3, iter_range0,
         ops_arg_dat(rhou1, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rho, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rhoE, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rhou2, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rhou0, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(u0, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(u1, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(u2, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(p, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(T, 1, stencil0, "double", OPS_WRITE));


         int iter_range1[] = {0, nx0, 0, nx1, 0, nx2};
         ops_par_loop(taylor_green_vortex_block0_1_kernel, "D(u0 x2)", taylor_green_vortex_block, 3, iter_range1,
         ops_arg_dat(u0, 1, stencil1, "double", OPS_READ),
         ops_arg_dat(wk0, 1, stencil0, "double", OPS_WRITE));


         int iter_range2[] = {0, nx0, 0, nx1, 0, nx2};
         ops_par_loop(taylor_green_vortex_block0_2_kernel, "D(u0 x1)", taylor_green_vortex_block, 3, iter_range2,
         ops_arg_dat(u0, 1, stencil2, "double", OPS_READ),
         ops_arg_dat(wk1, 1, stencil0, "double", OPS_WRITE));


         int iter_range3[] = {0, nx0, 0, nx1, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_3_kernel, "D(u2 x0)", taylor_green_vortex_block, 3, iter_range3,
         ops_arg_dat(u2, 1, stencil3, "double", OPS_READ),
         ops_arg_dat(wk2, 1, stencil0, "double", OPS_WRITE));


         int iter_range4[] = {0, nx0, 0, nx1, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_4_kernel, "D(u1[x0 x1 x2 t] x1)", taylor_green_vortex_block, 3, iter_range4,
         ops_arg_dat(u1, 1, stencil2, "double", OPS_READ),
         ops_arg_dat(wk3, 1, stencil0, "double", OPS_WRITE));


         int iter_range5[] = {0, nx0, -2, nx1 + 2, 0, nx2};
         ops_par_loop(taylor_green_vortex_block0_5_kernel, "D(u1[x0 x1 x2 t] x0)", taylor_green_vortex_block, 3, iter_range5,
         ops_arg_dat(u1, 1, stencil3, "double", OPS_READ),
         ops_arg_dat(wk4, 1, stencil0, "double", OPS_WRITE));


         int iter_range6[] = {0, nx0, 0, nx1, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_6_kernel, "D(u2 x1)", taylor_green_vortex_block, 3, iter_range6,
         ops_arg_dat(u2, 1, stencil2, "double", OPS_READ),
         ops_arg_dat(wk5, 1, stencil0, "double", OPS_WRITE));


         int iter_range7[] = {0, nx0, -2, nx1 + 2, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_7_kernel, "D(u0 x0)", taylor_green_vortex_block, 3, iter_range7,
         ops_arg_dat(u0, 1, stencil3, "double", OPS_READ),
         ops_arg_dat(wk6, 1, stencil0, "double", OPS_WRITE));


         int iter_range8[] = {0, nx0, 0, nx1, 0, nx2};
         ops_par_loop(taylor_green_vortex_block0_8_kernel, "D(u2 x2)", taylor_green_vortex_block, 3, iter_range8,
         ops_arg_dat(u2, 1, stencil1, "double", OPS_READ),
         ops_arg_dat(wk7, 1, stencil0, "double", OPS_WRITE));


         int iter_range9[] = {0, nx0, 0, nx1, 0, nx2};
         ops_par_loop(taylor_green_vortex_block0_9_kernel, "D(u1[x0 x1 x2 t] x2)", taylor_green_vortex_block, 3, iter_range9,
         ops_arg_dat(u1, 1, stencil1, "double", OPS_READ),
         ops_arg_dat(wk8, 1, stencil0, "double", OPS_WRITE));


         int iter_range10[] = {0, nx0, 0, nx1, 0, nx2};
         ops_par_loop(taylor_green_vortex_block0_10_kernel, "Residual of equation", taylor_green_vortex_block, 3, iter_range10,
         ops_arg_dat(wk3, 1, stencil4, "double", OPS_READ),
         ops_arg_dat(u0, 1, stencil5, "double", OPS_READ),
         ops_arg_dat(wk7, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(p, 1, stencil6, "double", OPS_READ),
         ops_arg_dat(wk1, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk2, 1, stencil4, "double", OPS_READ),
         ops_arg_dat(wk8, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk5, 1, stencil4, "double", OPS_READ),
         ops_arg_dat(wk6, 1, stencil7, "double", OPS_READ),
         ops_arg_dat(rho, 1, stencil5, "double", OPS_READ),
         ops_arg_dat(rhou2, 1, stencil5, "double", OPS_READ),
         ops_arg_dat(rhou1, 1, stencil5, "double", OPS_READ),
         ops_arg_dat(wk4, 1, stencil8, "double", OPS_READ),
         ops_arg_dat(rhou0, 1, stencil5, "double", OPS_READ),
         ops_arg_dat(rhoE, 1, stencil5, "double", OPS_READ),
         ops_arg_dat(u1, 1, stencil5, "double", OPS_READ),
         ops_arg_dat(u2, 1, stencil5, "double", OPS_READ),
         ops_arg_dat(wk0, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(T, 1, stencil5, "double", OPS_READ),
         ops_arg_dat(wk10, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(wk11, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(wk12, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(wk13, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(wk9, 1, stencil0, "double", OPS_WRITE));


         int iter_range11[] = {-2, nx0 + 2, -2, nx1 + 2, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_11_kernel, "RK new (subloop) update", taylor_green_vortex_block, 3, iter_range11,
         ops_arg_dat(wk11, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk12, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rhou1_old, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk10, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rhoE_old, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk9, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rhou2_old, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk13, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rhou0_old, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rho_old, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rhou1, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(rhoE, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(rho, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(rhou2, 1, stencil0, "double", OPS_WRITE),
         ops_arg_dat(rhou0, 1, stencil0, "double", OPS_WRITE),
         ops_arg_gbl(&rknew[stage], 1, "double", OPS_READ));


         int iter_range12[] = {-2, nx0 + 2, -2, nx1 + 2, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_12_kernel, "RK old update", taylor_green_vortex_block, 3, iter_range12,
         ops_arg_dat(wk11, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk12, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk10, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk9, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(wk13, 1, stencil0, "double", OPS_READ),
         ops_arg_dat(rhou1_old, 1, stencil0, "double", OPS_RW),
         ops_arg_dat(rhou2_old, 1, stencil0, "double", OPS_RW),
         ops_arg_dat(rhou0_old, 1, stencil0, "double", OPS_RW),
         ops_arg_dat(rho_old, 1, stencil0, "double", OPS_RW),
         ops_arg_dat(rhoE_old, 1, stencil0, "double", OPS_RW),
         ops_arg_gbl(&rkold[stage], 1, "double", OPS_READ));



         int iter_range15[] = {0, 1, -2, nx1 + 2, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_15_kernel, "Symmetry bc 0 Left", taylor_green_vortex_block, 3, iter_range15,
         ops_arg_dat(rhou1, 1, stencil3, "double", OPS_RW),
         ops_arg_dat(rhoE, 1, stencil3, "double", OPS_RW),
         ops_arg_dat(rho, 1, stencil3, "double", OPS_RW),
         ops_arg_dat(rhou2, 1, stencil3, "double", OPS_RW),
         ops_arg_dat(rhou0, 1, stencil3, "double", OPS_RW));


         int iter_range16[] = {nx0 - 1, nx0, -2, nx1 + 2, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_16_kernel, "Symmetry bc 0 Right", taylor_green_vortex_block, 3, iter_range16,
         ops_arg_dat(rhou1, 1, stencil3, "double", OPS_RW),
         ops_arg_dat(rhoE, 1, stencil3, "double", OPS_RW),
         ops_arg_dat(rho, 1, stencil3, "double", OPS_RW),
         ops_arg_dat(rhou2, 1, stencil3, "double", OPS_RW),
         ops_arg_dat(rhou0, 1, stencil3, "double", OPS_RW));


         int iter_range17[] = {-2, nx0 + 2, 0, 1, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_17_kernel, "Symmetry bc 1 Left", taylor_green_vortex_block, 3, iter_range17,
         ops_arg_dat(rhou1, 1, stencil2, "double", OPS_RW),
         ops_arg_dat(rhoE, 1, stencil2, "double", OPS_RW),
         ops_arg_dat(rho, 1, stencil2, "double", OPS_RW),
         ops_arg_dat(rhou2, 1, stencil2, "double", OPS_RW),
         ops_arg_dat(rhou0, 1, stencil2, "double", OPS_RW));


         int iter_range18[] = {-2, nx0 + 2, nx1 - 1, nx1, -2, nx2 + 2};
         ops_par_loop(taylor_green_vortex_block0_18_kernel, "Symmetry bc 1 Right", taylor_green_vortex_block, 3, iter_range18,
         ops_arg_dat(rhou1, 1, stencil2, "double", OPS_RW),
         ops_arg_dat(rhoE, 1, stencil2, "double", OPS_RW),
         ops_arg_dat(rho, 1, stencil2, "double", OPS_RW),
         ops_arg_dat(rhou2, 1, stencil2, "double", OPS_RW),
         ops_arg_dat(rhou0, 1, stencil2, "double", OPS_RW));


         int iter_range19[] = {-2, nx0 + 2, -2, nx1 + 2, 0, 1};
         ops_par_loop(taylor_green_vortex_block0_19_kernel, "Symmetry bc 2 Left", taylor_green_vortex_block, 3, iter_range19,
         ops_arg_dat(rhou1, 1, stencil1, "double", OPS_RW),
         ops_arg_dat(rhoE, 1, stencil1, "double", OPS_RW),
         ops_arg_dat(rho, 1, stencil1, "double", OPS_RW),
         ops_arg_dat(rhou2, 1, stencil1, "double", OPS_RW),
         ops_arg_dat(rhou0, 1, stencil1, "double", OPS_RW));


         int iter_range20[] = {-2, nx0 + 2, -2, nx1 + 2, nx2 - 1, nx2};
         ops_par_loop(taylor_green_vortex_block0_20_kernel, "Symmetry bc 2 Right", taylor_green_vortex_block, 3, iter_range20,
         ops_arg_dat(rhou1, 1, stencil1, "double", OPS_RW),
         ops_arg_dat(rhoE, 1, stencil1, "double", OPS_RW),
         ops_arg_dat(rho, 1, stencil1, "double", OPS_RW),
         ops_arg_dat(rhou2, 1, stencil1, "double", OPS_RW),
         ops_arg_dat(rhou0, 1, stencil1, "double", OPS_RW));



      }




     ops_execute();
   }

   double cpu_end, elapsed_end;
   ops_timers(&cpu_end, &elapsed_end);

   // OPS diagnostics output for a diagnostic level 1 
   // no ops diagnostics

   ops_printf("\nTimings are:\n");
   ops_printf("-----------------------------------------\n");
   ops_timing_output(stdout);
   ops_printf("Total Wall time %lf\n",elapsed_end-elapsed_start);

   // Exit OPS 
   ops_exit();

}
