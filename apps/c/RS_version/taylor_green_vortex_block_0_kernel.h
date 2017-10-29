#ifndef block_0_KERNEL_H
#define block_0_KERNEL_H

 void taylor_green_vortex_block0_0_kernel(const double *rhou1 , const double *rho , const double *rhoE , const double
*rhou2 , const double *rhou0 , double *u0 , double *u1 , double *u2 , double *p , double *T)
{
    p[OPS_ACC8(0,0,0)] = (gama - 1)*((-rc0*pow(rhou0[OPS_ACC4(0,0,0)], 2) - rc0*pow(rhou1[OPS_ACC0(0,0,0)], 2) -
      rc0*pow(rhou2[OPS_ACC3(0,0,0)], 2))/rho[OPS_ACC1(0,0,0)] + rhoE[OPS_ACC2(0,0,0)]);
   u0[OPS_ACC5(0,0,0)] = rhou0[OPS_ACC4(0,0,0)]/rho[OPS_ACC1(0,0,0)];
   u2[OPS_ACC7(0,0,0)] = rhou2[OPS_ACC3(0,0,0)]/rho[OPS_ACC1(0,0,0)];
   u1[OPS_ACC6(0,0,0)] = rhou1[OPS_ACC0(0,0,0)]/rho[OPS_ACC1(0,0,0)];
    T[OPS_ACC9(0,0,0)] = gama*(gama - 1)*((-rc0*pow(rhou0[OPS_ACC4(0,0,0)], 2) - rc0*pow(rhou1[OPS_ACC0(0,0,0)], 2) -
      rc0*pow(rhou2[OPS_ACC3(0,0,0)], 2))/rho[OPS_ACC1(0,0,0)] + rhoE[OPS_ACC2(0,0,0)])*pow(Minf,
      2)/rho[OPS_ACC1(0,0,0)];
}


void taylor_green_vortex_block0_1_kernel(const double *u0 , double *wk0)
{
    wk0[OPS_ACC1(0,0,0)] = rinv1*((rc2)*u0[OPS_ACC0(0,0,-2)] - rc3*u0[OPS_ACC0(0,0,-1)] + (rc3)*u0[OPS_ACC0(0,0,1)] -
      rc2*u0[OPS_ACC0(0,0,2)]);
}


void taylor_green_vortex_block0_2_kernel(const double *u0 , double *wk1)
{
    wk1[OPS_ACC1(0,0,0)] = rinv4*((rc2)*u0[OPS_ACC0(0,-2,0)] - rc3*u0[OPS_ACC0(0,-1,0)] + (rc3)*u0[OPS_ACC0(0,1,0)] -
      rc2*u0[OPS_ACC0(0,2,0)]);
}


void taylor_green_vortex_block0_3_kernel(const double *u2 , double *wk2)
{
    wk2[OPS_ACC1(0,0,0)] = rinv5*((rc2)*u2[OPS_ACC0(-2,0,0)] - rc3*u2[OPS_ACC0(-1,0,0)] + (rc3)*u2[OPS_ACC0(1,0,0)] -
      rc2*u2[OPS_ACC0(2,0,0)]);
}


void taylor_green_vortex_block0_4_kernel(const double *u1 , double *wk3)
{
    wk3[OPS_ACC1(0,0,0)] = rinv4*((rc2)*u1[OPS_ACC0(0,-2,0)] - rc3*u1[OPS_ACC0(0,-1,0)] + (rc3)*u1[OPS_ACC0(0,1,0)] -
      rc2*u1[OPS_ACC0(0,2,0)]);
}


void taylor_green_vortex_block0_5_kernel(const double *u1 , double *wk4)
{
    wk4[OPS_ACC1(0,0,0)] = rinv5*((rc2)*u1[OPS_ACC0(-2,0,0)] - rc3*u1[OPS_ACC0(-1,0,0)] + (rc3)*u1[OPS_ACC0(1,0,0)] -
      rc2*u1[OPS_ACC0(2,0,0)]);
}


void taylor_green_vortex_block0_6_kernel(const double *u2 , double *wk5)
{
    wk5[OPS_ACC1(0,0,0)] = rinv4*((rc2)*u2[OPS_ACC0(0,-2,0)] - rc3*u2[OPS_ACC0(0,-1,0)] + (rc3)*u2[OPS_ACC0(0,1,0)] -
      rc2*u2[OPS_ACC0(0,2,0)]);
}


void taylor_green_vortex_block0_7_kernel(const double *u0 , double *wk6)
{
    wk6[OPS_ACC1(0,0,0)] = rinv5*((rc2)*u0[OPS_ACC0(-2,0,0)] - rc3*u0[OPS_ACC0(-1,0,0)] + (rc3)*u0[OPS_ACC0(1,0,0)] -
      rc2*u0[OPS_ACC0(2,0,0)]);
}


void taylor_green_vortex_block0_8_kernel(const double *u2 , double *wk7)
{
    wk7[OPS_ACC1(0,0,0)] = rinv1*((rc2)*u2[OPS_ACC0(0,0,-2)] - rc3*u2[OPS_ACC0(0,0,-1)] + (rc3)*u2[OPS_ACC0(0,0,1)] -
      rc2*u2[OPS_ACC0(0,0,2)]);
}


void taylor_green_vortex_block0_9_kernel(const double *u1 , double *wk8)
{
    wk8[OPS_ACC1(0,0,0)] = rinv1*((rc2)*u1[OPS_ACC0(0,0,-2)] - rc3*u1[OPS_ACC0(0,0,-1)] + (rc3)*u1[OPS_ACC0(0,0,1)] -
      rc2*u1[OPS_ACC0(0,0,2)]);
}


 void taylor_green_vortex_block0_10_kernel(const double *wk3 , const double *u0 , const double *wk7 , const double *p ,
const double *wk1 , const double *wk2 , const double *wk8 , const double *wk5 , const double *wk6 , const double *rho ,
const double *rhou2 , const double *rhou1 , const double *wk4 , const double *rhou0 , const double *rhoE , const double
*u1 , const double *u2 , const double *wk0 , const double *T , double *wk10 , double *wk11 , double *wk12 , double *wk13
, double *wk9)
{
    wk9[OPS_ACC23(0,0,0)] = -0.5*rinv1*((rc2)*rho[OPS_ACC9(0,0,-2)]*u2[OPS_ACC16(0,0,-2)] -
      rc3*rho[OPS_ACC9(0,0,-1)]*u2[OPS_ACC16(0,0,-1)] + (rc3)*rho[OPS_ACC9(0,0,1)]*u2[OPS_ACC16(0,0,1)] -
      rc2*rho[OPS_ACC9(0,0,2)]*u2[OPS_ACC16(0,0,2)]) - 0.5*rinv1*((rc2)*rho[OPS_ACC9(0,0,-2)] -
      rc3*rho[OPS_ACC9(0,0,-1)] + (rc3)*rho[OPS_ACC9(0,0,1)] - rc2*rho[OPS_ACC9(0,0,2)])*u2[OPS_ACC16(0,0,0)] -
      0.5*rinv4*((rc2)*rho[OPS_ACC9(0,-2,0)]*u1[OPS_ACC15(0,-2,0)] - rc3*rho[OPS_ACC9(0,-1,0)]*u1[OPS_ACC15(0,-1,0)] +
      (rc3)*rho[OPS_ACC9(0,1,0)]*u1[OPS_ACC15(0,1,0)] - rc2*rho[OPS_ACC9(0,2,0)]*u1[OPS_ACC15(0,2,0)]) -
      0.5*rinv4*((rc2)*rho[OPS_ACC9(0,-2,0)] - rc3*rho[OPS_ACC9(0,-1,0)] + (rc3)*rho[OPS_ACC9(0,1,0)] -
      rc2*rho[OPS_ACC9(0,2,0)])*u1[OPS_ACC15(0,0,0)] - 0.5*rinv5*((rc2)*rho[OPS_ACC9(-2,0,0)]*u0[OPS_ACC1(-2,0,0)] -
      rc3*rho[OPS_ACC9(-1,0,0)]*u0[OPS_ACC1(-1,0,0)] + (rc3)*rho[OPS_ACC9(1,0,0)]*u0[OPS_ACC1(1,0,0)] -
      rc2*rho[OPS_ACC9(2,0,0)]*u0[OPS_ACC1(2,0,0)]) - 0.5*rinv5*((rc2)*rho[OPS_ACC9(-2,0,0)] - rc3*rho[OPS_ACC9(-1,0,0)]
      + (rc3)*rho[OPS_ACC9(1,0,0)] - rc2*rho[OPS_ACC9(2,0,0)])*u0[OPS_ACC1(0,0,0)] - 0.5*(wk3[OPS_ACC0(0,0,0)] +
      wk6[OPS_ACC8(0,0,0)] + wk7[OPS_ACC2(0,0,0)])*rho[OPS_ACC9(0,0,0)];
    wk10[OPS_ACC19(0,0,0)] = -0.5*sqrt(rinv6)*((rc2)*rhou0[OPS_ACC13(0,-2,0)]*u1[OPS_ACC15(0,-2,0)] -
      rc3*rhou0[OPS_ACC13(0,-1,0)]*u1[OPS_ACC15(0,-1,0)] + (rc3)*rhou0[OPS_ACC13(0,1,0)]*u1[OPS_ACC15(0,1,0)] -
      rc2*rhou0[OPS_ACC13(0,2,0)]*u1[OPS_ACC15(0,2,0)]) - 0.5*sqrt(rinv6)*((rc2)*rhou0[OPS_ACC13(0,-2,0)] -
      rc3*rhou0[OPS_ACC13(0,-1,0)] + (rc3)*rhou0[OPS_ACC13(0,1,0)] - rc2*rhou0[OPS_ACC13(0,2,0)])*u1[OPS_ACC15(0,0,0)] -
      0.5*sqrt(rinv7)*((rc2)*rhou0[OPS_ACC13(-2,0,0)]*u0[OPS_ACC1(-2,0,0)] -
      rc3*rhou0[OPS_ACC13(-1,0,0)]*u0[OPS_ACC1(-1,0,0)] + (rc3)*rhou0[OPS_ACC13(1,0,0)]*u0[OPS_ACC1(1,0,0)] -
      rc2*rhou0[OPS_ACC13(2,0,0)]*u0[OPS_ACC1(2,0,0)]) - sqrt(rinv7)*((rc2)*p[OPS_ACC3(-2,0,0)] -
      rc3*p[OPS_ACC3(-1,0,0)] + (rc3)*p[OPS_ACC3(1,0,0)] - rc2*p[OPS_ACC3(2,0,0)]) -
      0.5*sqrt(rinv7)*((rc2)*rhou0[OPS_ACC13(-2,0,0)] - rc3*rhou0[OPS_ACC13(-1,0,0)] + (rc3)*rhou0[OPS_ACC13(1,0,0)] -
      rc2*rhou0[OPS_ACC13(2,0,0)])*u0[OPS_ACC1(0,0,0)] + rinv8*(sqrt(rinv6)*((rc2)*wk4[OPS_ACC12(0,-2,0)] -
      rc3*wk4[OPS_ACC12(0,-1,0)] + (rc3)*wk4[OPS_ACC12(0,1,0)] - rc2*wk4[OPS_ACC12(0,2,0)]) +
      rinv6*(-rc10*u0[OPS_ACC1(0,0,0)] - rc2*u0[OPS_ACC1(0,-2,0)] + (rc11)*u0[OPS_ACC1(0,-1,0)] +
      (rc11)*u0[OPS_ACC1(0,1,0)] - rc2*u0[OPS_ACC1(0,2,0)])) + rinv8*(sqrt(rinv9)*((rc2)*wk2[OPS_ACC5(0,0,-2)] -
      rc3*wk2[OPS_ACC5(0,0,-1)] + (rc3)*wk2[OPS_ACC5(0,0,1)] - rc2*wk2[OPS_ACC5(0,0,2)]) +
      rinv9*(-rc10*u0[OPS_ACC1(0,0,0)] - rc2*u0[OPS_ACC1(0,0,-2)] + (rc11)*u0[OPS_ACC1(0,0,-1)] +
      (rc11)*u0[OPS_ACC1(0,0,1)] - rc2*u0[OPS_ACC1(0,0,2)])) + rinv8*(-rc3*sqrt(rinv6)*((rc2)*wk4[OPS_ACC12(0,-2,0)] -
      rc3*wk4[OPS_ACC12(0,-1,0)] + (rc3)*wk4[OPS_ACC12(0,1,0)] - rc2*wk4[OPS_ACC12(0,2,0)]) +
      (rc11)*rinv7*(-rc10*u0[OPS_ACC1(0,0,0)] - rc2*u0[OPS_ACC1(-2,0,0)] + (rc11)*u0[OPS_ACC1(-1,0,0)] +
      (rc11)*u0[OPS_ACC1(1,0,0)] - rc2*u0[OPS_ACC1(2,0,0)]) - rc3*sqrt(rinv9)*((rc2)*wk2[OPS_ACC5(0,0,-2)] -
      rc3*wk2[OPS_ACC5(0,0,-1)] + (rc3)*wk2[OPS_ACC5(0,0,1)] - rc2*wk2[OPS_ACC5(0,0,2)])) -
      0.5*sqrt(rinv9)*((rc2)*rhou0[OPS_ACC13(0,0,-2)]*u2[OPS_ACC16(0,0,-2)] -
      rc3*rhou0[OPS_ACC13(0,0,-1)]*u2[OPS_ACC16(0,0,-1)] + (rc3)*rhou0[OPS_ACC13(0,0,1)]*u2[OPS_ACC16(0,0,1)] -
      rc2*rhou0[OPS_ACC13(0,0,2)]*u2[OPS_ACC16(0,0,2)]) - 0.5*sqrt(rinv9)*((rc2)*rhou0[OPS_ACC13(0,0,-2)] -
      rc3*rhou0[OPS_ACC13(0,0,-1)] + (rc3)*rhou0[OPS_ACC13(0,0,1)] - rc2*rhou0[OPS_ACC13(0,0,2)])*u2[OPS_ACC16(0,0,0)] -
      0.5*(wk3[OPS_ACC0(0,0,0)] + wk6[OPS_ACC8(0,0,0)] + wk7[OPS_ACC2(0,0,0)])*rhou0[OPS_ACC13(0,0,0)];
    wk11[OPS_ACC20(0,0,0)] = -0.5*sqrt(rinv6)*((rc2)*rhou1[OPS_ACC11(0,-2,0)]*u1[OPS_ACC15(0,-2,0)] -
      rc3*rhou1[OPS_ACC11(0,-1,0)]*u1[OPS_ACC15(0,-1,0)] + (rc3)*rhou1[OPS_ACC11(0,1,0)]*u1[OPS_ACC15(0,1,0)] -
      rc2*rhou1[OPS_ACC11(0,2,0)]*u1[OPS_ACC15(0,2,0)]) - sqrt(rinv6)*((rc2)*p[OPS_ACC3(0,-2,0)] -
      rc3*p[OPS_ACC3(0,-1,0)] + (rc3)*p[OPS_ACC3(0,1,0)] - rc2*p[OPS_ACC3(0,2,0)]) -
      0.5*sqrt(rinv6)*((rc2)*rhou1[OPS_ACC11(0,-2,0)] - rc3*rhou1[OPS_ACC11(0,-1,0)] + (rc3)*rhou1[OPS_ACC11(0,1,0)] -
      rc2*rhou1[OPS_ACC11(0,2,0)])*u1[OPS_ACC15(0,0,0)] -
      0.5*sqrt(rinv7)*((rc2)*rhou1[OPS_ACC11(-2,0,0)]*u0[OPS_ACC1(-2,0,0)] -
      rc3*rhou1[OPS_ACC11(-1,0,0)]*u0[OPS_ACC1(-1,0,0)] + (rc3)*rhou1[OPS_ACC11(1,0,0)]*u0[OPS_ACC1(1,0,0)] -
      rc2*rhou1[OPS_ACC11(2,0,0)]*u0[OPS_ACC1(2,0,0)]) - 0.5*sqrt(rinv7)*((rc2)*rhou1[OPS_ACC11(-2,0,0)] -
      rc3*rhou1[OPS_ACC11(-1,0,0)] + (rc3)*rhou1[OPS_ACC11(1,0,0)] - rc2*rhou1[OPS_ACC11(2,0,0)])*u0[OPS_ACC1(0,0,0)] +
      rinv8*(sqrt(rinv6)*((rc2)*wk6[OPS_ACC8(0,-2,0)] - rc3*wk6[OPS_ACC8(0,-1,0)] + (rc3)*wk6[OPS_ACC8(0,1,0)] -
      rc2*wk6[OPS_ACC8(0,2,0)]) + rinv7*(-rc10*u1[OPS_ACC15(0,0,0)] - rc2*u1[OPS_ACC15(-2,0,0)] +
      (rc11)*u1[OPS_ACC15(-1,0,0)] + (rc11)*u1[OPS_ACC15(1,0,0)] - rc2*u1[OPS_ACC15(2,0,0)])) +
      rinv8*(sqrt(rinv9)*((rc2)*wk5[OPS_ACC7(0,0,-2)] - rc3*wk5[OPS_ACC7(0,0,-1)] + (rc3)*wk5[OPS_ACC7(0,0,1)] -
      rc2*wk5[OPS_ACC7(0,0,2)]) + rinv9*(-rc10*u1[OPS_ACC15(0,0,0)] - rc2*u1[OPS_ACC15(0,0,-2)] +
      (rc11)*u1[OPS_ACC15(0,0,-1)] + (rc11)*u1[OPS_ACC15(0,0,1)] - rc2*u1[OPS_ACC15(0,0,2)])) +
      rinv8*(-rc3*sqrt(rinv6)*((rc2)*wk6[OPS_ACC8(0,-2,0)] - rc3*wk6[OPS_ACC8(0,-1,0)] + (rc3)*wk6[OPS_ACC8(0,1,0)] -
      rc2*wk6[OPS_ACC8(0,2,0)]) + (rc11)*rinv6*(-rc10*u1[OPS_ACC15(0,0,0)] - rc2*u1[OPS_ACC15(0,-2,0)] +
      (rc11)*u1[OPS_ACC15(0,-1,0)] + (rc11)*u1[OPS_ACC15(0,1,0)] - rc2*u1[OPS_ACC15(0,2,0)]) -
      rc3*sqrt(rinv9)*((rc2)*wk5[OPS_ACC7(0,0,-2)] - rc3*wk5[OPS_ACC7(0,0,-1)] + (rc3)*wk5[OPS_ACC7(0,0,1)] -
      rc2*wk5[OPS_ACC7(0,0,2)])) - 0.5*sqrt(rinv9)*((rc2)*rhou1[OPS_ACC11(0,0,-2)]*u2[OPS_ACC16(0,0,-2)] -
      rc3*rhou1[OPS_ACC11(0,0,-1)]*u2[OPS_ACC16(0,0,-1)] + (rc3)*rhou1[OPS_ACC11(0,0,1)]*u2[OPS_ACC16(0,0,1)] -
      rc2*rhou1[OPS_ACC11(0,0,2)]*u2[OPS_ACC16(0,0,2)]) - 0.5*sqrt(rinv9)*((rc2)*rhou1[OPS_ACC11(0,0,-2)] -
      rc3*rhou1[OPS_ACC11(0,0,-1)] + (rc3)*rhou1[OPS_ACC11(0,0,1)] - rc2*rhou1[OPS_ACC11(0,0,2)])*u2[OPS_ACC16(0,0,0)] -
      0.5*(wk3[OPS_ACC0(0,0,0)] + wk6[OPS_ACC8(0,0,0)] + wk7[OPS_ACC2(0,0,0)])*rhou1[OPS_ACC11(0,0,0)];
    wk12[OPS_ACC21(0,0,0)] = -0.5*sqrt(rinv6)*((rc2)*rhou2[OPS_ACC10(0,-2,0)]*u1[OPS_ACC15(0,-2,0)] -
      rc3*rhou2[OPS_ACC10(0,-1,0)]*u1[OPS_ACC15(0,-1,0)] + (rc3)*rhou2[OPS_ACC10(0,1,0)]*u1[OPS_ACC15(0,1,0)] -
      rc2*rhou2[OPS_ACC10(0,2,0)]*u1[OPS_ACC15(0,2,0)]) - 0.5*sqrt(rinv6)*((rc2)*rhou2[OPS_ACC10(0,-2,0)] -
      rc3*rhou2[OPS_ACC10(0,-1,0)] + (rc3)*rhou2[OPS_ACC10(0,1,0)] - rc2*rhou2[OPS_ACC10(0,2,0)])*u1[OPS_ACC15(0,0,0)] -
      0.5*sqrt(rinv7)*((rc2)*rhou2[OPS_ACC10(-2,0,0)]*u0[OPS_ACC1(-2,0,0)] -
      rc3*rhou2[OPS_ACC10(-1,0,0)]*u0[OPS_ACC1(-1,0,0)] + (rc3)*rhou2[OPS_ACC10(1,0,0)]*u0[OPS_ACC1(1,0,0)] -
      rc2*rhou2[OPS_ACC10(2,0,0)]*u0[OPS_ACC1(2,0,0)]) - 0.5*sqrt(rinv7)*((rc2)*rhou2[OPS_ACC10(-2,0,0)] -
      rc3*rhou2[OPS_ACC10(-1,0,0)] + (rc3)*rhou2[OPS_ACC10(1,0,0)] - rc2*rhou2[OPS_ACC10(2,0,0)])*u0[OPS_ACC1(0,0,0)] +
      rinv8*(rinv6*(-rc10*u2[OPS_ACC16(0,0,0)] - rc2*u2[OPS_ACC16(0,-2,0)] + (rc11)*u2[OPS_ACC16(0,-1,0)] +
      (rc11)*u2[OPS_ACC16(0,1,0)] - rc2*u2[OPS_ACC16(0,2,0)]) + sqrt(rinv9)*((rc2)*wk3[OPS_ACC0(0,0,-2)] -
      rc3*wk3[OPS_ACC0(0,0,-1)] + (rc3)*wk3[OPS_ACC0(0,0,1)] - rc2*wk3[OPS_ACC0(0,0,2)])) +
      rinv8*(rinv7*(-rc10*u2[OPS_ACC16(0,0,0)] - rc2*u2[OPS_ACC16(-2,0,0)] + (rc11)*u2[OPS_ACC16(-1,0,0)] +
      (rc11)*u2[OPS_ACC16(1,0,0)] - rc2*u2[OPS_ACC16(2,0,0)]) + sqrt(rinv9)*((rc2)*wk6[OPS_ACC8(0,0,-2)] -
      rc3*wk6[OPS_ACC8(0,0,-1)] + (rc3)*wk6[OPS_ACC8(0,0,1)] - rc2*wk6[OPS_ACC8(0,0,2)])) +
      rinv8*(-rc3*sqrt(rinv9)*((rc2)*wk3[OPS_ACC0(0,0,-2)] - rc3*wk3[OPS_ACC0(0,0,-1)] + (rc3)*wk3[OPS_ACC0(0,0,1)] -
      rc2*wk3[OPS_ACC0(0,0,2)]) - rc3*sqrt(rinv9)*((rc2)*wk6[OPS_ACC8(0,0,-2)] - rc3*wk6[OPS_ACC8(0,0,-1)] +
      (rc3)*wk6[OPS_ACC8(0,0,1)] - rc2*wk6[OPS_ACC8(0,0,2)]) + (rc11)*rinv9*(-rc10*u2[OPS_ACC16(0,0,0)] -
      rc2*u2[OPS_ACC16(0,0,-2)] + (rc11)*u2[OPS_ACC16(0,0,-1)] + (rc11)*u2[OPS_ACC16(0,0,1)] -
      rc2*u2[OPS_ACC16(0,0,2)])) - 0.5*sqrt(rinv9)*((rc2)*rhou2[OPS_ACC10(0,0,-2)]*u2[OPS_ACC16(0,0,-2)] -
      rc3*rhou2[OPS_ACC10(0,0,-1)]*u2[OPS_ACC16(0,0,-1)] + (rc3)*rhou2[OPS_ACC10(0,0,1)]*u2[OPS_ACC16(0,0,1)] -
      rc2*rhou2[OPS_ACC10(0,0,2)]*u2[OPS_ACC16(0,0,2)]) - sqrt(rinv9)*((rc2)*p[OPS_ACC3(0,0,-2)] -
      rc3*p[OPS_ACC3(0,0,-1)] + (rc3)*p[OPS_ACC3(0,0,1)] - rc2*p[OPS_ACC3(0,0,2)]) -
      0.5*sqrt(rinv9)*((rc2)*rhou2[OPS_ACC10(0,0,-2)] - rc3*rhou2[OPS_ACC10(0,0,-1)] + (rc3)*rhou2[OPS_ACC10(0,0,1)] -
      rc2*rhou2[OPS_ACC10(0,0,2)])*u2[OPS_ACC16(0,0,0)] - 0.5*(wk3[OPS_ACC0(0,0,0)] + wk6[OPS_ACC8(0,0,0)] +
      wk7[OPS_ACC2(0,0,0)])*rhou2[OPS_ACC10(0,0,0)];
    wk13[OPS_ACC22(0,0,0)] = rinv12*rinv13*rinv14*rinv6*rinv8*(-rc10*T[OPS_ACC18(0,0,0)] - rc2*T[OPS_ACC18(0,-2,0)] +
      (rc11)*T[OPS_ACC18(0,-1,0)] + (rc11)*T[OPS_ACC18(0,1,0)] - rc2*T[OPS_ACC18(0,2,0)]) +
      rinv12*rinv13*rinv14*rinv7*rinv8*(-rc10*T[OPS_ACC18(0,0,0)] - rc2*T[OPS_ACC18(-2,0,0)] +
      (rc11)*T[OPS_ACC18(-1,0,0)] + (rc11)*T[OPS_ACC18(1,0,0)] - rc2*T[OPS_ACC18(2,0,0)]) +
      rinv12*rinv13*rinv14*rinv8*rinv9*(-rc10*T[OPS_ACC18(0,0,0)] - rc2*T[OPS_ACC18(0,0,-2)] +
      (rc11)*T[OPS_ACC18(0,0,-1)] + (rc11)*T[OPS_ACC18(0,0,1)] - rc2*T[OPS_ACC18(0,0,2)]) -
      sqrt(rinv6)*((rc2)*p[OPS_ACC3(0,-2,0)]*u1[OPS_ACC15(0,-2,0)] - rc3*p[OPS_ACC3(0,-1,0)]*u1[OPS_ACC15(0,-1,0)] +
      (rc3)*p[OPS_ACC3(0,1,0)]*u1[OPS_ACC15(0,1,0)] - rc2*p[OPS_ACC3(0,2,0)]*u1[OPS_ACC15(0,2,0)]) -
      0.5*sqrt(rinv6)*((rc2)*rhoE[OPS_ACC14(0,-2,0)]*u1[OPS_ACC15(0,-2,0)] -
      rc3*rhoE[OPS_ACC14(0,-1,0)]*u1[OPS_ACC15(0,-1,0)] + (rc3)*rhoE[OPS_ACC14(0,1,0)]*u1[OPS_ACC15(0,1,0)] -
      rc2*rhoE[OPS_ACC14(0,2,0)]*u1[OPS_ACC15(0,2,0)]) - 0.5*sqrt(rinv6)*((rc2)*rhoE[OPS_ACC14(0,-2,0)] -
      rc3*rhoE[OPS_ACC14(0,-1,0)] + (rc3)*rhoE[OPS_ACC14(0,1,0)] - rc2*rhoE[OPS_ACC14(0,2,0)])*u1[OPS_ACC15(0,0,0)] -
      sqrt(rinv7)*((rc2)*p[OPS_ACC3(-2,0,0)]*u0[OPS_ACC1(-2,0,0)] - rc3*p[OPS_ACC3(-1,0,0)]*u0[OPS_ACC1(-1,0,0)] +
      (rc3)*p[OPS_ACC3(1,0,0)]*u0[OPS_ACC1(1,0,0)] - rc2*p[OPS_ACC3(2,0,0)]*u0[OPS_ACC1(2,0,0)]) -
      0.5*sqrt(rinv7)*((rc2)*rhoE[OPS_ACC14(-2,0,0)]*u0[OPS_ACC1(-2,0,0)] -
      rc3*rhoE[OPS_ACC14(-1,0,0)]*u0[OPS_ACC1(-1,0,0)] + (rc3)*rhoE[OPS_ACC14(1,0,0)]*u0[OPS_ACC1(1,0,0)] -
      rc2*rhoE[OPS_ACC14(2,0,0)]*u0[OPS_ACC1(2,0,0)]) - 0.5*sqrt(rinv7)*((rc2)*rhoE[OPS_ACC14(-2,0,0)] -
      rc3*rhoE[OPS_ACC14(-1,0,0)] + (rc3)*rhoE[OPS_ACC14(1,0,0)] - rc2*rhoE[OPS_ACC14(2,0,0)])*u0[OPS_ACC1(0,0,0)] +
      rinv8*(sqrt(rinv6)*((rc2)*wk4[OPS_ACC12(0,-2,0)] - rc3*wk4[OPS_ACC12(0,-1,0)] + (rc3)*wk4[OPS_ACC12(0,1,0)] -
      rc2*wk4[OPS_ACC12(0,2,0)]) + rinv6*(-rc10*u0[OPS_ACC1(0,0,0)] - rc2*u0[OPS_ACC1(0,-2,0)] +
      (rc11)*u0[OPS_ACC1(0,-1,0)] + (rc11)*u0[OPS_ACC1(0,1,0)] - rc2*u0[OPS_ACC1(0,2,0)]))*u0[OPS_ACC1(0,0,0)] +
      rinv8*(sqrt(rinv6)*((rc2)*wk6[OPS_ACC8(0,-2,0)] - rc3*wk6[OPS_ACC8(0,-1,0)] + (rc3)*wk6[OPS_ACC8(0,1,0)] -
      rc2*wk6[OPS_ACC8(0,2,0)]) + rinv7*(-rc10*u1[OPS_ACC15(0,0,0)] - rc2*u1[OPS_ACC15(-2,0,0)] +
      (rc11)*u1[OPS_ACC15(-1,0,0)] + (rc11)*u1[OPS_ACC15(1,0,0)] - rc2*u1[OPS_ACC15(2,0,0)]))*u1[OPS_ACC15(0,0,0)] +
      rinv8*(rinv6*(-rc10*u2[OPS_ACC16(0,0,0)] - rc2*u2[OPS_ACC16(0,-2,0)] + (rc11)*u2[OPS_ACC16(0,-1,0)] +
      (rc11)*u2[OPS_ACC16(0,1,0)] - rc2*u2[OPS_ACC16(0,2,0)]) + sqrt(rinv9)*((rc2)*wk3[OPS_ACC0(0,0,-2)] -
      rc3*wk3[OPS_ACC0(0,0,-1)] + (rc3)*wk3[OPS_ACC0(0,0,1)] - rc2*wk3[OPS_ACC0(0,0,2)]))*u2[OPS_ACC16(0,0,0)] +
      rinv8*(rinv7*(-rc10*u2[OPS_ACC16(0,0,0)] - rc2*u2[OPS_ACC16(-2,0,0)] + (rc11)*u2[OPS_ACC16(-1,0,0)] +
      (rc11)*u2[OPS_ACC16(1,0,0)] - rc2*u2[OPS_ACC16(2,0,0)]) + sqrt(rinv9)*((rc2)*wk6[OPS_ACC8(0,0,-2)] -
      rc3*wk6[OPS_ACC8(0,0,-1)] + (rc3)*wk6[OPS_ACC8(0,0,1)] - rc2*wk6[OPS_ACC8(0,0,2)]))*u2[OPS_ACC16(0,0,0)] +
      rinv8*(sqrt(rinv9)*((rc2)*wk2[OPS_ACC5(0,0,-2)] - rc3*wk2[OPS_ACC5(0,0,-1)] + (rc3)*wk2[OPS_ACC5(0,0,1)] -
      rc2*wk2[OPS_ACC5(0,0,2)]) + rinv9*(-rc10*u0[OPS_ACC1(0,0,0)] - rc2*u0[OPS_ACC1(0,0,-2)] +
      (rc11)*u0[OPS_ACC1(0,0,-1)] + (rc11)*u0[OPS_ACC1(0,0,1)] - rc2*u0[OPS_ACC1(0,0,2)]))*u0[OPS_ACC1(0,0,0)] +
      rinv8*(sqrt(rinv9)*((rc2)*wk5[OPS_ACC7(0,0,-2)] - rc3*wk5[OPS_ACC7(0,0,-1)] + (rc3)*wk5[OPS_ACC7(0,0,1)] -
      rc2*wk5[OPS_ACC7(0,0,2)]) + rinv9*(-rc10*u1[OPS_ACC15(0,0,0)] - rc2*u1[OPS_ACC15(0,0,-2)] +
      (rc11)*u1[OPS_ACC15(0,0,-1)] + (rc11)*u1[OPS_ACC15(0,0,1)] - rc2*u1[OPS_ACC15(0,0,2)]))*u1[OPS_ACC15(0,0,0)] +
      rinv8*(wk0[OPS_ACC17(0,0,0)] + wk2[OPS_ACC5(0,0,0)])*wk0[OPS_ACC17(0,0,0)] + rinv8*(wk0[OPS_ACC17(0,0,0)] +
      wk2[OPS_ACC5(0,0,0)])*wk2[OPS_ACC5(0,0,0)] + rinv8*(wk1[OPS_ACC4(0,0,0)] +
      wk4[OPS_ACC12(0,0,0)])*wk1[OPS_ACC4(0,0,0)] + rinv8*(wk1[OPS_ACC4(0,0,0)] +
      wk4[OPS_ACC12(0,0,0)])*wk4[OPS_ACC12(0,0,0)] + rinv8*(wk5[OPS_ACC7(0,0,0)] +
      wk8[OPS_ACC6(0,0,0)])*wk5[OPS_ACC7(0,0,0)] + rinv8*(wk5[OPS_ACC7(0,0,0)] +
      wk8[OPS_ACC6(0,0,0)])*wk8[OPS_ACC6(0,0,0)] + rinv8*(-rc3*sqrt(rinv6)*((rc2)*wk4[OPS_ACC12(0,-2,0)] -
      rc3*wk4[OPS_ACC12(0,-1,0)] + (rc3)*wk4[OPS_ACC12(0,1,0)] - rc2*wk4[OPS_ACC12(0,2,0)]) +
      (rc11)*rinv7*(-rc10*u0[OPS_ACC1(0,0,0)] - rc2*u0[OPS_ACC1(-2,0,0)] + (rc11)*u0[OPS_ACC1(-1,0,0)] +
      (rc11)*u0[OPS_ACC1(1,0,0)] - rc2*u0[OPS_ACC1(2,0,0)]) - rc3*sqrt(rinv9)*((rc2)*wk2[OPS_ACC5(0,0,-2)] -
      rc3*wk2[OPS_ACC5(0,0,-1)] + (rc3)*wk2[OPS_ACC5(0,0,1)] - rc2*wk2[OPS_ACC5(0,0,2)]))*u0[OPS_ACC1(0,0,0)] +
      rinv8*(-rc3*sqrt(rinv6)*((rc2)*wk6[OPS_ACC8(0,-2,0)] - rc3*wk6[OPS_ACC8(0,-1,0)] + (rc3)*wk6[OPS_ACC8(0,1,0)] -
      rc2*wk6[OPS_ACC8(0,2,0)]) + (rc11)*rinv6*(-rc10*u1[OPS_ACC15(0,0,0)] - rc2*u1[OPS_ACC15(0,-2,0)] +
      (rc11)*u1[OPS_ACC15(0,-1,0)] + (rc11)*u1[OPS_ACC15(0,1,0)] - rc2*u1[OPS_ACC15(0,2,0)]) -
      rc3*sqrt(rinv9)*((rc2)*wk5[OPS_ACC7(0,0,-2)] - rc3*wk5[OPS_ACC7(0,0,-1)] + (rc3)*wk5[OPS_ACC7(0,0,1)] -
      rc2*wk5[OPS_ACC7(0,0,2)]))*u1[OPS_ACC15(0,0,0)] + rinv8*(-rc3*sqrt(rinv9)*((rc2)*wk3[OPS_ACC0(0,0,-2)] -
      rc3*wk3[OPS_ACC0(0,0,-1)] + (rc3)*wk3[OPS_ACC0(0,0,1)] - rc2*wk3[OPS_ACC0(0,0,2)]) -
      rc3*sqrt(rinv9)*((rc2)*wk6[OPS_ACC8(0,0,-2)] - rc3*wk6[OPS_ACC8(0,0,-1)] + (rc3)*wk6[OPS_ACC8(0,0,1)] -
      rc2*wk6[OPS_ACC8(0,0,2)]) + (rc11)*rinv9*(-rc10*u2[OPS_ACC16(0,0,0)] - rc2*u2[OPS_ACC16(0,0,-2)] +
      (rc11)*u2[OPS_ACC16(0,0,-1)] + (rc11)*u2[OPS_ACC16(0,0,1)] - rc2*u2[OPS_ACC16(0,0,2)]))*u2[OPS_ACC16(0,0,0)] +
      rinv8*(-rc3*wk3[OPS_ACC0(0,0,0)] - rc3*wk6[OPS_ACC8(0,0,0)] + (rc11)*wk7[OPS_ACC2(0,0,0)])*wk7[OPS_ACC2(0,0,0)] +
      rinv8*(-rc3*wk3[OPS_ACC0(0,0,0)] + (rc11)*wk6[OPS_ACC8(0,0,0)] - rc3*wk7[OPS_ACC2(0,0,0)])*wk6[OPS_ACC8(0,0,0)] +
      rinv8*((rc11)*wk3[OPS_ACC0(0,0,0)] - rc3*wk6[OPS_ACC8(0,0,0)] - rc3*wk7[OPS_ACC2(0,0,0)])*wk3[OPS_ACC0(0,0,0)] -
      sqrt(rinv9)*((rc2)*p[OPS_ACC3(0,0,-2)]*u2[OPS_ACC16(0,0,-2)] - rc3*p[OPS_ACC3(0,0,-1)]*u2[OPS_ACC16(0,0,-1)] +
      (rc3)*p[OPS_ACC3(0,0,1)]*u2[OPS_ACC16(0,0,1)] - rc2*p[OPS_ACC3(0,0,2)]*u2[OPS_ACC16(0,0,2)]) -
      0.5*sqrt(rinv9)*((rc2)*rhoE[OPS_ACC14(0,0,-2)]*u2[OPS_ACC16(0,0,-2)] -
      rc3*rhoE[OPS_ACC14(0,0,-1)]*u2[OPS_ACC16(0,0,-1)] + (rc3)*rhoE[OPS_ACC14(0,0,1)]*u2[OPS_ACC16(0,0,1)] -
      rc2*rhoE[OPS_ACC14(0,0,2)]*u2[OPS_ACC16(0,0,2)]) - 0.5*sqrt(rinv9)*((rc2)*rhoE[OPS_ACC14(0,0,-2)] -
      rc3*rhoE[OPS_ACC14(0,0,-1)] + (rc3)*rhoE[OPS_ACC14(0,0,1)] - rc2*rhoE[OPS_ACC14(0,0,2)])*u2[OPS_ACC16(0,0,0)] -
      0.5*(wk3[OPS_ACC0(0,0,0)] + wk6[OPS_ACC8(0,0,0)] + wk7[OPS_ACC2(0,0,0)])*rhoE[OPS_ACC14(0,0,0)];
}


 void taylor_green_vortex_block0_11_kernel(const double *wk11 , const double *wk12 , const double *rhou1_old , const
double *wk10 , const double *rhoE_old , const double *wk9 , const double *rhou2_old , const double *wk13 , const double
*rhou0_old , const double *rho_old , double *rhou1 , double *rhoE , double *rho , double *rhou2 , double *rhou0 , const
double *rknew)
{
   rho[OPS_ACC12(0,0,0)] = deltat*rknew[0]*wk9[OPS_ACC5(0,0,0)] + rho_old[OPS_ACC9(0,0,0)];
   rhou0[OPS_ACC14(0,0,0)] = deltat*rknew[0]*wk10[OPS_ACC3(0,0,0)] + rhou0_old[OPS_ACC8(0,0,0)];
   rhou1[OPS_ACC10(0,0,0)] = deltat*rknew[0]*wk11[OPS_ACC0(0,0,0)] + rhou1_old[OPS_ACC2(0,0,0)];
   rhou2[OPS_ACC13(0,0,0)] = deltat*rknew[0]*wk12[OPS_ACC1(0,0,0)] + rhou2_old[OPS_ACC6(0,0,0)];
   rhoE[OPS_ACC11(0,0,0)] = deltat*rknew[0]*wk13[OPS_ACC7(0,0,0)] + rhoE_old[OPS_ACC4(0,0,0)];
}


 void taylor_green_vortex_block0_12_kernel(const double *wk11 , const double *wk12 , const double *wk10 , const double
*wk9 , const double *wk13 , double *rhou1_old , double *rhou2_old , double *rhou0_old , double *rho_old , double
*rhoE_old , const double *rkold)
{
   rho_old[OPS_ACC8(0,0,0)] = deltat*rkold[0]*wk9[OPS_ACC3(0,0,0)] + rho_old[OPS_ACC8(0,0,0)];
   rhou0_old[OPS_ACC7(0,0,0)] = deltat*rkold[0]*wk10[OPS_ACC2(0,0,0)] + rhou0_old[OPS_ACC7(0,0,0)];
   rhou1_old[OPS_ACC5(0,0,0)] = deltat*rkold[0]*wk11[OPS_ACC0(0,0,0)] + rhou1_old[OPS_ACC5(0,0,0)];
   rhou2_old[OPS_ACC6(0,0,0)] = deltat*rkold[0]*wk12[OPS_ACC1(0,0,0)] + rhou2_old[OPS_ACC6(0,0,0)];
   rhoE_old[OPS_ACC9(0,0,0)] = deltat*rkold[0]*wk13[OPS_ACC4(0,0,0)] + rhoE_old[OPS_ACC9(0,0,0)];
}


 void taylor_green_vortex_block0_13_kernel(const double *rhou1 , const double *rhoE , const double *rho , const double
*rhou2 , const double *rhou0 , double *rhou1_old , double *rhou2_old , double *rhou0_old , double *rho_old , double
*rhoE_old)
{
   rho_old[OPS_ACC8(0,0,0)] = rho[OPS_ACC2(0,0,0)];
   rhou0_old[OPS_ACC7(0,0,0)] = rhou0[OPS_ACC4(0,0,0)];
   rhou1_old[OPS_ACC5(0,0,0)] = rhou1[OPS_ACC0(0,0,0)];
   rhou2_old[OPS_ACC6(0,0,0)] = rhou2[OPS_ACC3(0,0,0)];
   rhoE_old[OPS_ACC9(0,0,0)] = rhoE[OPS_ACC1(0,0,0)];
}


 void taylor_green_vortex_block0_14_kernel(double *rhou1 , double *rhoE , double *rho , double *rhou2 , double *rhou0 ,
const int *idx)
{
   double x = deltai0*idx[0];
   double y = deltai1*idx[1];
   double z = deltai2*idx[2];
   double u = sin(x)*cos(y)*cos(z);
   double v = -cos(x)*sin(y)*cos(z);
   double w = 0.0;
   double p = 1.0*rinv15 + 0.0625*(cos(2.0*x) + cos(2.0*y))*(2.0 + cos(2.0*z));
   double r = gama*pow(Minf, 2)*p;
   rho[OPS_ACC2(0,0,0)] = r;
   rhou0[OPS_ACC4(0,0,0)] = r*u;
   rhou1[OPS_ACC0(0,0,0)] = r*v;
   rhou2[OPS_ACC3(0,0,0)] = 0.0;
   rhoE[OPS_ACC1(0,0,0)] = rinv12*p + 0.5*r*(pow(u, 2) + pow(v, 2) + pow(w, 2));
}


void taylor_green_vortex_block0_15_kernel(double *rhou1 , double *rhoE , double *rho , double *rhou2 , double *rhou0)
{
   rho[OPS_ACC2(-1,0,0)] = rho[OPS_ACC2(1,0,0)];
   rho[OPS_ACC2(-2,0,0)] = rho[OPS_ACC2(2,0,0)];
   rhou0[OPS_ACC4(-1,0,0)] = rhou0[OPS_ACC4(1,0,0)];
   rhou0[OPS_ACC4(-2,0,0)] = rhou0[OPS_ACC4(2,0,0)];
   rhou1[OPS_ACC0(-1,0,0)] = rhou1[OPS_ACC0(1,0,0)];
   rhou1[OPS_ACC0(-2,0,0)] = rhou1[OPS_ACC0(2,0,0)];
   rhou2[OPS_ACC3(-1,0,0)] = rhou2[OPS_ACC3(1,0,0)];
   rhou2[OPS_ACC3(-2,0,0)] = rhou2[OPS_ACC3(2,0,0)];
   rhoE[OPS_ACC1(-1,0,0)] = rhoE[OPS_ACC1(1,0,0)];
   rhoE[OPS_ACC1(-2,0,0)] = rhoE[OPS_ACC1(2,0,0)];
}


void taylor_green_vortex_block0_16_kernel(double *rhou1 , double *rhoE , double *rho , double *rhou2 , double *rhou0)
{
   rho[OPS_ACC2(1,0,0)] = rho[OPS_ACC2(-1,0,0)];
   rho[OPS_ACC2(2,0,0)] = rho[OPS_ACC2(-2,0,0)];
   rhou0[OPS_ACC4(1,0,0)] = rhou0[OPS_ACC4(-1,0,0)];
   rhou0[OPS_ACC4(2,0,0)] = rhou0[OPS_ACC4(-2,0,0)];
   rhou1[OPS_ACC0(1,0,0)] = rhou1[OPS_ACC0(-1,0,0)];
   rhou1[OPS_ACC0(2,0,0)] = rhou1[OPS_ACC0(-2,0,0)];
   rhou2[OPS_ACC3(1,0,0)] = rhou2[OPS_ACC3(-1,0,0)];
   rhou2[OPS_ACC3(2,0,0)] = rhou2[OPS_ACC3(-2,0,0)];
   rhoE[OPS_ACC1(1,0,0)] = rhoE[OPS_ACC1(-1,0,0)];
   rhoE[OPS_ACC1(2,0,0)] = rhoE[OPS_ACC1(-2,0,0)];
}


void taylor_green_vortex_block0_17_kernel(double *rhou1 , double *rhoE , double *rho , double *rhou2 , double *rhou0)
{
   rho[OPS_ACC2(0,-1,0)] = rho[OPS_ACC2(0,1,0)];
   rho[OPS_ACC2(0,-2,0)] = rho[OPS_ACC2(0,2,0)];
   rhou0[OPS_ACC4(0,-1,0)] = rhou0[OPS_ACC4(0,1,0)];
   rhou0[OPS_ACC4(0,-2,0)] = rhou0[OPS_ACC4(0,2,0)];
   rhou1[OPS_ACC0(0,-1,0)] = rhou1[OPS_ACC0(0,1,0)];
   rhou1[OPS_ACC0(0,-2,0)] = rhou1[OPS_ACC0(0,2,0)];
   rhou2[OPS_ACC3(0,-1,0)] = rhou2[OPS_ACC3(0,1,0)];
   rhou2[OPS_ACC3(0,-2,0)] = rhou2[OPS_ACC3(0,2,0)];
   rhoE[OPS_ACC1(0,-1,0)] = rhoE[OPS_ACC1(0,1,0)];
   rhoE[OPS_ACC1(0,-2,0)] = rhoE[OPS_ACC1(0,2,0)];
}


void taylor_green_vortex_block0_18_kernel(double *rhou1 , double *rhoE , double *rho , double *rhou2 , double *rhou0)
{
   rho[OPS_ACC2(0,1,0)] = rho[OPS_ACC2(0,-1,0)];
   rho[OPS_ACC2(0,2,0)] = rho[OPS_ACC2(0,-2,0)];
   rhou0[OPS_ACC4(0,1,0)] = rhou0[OPS_ACC4(0,-1,0)];
   rhou0[OPS_ACC4(0,2,0)] = rhou0[OPS_ACC4(0,-2,0)];
   rhou1[OPS_ACC0(0,1,0)] = rhou1[OPS_ACC0(0,-1,0)];
   rhou1[OPS_ACC0(0,2,0)] = rhou1[OPS_ACC0(0,-2,0)];
   rhou2[OPS_ACC3(0,1,0)] = rhou2[OPS_ACC3(0,-1,0)];
   rhou2[OPS_ACC3(0,2,0)] = rhou2[OPS_ACC3(0,-2,0)];
   rhoE[OPS_ACC1(0,1,0)] = rhoE[OPS_ACC1(0,-1,0)];
   rhoE[OPS_ACC1(0,2,0)] = rhoE[OPS_ACC1(0,-2,0)];
}


void taylor_green_vortex_block0_19_kernel(double *rhou1 , double *rhoE , double *rho , double *rhou2 , double *rhou0)
{
   rho[OPS_ACC2(0,0,-1)] = rho[OPS_ACC2(0,0,1)];
   rho[OPS_ACC2(0,0,-2)] = rho[OPS_ACC2(0,0,2)];
   rhou0[OPS_ACC4(0,0,-1)] = rhou0[OPS_ACC4(0,0,1)];
   rhou0[OPS_ACC4(0,0,-2)] = rhou0[OPS_ACC4(0,0,2)];
   rhou1[OPS_ACC0(0,0,-1)] = rhou1[OPS_ACC0(0,0,1)];
   rhou1[OPS_ACC0(0,0,-2)] = rhou1[OPS_ACC0(0,0,2)];
   rhou2[OPS_ACC3(0,0,-1)] = rhou2[OPS_ACC3(0,0,1)];
   rhou2[OPS_ACC3(0,0,-2)] = rhou2[OPS_ACC3(0,0,2)];
   rhoE[OPS_ACC1(0,0,-1)] = rhoE[OPS_ACC1(0,0,1)];
   rhoE[OPS_ACC1(0,0,-2)] = rhoE[OPS_ACC1(0,0,2)];
}


void taylor_green_vortex_block0_20_kernel(double *rhou1 , double *rhoE , double *rho , double *rhou2 , double *rhou0)
{
   rho[OPS_ACC2(0,0,1)] = rho[OPS_ACC2(0,0,-1)];
   rho[OPS_ACC2(0,0,2)] = rho[OPS_ACC2(0,0,-2)];
   rhou0[OPS_ACC4(0,0,1)] = rhou0[OPS_ACC4(0,0,-1)];
   rhou0[OPS_ACC4(0,0,2)] = rhou0[OPS_ACC4(0,0,-2)];
   rhou1[OPS_ACC0(0,0,1)] = rhou1[OPS_ACC0(0,0,-1)];
   rhou1[OPS_ACC0(0,0,2)] = rhou1[OPS_ACC0(0,0,-2)];
   rhou2[OPS_ACC3(0,0,1)] = rhou2[OPS_ACC3(0,0,-1)];
   rhou2[OPS_ACC3(0,0,2)] = rhou2[OPS_ACC3(0,0,-2)];
   rhoE[OPS_ACC1(0,0,1)] = rhoE[OPS_ACC1(0,0,-1)];
   rhoE[OPS_ACC1(0,0,2)] = rhoE[OPS_ACC1(0,0,-2)];
}


#endif