//
// auto-generated by ops.py
//

#define OPS_GPU

int xdim0_set_val;
int ydim0_set_val;


#undef OPS_ACC0


#define OPS_ACC0(x,y,z) (x+xdim0_set_val*(y)+xdim0_set_val*ydim0_set_val*(z))

//user function
inline 
void set_val(double *dat, const double *val)
{

    dat[OPS_ACC0(0,0,0)] = *val;
}


#undef OPS_ACC0



void set_val_c_wrapper(
  double *p_a0,
  double p_a1,
  int x_size, int y_size, int z_size) {
  #ifdef OPS_GPU
  #pragma acc parallel deviceptr(p_a0)
  #pragma acc loop
  #endif
  for ( int n_z=0; n_z<z_size; n_z++ ){
    #ifdef OPS_GPU
    #pragma acc loop
    #endif
    for ( int n_y=0; n_y<y_size; n_y++ ){
      #ifdef OPS_GPU
      #pragma acc loop
      #endif
      for ( int n_x=0; n_x<x_size; n_x++ ){
        set_val(  p_a0 + n_x*1*1 + n_y*xdim0_set_val*1*1 + n_z*xdim0_set_val*ydim0_set_val*1*1,
           &p_a1 );

      }
    }
  }
}
