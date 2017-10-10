/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @brief ops cuda specific runtime support functions
  * @author Gihan Mudalige
  * @details Implements cuda backend runtime support functions
  */

//
// header files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <ops_cuda_rt_support.h>
#include <ops_lib_core.h>

#include <vector>
#include <limits.h>

int ops_cyclic = 0;
/*__global__ void copy_kernel(char *dest, char *src, int size ) {
  int tid = blockIdx.x;
  memcpy(&dest[tid],&src[tid],size);
}*/

__global__ void copy_kernel_tobuf(char *dest, char *src, int rx_s, int rx_e,
                                  int ry_s, int ry_e, int rz_s, int rz_e,
                                  int x_step, int y_step, int z_step,
                                  int size_x, int size_y, int size_z,
                                  int buf_strides_x, int buf_strides_y,
                                  int buf_strides_z, int elem_size) {

  int idx_z = rz_s + z_step * (blockDim.z * blockIdx.z + threadIdx.z);
  int idx_y = ry_s + y_step * (blockDim.y * blockIdx.y + threadIdx.y);
  int idx_x = rx_s + x_step * (blockDim.x * blockIdx.x + threadIdx.x);

  if ((x_step == 1 ? idx_x < rx_e : idx_x > rx_e) &&
      (y_step == 1 ? idx_y < ry_e : idx_y > ry_e) &&
      (z_step == 1 ? idx_z < rz_e : idx_z > rz_e)) {

    src += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * elem_size;
    dest += ((idx_z - rz_s) * z_step * buf_strides_z +
             (idx_y - ry_s) * y_step * buf_strides_y +
             (idx_x - rx_s) * x_step * buf_strides_x) *
            elem_size;
    memcpy(dest, src, elem_size);
  }
}

__global__ void copy_kernel_frombuf(char *dest, char *src, int rx_s, int rx_e,
                                    int ry_s, int ry_e, int rz_s, int rz_e,
                                    int x_step, int y_step, int z_step,
                                    int size_x, int size_y, int size_z,
                                    int buf_strides_x, int buf_strides_y,
                                    int buf_strides_z, int elem_size) {

  int idx_z = rz_s + z_step * (blockDim.z * blockIdx.z + threadIdx.z);
  int idx_y = ry_s + y_step * (blockDim.y * blockIdx.y + threadIdx.y);
  int idx_x = rx_s + x_step * (blockDim.x * blockIdx.x + threadIdx.x);

  if ((x_step == 1 ? idx_x < rx_e : idx_x > rx_e) &&
      (y_step == 1 ? idx_y < ry_e : idx_y > ry_e) &&
      (z_step == 1 ? idx_z < rz_e : idx_z > rz_e)) {

    dest += (idx_z * size_x * size_y + idx_y * size_x + idx_x) * elem_size;
    src += ((idx_z - rz_s) * z_step * buf_strides_z +
            (idx_y - ry_s) * y_step * buf_strides_y +
            (idx_x - rx_s) * x_step * buf_strides_x) *
           elem_size;
    memcpy(dest, src, elem_size);
  }
}

void ops_halo_copy_tobuf(char *dest, int dest_offset, ops_dat src, int rx_s,
                         int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                         int x_step, int y_step, int z_step, int buf_strides_x,
                         int buf_strides_y, int buf_strides_z) {

  dest += dest_offset;
  int thr_x = abs(rx_s - rx_e);
  int blk_x = 1;
  if (abs(rx_s - rx_e) > 8) {
    blk_x = (thr_x - 1) / 8 + 1;
    thr_x = 8;
  }
  int thr_y = abs(ry_s - ry_e);
  int blk_y = 1;
  if (abs(ry_s - ry_e) > 8) {
    blk_y = (thr_y - 1) / 8 + 1;
    thr_y = 8;
  }
  int thr_z = abs(rz_s - rz_e);
  int blk_z = 1;
  if (abs(rz_s - rz_e) > 8) {
    blk_z = (thr_z - 1) / 8 + 1;
    thr_z = 8;
  }

  dim3 grid(blk_x, blk_y, blk_z);
  dim3 tblock(thr_x, thr_y, thr_z);
  copy_kernel_tobuf<<<grid, tblock>>>(
      dest, src->data_d, rx_s, rx_e, ry_s, ry_e, rz_s, rz_e, x_step, y_step,
      z_step, src->size[0], src->size[1], src->size[2], buf_strides_x,
      buf_strides_y, buf_strides_z, src->elem_size);

  // TODO: MPI buffers and GPUDirect
}

void ops_halo_copy_frombuf(ops_dat dest, char *src, int src_offset, int rx_s,
                           int rx_e, int ry_s, int ry_e, int rz_s, int rz_e,
                           int x_step, int y_step, int z_step,
                           int buf_strides_x, int buf_strides_y,
                           int buf_strides_z) {

  src += src_offset;
  int thr_x = abs(rx_s - rx_e);
  int blk_x = 1;
  if (abs(rx_s - rx_e) > 8) {
    blk_x = (thr_x - 1) / 8 + 1;
    thr_x = 8;
  }
  int thr_y = abs(ry_s - ry_e);
  int blk_y = 1;
  if (abs(ry_s - ry_e) > 8) {
    blk_y = (thr_y - 1) / 8 + 1;
    thr_y = 8;
  }
  int thr_z = abs(rz_s - rz_e);
  int blk_z = 1;
  if (abs(rz_s - rz_e) > 8) {
    blk_z = (thr_z - 1) / 8 + 1;
    thr_z = 8;
  }

  dim3 grid(blk_x, blk_y, blk_z);
  dim3 tblock(thr_x, thr_y, thr_z);
  copy_kernel_frombuf<<<grid, tblock>>>(
      dest->data_d, src, rx_s, rx_e, ry_s, ry_e, rz_s, rz_e, x_step, y_step,
      z_step, dest->size[0], dest->size[1], dest->size[2], buf_strides_x,
      buf_strides_y, buf_strides_z, dest->elem_size);
  dest->dirty_hd = 2;
}


__global__ void toucher(char *dat, int size, double fac) {
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  if (id < size) {
    char val = dat[id];
    if (fac == 0) val = val + 1;
    if (fac > 1) val = val * fac;
    else val = val * (char)fac;
    dat[id] = val;
  }
}
extern "C" {
void ops_touch(char *dat, int size, double fac) {
  int nthreads = 1024;
  int nblocks = (size-1)/nthreads+1;
  toucher<<<nblocks,nthreads>>>(dat,size,fac);
}
}

int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

inline int intersection(int range1_beg, int range1_end, int range2_beg,
                 int range2_end, int *intersect_begin) {
  if (range1_beg >= range1_end || range2_beg >= range2_end) return 0;
  int i_min = MAX(range1_beg, range2_beg);
  int i_max = MIN(range1_end, range2_end);
  *intersect_begin = i_min;
  return i_max > i_min ? i_max - i_min : 0;
}


struct trans {
  int ID;
  ops_dat dat;
  int tile;
  int slot;
  int range_beg;
  int range_end;
  long hptr_begin;
  long hptr_end;
  long dptr_begin;
  long dptr_end;
};
#define E_UP 0
#define E_DOWN 1
#define E_COMP 2
#include <vector>
std::vector<trans> uploads(0);
std::vector<trans> downloads(0);
std::vector<trans> compute(0);
int upctr = 0;
int downctr = 10000;
int compctr = 20000;

void add_trans_entry(int type, int ID, ops_dat dat, int tile, int slot, int range_beg, int range_end, long hptr_begin, long hptr_end, long dptr_begin, long dptr_end) {
  trans t;
  t.ID = ID;
  if (type == E_UP && ID>=10000) printf("Error transaction log overflow\n");
  if (type == E_DOWN && ID>=20000) printf("Error transaction log overflow\n");
  t.dat = dat;
  t.tile = tile;
  t.slot = slot;
  t.range_beg = range_beg;
  t.range_end = range_end;
  t.hptr_begin = hptr_begin;
  t.hptr_end = hptr_end;
  t.dptr_begin = dptr_begin;
  t.dptr_end = dptr_end;
  if (type == E_UP)
    uploads.push_back(t);
  else if (type == E_DOWN)
    downloads.push_back(t);
  else
    compute.push_back(t);
}

void remove_trans(int ID, int type) {
  std::vector<trans>::iterator i;
  if (type == E_UP) {
    for (i = uploads.begin(); i != uploads.end();) {
      if (i->ID == ID) i = uploads.erase(i);
      else ++i;
    }
  } else if (type == E_DOWN) {
    for (i = downloads.begin(); i != downloads.end();) {
      if (i->ID == ID) i = downloads.erase(i);
      else ++i;
    }
  } else {
    for (i = compute.begin(); i != compute.end();) {
      if (i->ID == ID) i = compute.erase(i);
      else ++i;
    }
  }
}

void check_trans(int ID, int type) {
  std::vector<trans>& from = (type == E_UP ? uploads : (type==E_DOWN ? downloads : compute));

  for (int i = 0; i < from.size(); i++) {
    for (int arr = 0; arr < 3; arr++) {
      std::vector<trans>& to = (arr == E_UP ? uploads : (arr==E_DOWN ? downloads : compute));
      for (int j = 0; j < to.size(); j++) {
        int intersect_begin;
        int intersect_len = intersection(from[i].dptr_begin,from[i].dptr_end,
            to[j].dptr_begin,to[j].dptr_end,&intersect_begin);
        int intersect_len2 = intersection(from[i].hptr_begin,from[i].hptr_end,
            to[j].hptr_begin,to[j].hptr_end,&intersect_begin);
        if (from[i].ID == ID && from[i].ID != to[j].ID && from[i].dat->index == to[j].dat->index && (intersect_len > 0 || intersect_len2 > 0)) {
          printf("Error: %s new %d (%d) slot %d intersecting with old %d (%d) slot %d: device %ld-%ld vs %ld-%ld host %ld-%ld vs %ld-%ld\n",
              from[i].dat->name, type, ID, from[i].slot,
              arr, to[j].ID, to[j].slot, 
              from[i].dptr_begin,from[i].dptr_end,
                to[j].dptr_begin,to[j].dptr_end,
              from[i].hptr_begin,from[i].hptr_end,
                to[j].hptr_begin,to[j].hptr_end);
        }
      }
    }
  }
}

struct datasets {
  int size[OPS_MAX_DIM];
  int base_offset;
  int max_width;
  int curr_slot;
  int curr_chunk[2];
  int actually_uploaded;
  ops_dat dat;
  long bytes;
  long curr_offset;
  long last_offset;
  long curr_size;
  long last_size;
  long copy_from;
  long copy_amount;
};

std::vector<datasets> dats(0);

cudaStream_t stream_copy_up = 0;
cudaStream_t stream_copy_down = 0;
cudaStream_t stream_compute = 0;
cudaStream_t stream = 0;

int upload_me(int idx) {
  //return !(idx>=5 && idx <= 31);
  //return !((idx>=5 && idx <= 31) || idx <=-1 || idx == 1 || idx == 3);
  return 0;
}

//TODO: v1 seems okay 
void ops_get_offsets_deprange(long &base_ptr, long &end_ptr, ops_dat dat, std::vector<std::vector<int> > &dependency_ranges, int tile, int num_tiles, int lrf, long &delta) {

  //calculate base_offset without last used dimension
  long line_begin_offset = 0;
  long cumsize = 1;
  for (int i = 0; i < dat->block->dims-1; i++) {
    line_begin_offset +=
        dat->elem_size * cumsize * (-dat->base[i] - dat->d_m[i]); //TODO: different for MPI
    cumsize *= dats[dat->index].size[i];
  }
  line_begin_offset = dats[dat->index].base_offset - line_begin_offset;
  base_ptr = line_begin_offset; //go back to beginning of x line in 2D or last x-y plane in 3D
  end_ptr = line_begin_offset; //go back to beginning of x line in 2D or last x-y plane in 3D
  //For first n-1 dimensions, we copy all of them
  long prod = dat->elem_size;
  for (int d = 0; d < dat->block->dims-1; d++) {
    prod *= dats[dat->index].size[d];
  }
  //For the nth dimension we just copy up to where we need it
  int d = dat->block->dims -1;
  int prevrange = dependency_ranges[dat->index][mod(tile-1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 1] - dependency_ranges[dat->index][mod(tile-1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 0];
  int nextrange = dependency_ranges[dat->index][mod(tile+1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 1] - dependency_ranges[dat->index][mod(tile+1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 0];
  //Left or full - start of this tile
  if (lrf == 0 || lrf == 2 || tile == 0 || dat->size[d] == 1 || prevrange == 0)
    base_ptr += dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * d + 0] * prod;
  else //right - end of previous tile
    base_ptr += MAX(dependency_ranges[dat->index][mod(tile-1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 1],
                    dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * d + 0]) * prod;
  //Right or full - end of this tile
  if (lrf == 1 || lrf ==2 || tile == num_tiles-1 || dat->size[d] == 1 || nextrange == 0) 
    end_ptr  += dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * d + 1] * prod;
  else //left - start of next tile
    end_ptr  += MIN(dependency_ranges[dat->index][mod(tile+1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 0],
                    dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * d + 1]) * prod;

  if (dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * d + 0] 
      == dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * d + 1]) end_ptr = base_ptr; //zero dependency

  //I need to offset the beginning fo tile 0, so tile 3 (laoding to slot 0) won't bite tile 1's tail
  //extra space needed is the difference between largest tile and left range of tile 0
  if (num_tiles > 1 && tile == 0 && nextrange > 1 && dependency_ranges[dat->index][0 * 2 * OPS_MAX_DIM + 2 * d + 1] - dependency_ranges[dat->index][0 * 2 * OPS_MAX_DIM + 2 * d + 0] > 1)
    delta = MAX(2,(dats[dat->index].max_width - dependency_ranges[dat->index][1 * 2 * OPS_MAX_DIM + 2 * d + 0] + dependency_ranges[dat->index][0 * 2 * OPS_MAX_DIM + 2 * d + 0]))*prod;
  else delta = 0;

  if (end_ptr < base_ptr) {printf("WARNING: overreaching depranges! Please check, shouldn't happen\n%s %ld-%ld, dep range: %d-%d prev %d - %d next start %d\n",dat->name, base_ptr, end_ptr, dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * d + 0], dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * d + 1],dependency_ranges[dat->index][mod(tile-1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 0],dependency_ranges[dat->index][mod(tile-1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 1], dependency_ranges[dat->index][mod(tile+1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 0]); end_ptr = base_ptr;} //zero ranges
}

void ops_tiling_init_streams() {
  cutilSafeCall(cudaStreamCreateWithFlags(&stream_copy_down,cudaStreamNonBlocking));
  cutilSafeCall(cudaStreamCreateWithFlags(&stream_copy_up,cudaStreamNonBlocking));
  int leastPriority, greatestPriority;
  cudaDeviceGetStreamPriorityRange ( &leastPriority, &greatestPriority );
  cutilSafeCall(cudaStreamCreateWithPriority(&stream_compute,cudaStreamNonBlocking,greatestPriority));
  stream = stream_compute;
}

void ops_tiling_datastructures_init(int size) {
  dats.resize(size);
  for (int i = 0; i < dats.size(); i++) {
    dats[i].bytes = 0;
    dats[i].dat = NULL;
  }
  ops_dat_entry *item, *tmp_item;
  for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    dats[item->dat->index].dat = item->dat;
    dats[item->dat->index].base_offset = item->dat->base_offset;
    memcpy(dats[item->dat->index].size, item->dat->size, sizeof(int)*OPS_MAX_DIM);
  }
}

int deviceIdUM = -1;
cudaEvent_t e1, e2;

void ops_prepare_tile_managed(int tile, int total_tiles, std::vector<std::vector<int> > &tiled_ranges, std::vector<std::vector<int> > &dependency_ranges, std::vector<int> &datasets_access_type) {
  int first = 0;
  //First time
  if (deviceIdUM == -1) {
    first = 1;
    cutilSafeCall(cudaGetDevice(&deviceIdUM));
    ops_tiling_init_streams();
    ops_tiling_datastructures_init(dependency_ranges.size());
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
  }
  cudaEventSynchronize(e1);
  cudaEventSynchronize(e2);
  if (tile == 0 && first) {
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_dat dat = item->dat;
      long base_ptr, end_ptr, delta;
      ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, tile, total_tiles, 2, delta); //Full
      if (end_ptr > base_ptr)
        cutilSafeCall(cudaMemPrefetchAsync(dat->data+base_ptr,end_ptr-base_ptr,deviceIdUM,stream));
    }
    cutilSafeCall(cudaStreamSynchronize(stream));
  }
}

void ops_finish_tile_managed(int tile, int total_tiles, std::vector<std::vector<int> > &tiled_ranges, std::vector<std::vector<int> > &dependency_ranges, std::vector<int> &datasets_access_type) {
  cudaEventRecord(e1, stream);
  int next_tile = (tile+1)%total_tiles;
  int prev_tile = mod(tile-1,total_tiles);
  ops_dat_entry *item, *tmp_item;
  for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    ops_dat dat = item->dat;
    cudaStreamSynchronize(stream_copy_up);
    {
      long base_ptr, end_ptr, delta;
      ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, next_tile, total_tiles, next_tile == 0 ? 2 : 1, delta); //Right
      if (end_ptr > base_ptr)
        cutilSafeCall(cudaMemPrefetchAsync(dat->data+base_ptr,end_ptr-base_ptr,deviceIdUM,stream_copy_up));
      else {
        long base_ptr, end_ptr, delta;
        ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, (next_tile+1)%total_tiles, total_tiles, (next_tile+1)%total_tiles == 0 ? 2 : 1, delta); //Right
        if (end_ptr > base_ptr)
          cutilSafeCall(cudaMemPrefetchAsync(dat->data+base_ptr,end_ptr-base_ptr,deviceIdUM,stream_copy_up));
      }
    }
      long base_ptr, end_ptr, delta;
      //ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, tile, total_tiles, tile == total_tiles -1 ? 2 : 0, delta); //Left
      ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, prev_tile, total_tiles, prev_tile == total_tiles -1 ? 2 : 0, delta); //Left
      if (end_ptr > base_ptr)
        cutilSafeCall(cudaMemPrefetchAsync(dat->data+base_ptr,end_ptr-base_ptr,cudaCpuDeviceId,stream));
  }
  cudaEventRecord(e2, stream_copy_up);
/*  if (tile == total_tiles-1) {
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_dat dat = item->dat;
      int idx = dat->index;
      cudaStreamSynchronize(stream_copy_up);
      long base_ptr, end_ptr, delta;
      ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, tile, total_tiles, 2, delta); //Left
      if (end_ptr > base_ptr)
        cutilSafeCall(cudaMemPrefetchAsync(dat->data+base_ptr,end_ptr-base_ptr,cudaCpuDeviceId,stream));
    }
  }*/
  // rotate streams and swap events
  cudaStream_t st;
  cudaEvent_t et;
  st = stream; stream = stream_copy_up; stream_copy_up = st;
  st = stream_copy_up; stream_copy_up = stream_copy_down; stream_copy_down = st;
  et = e1; e1 = e2; e2 = et;
}

void ops_prepare_tile(int tile, int total_tiles, std::vector<std::vector<int> > &tiled_ranges, std::vector<std::vector<int> > &dependency_ranges, std::vector<int> &datasets_access_type) {

  if (ops_managed) {ops_prepare_tile_managed(tile, total_tiles, tiled_ranges, dependency_ranges, datasets_access_type); return;}
  if (tile == 0) {
    cutilSafeCall(cudaStreamSynchronize(stream_copy_up));
    cutilSafeCall(cudaStreamSynchronize(stream_compute));
//    remove_trans(compctr,E_COMP);
//    remove_trans(upctr,E_UP);

    //First time ever - initialise
    if (dats.size()==0) {
      ops_tiling_datastructures_init(dependency_ranges.size());
      ops_tiling_init_streams();
    }

    //determine biggest dependency range for each dataset to allocate scratch memory on GPU
    ops_dat_entry *item, *tmp_item;
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      int idx = item->dat->index;
      int maxsize = 0;
      for (int t = 0; t < total_tiles; t++) {
        int d = item->dat->block->dims-1; //Only for last dimension
        maxsize = MAX(maxsize,dependency_ranges[idx][t * 2 * OPS_MAX_DIM + 2 * d + 1]
                            - dependency_ranges[idx][t * 2 * OPS_MAX_DIM + 2 * d + 0]);
      }
      dats[idx].max_width = maxsize;
      //Allocate it a little larger, if not edge dat in this dim (or just unused)
      if (maxsize > 1 && dats[idx].bytes == 0) maxsize += 17; 

      // total required memory
      long cum_size = item->dat->elem_size; 
      for (int d = 0; d < item->dat->block->dims-1; d++) cum_size *= item->dat->size[d];
      cum_size *= maxsize;

      //3 slots
      cum_size *= 3;
      if (cum_size > dats[idx].bytes) {
        //printf("Reallocating memory for %s: %ld->%ld\n",item->dat->name, dats[idx].bytes,cum_size);
        cutilSafeCall(cudaStreamSynchronize(stream_copy_down)); //Need to make sure all previous copies finished before dealloc
        cutilSafeCall(cudaFree(item->dat->data_d));
        cutilSafeCall(cudaMalloc(&item->dat->data_d, cum_size));
        dats[idx].bytes = cum_size;
        dats[idx].curr_slot = 2; //last used slot, so next one is 0
        dats[idx].curr_chunk[0] = 0;
        dats[idx].curr_chunk[1] = 0;
        dats[idx].curr_offset = 0;
        dats[idx].last_offset = 0;
        dats[idx].last_size = 0;
        dats[idx].curr_size = 0;
        dats[idx].copy_from = 0;
        dats[idx].copy_amount = 0;
        dats[idx].actually_uploaded = 0;
      }
    }
  }

  //Create event that we can sync on an the end to make sure previous copies have finished 
  cudaEvent_t e_copyup;
  cudaEventCreate(&e_copyup);
  if (tile != 0)
    cudaEventRecord(e_copyup, stream_copy_up);

  //Copy required data to GPU
  //TODO: assert that only last dim is tiled
  //TODO: SoA: set dat->size[dat->block->dims-1]
  ops_dat_entry *item, *tmp_item;
  if (tile == 0) {
//#define NOPREFETCH
#ifdef NOPREFETCH
    upctr++; 
#endif
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_dat dat = item->dat;
      int idx = dat->index;

      //Determine data to required on to the GPU
      long base_ptr, end_ptr, delta;
      ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, tile, total_tiles, 2, delta); //Full
      //If no data needed
      if (end_ptr-base_ptr == 0) {
        continue;
      }
      long slice_size = (end_ptr - base_ptr) / (dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1]-
              dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0]);
      int intersect_begin;
      int intersect_len = intersection(dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0],
                                       dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1],
                                       dats[idx].curr_chunk[0],
                                       dats[idx].curr_chunk[1],&intersect_begin);
      //If already uploaded
      if (intersect_begin == dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0] && 
           intersect_len == (dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1]-
                             dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0]) &&
          //either needs upload and uploaded, or doesn't need upload
          (((datasets_access_type[idx] > 0 || (datasets_access_type[idx] == 0 && upload_me(idx))) &&  dats[idx].actually_uploaded == 1)
           ||(datasets_access_type[idx] == 0 && !upload_me(idx)) )) {

        //if the original started before this one
        if (dats[idx].curr_chunk[0] < dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0]) {
          dats[idx].curr_offset += (dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0]-dats[idx].curr_chunk[0]) * slice_size;
        }
        //if the original was longer
        dats[idx].curr_size = end_ptr-base_ptr;
        //update for this chunk
        dats[idx].curr_chunk[0] = dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0];
        dats[idx].curr_chunk[1] = dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1];
        continue;
      };

      //If no intersection, start upload into next slot
      if (intersect_len == 0
#ifdef NOPREFETCH
                    || true
#endif
                    ) {
        dats[idx].curr_slot = mod(dats[idx].curr_slot+1,3);
        dats[idx].last_offset = dats[idx].curr_offset;
        dats[idx].curr_offset = (dats[idx].bytes/3) * dats[idx].curr_slot;
        if (dats[idx].curr_offset + end_ptr-base_ptr > (dats[idx].bytes/3) * (dats[idx].curr_slot+1)) printf("Error, out of bounds copy for %s in tile==0: copying tile %d to slot %d: %p+%ld size %ld, but size is %ld\n",dat->name, tile, dats[idx].curr_slot, dat->data_d, dats[idx].curr_offset, end_ptr - base_ptr, (dats[idx].bytes/3) * (dats[idx].curr_slot+1));
        if (datasets_access_type[idx] > 0 || (datasets_access_type[idx] == 0 && upload_me(idx))) { //read first 
          //printf("Tile 0 fetching to NEW slot %d Copying %s from %p+%ld to %p+%ld (%p-%p), size %ld, delta %ld\n", dats[idx].curr_slot, dat->name, dat->data, base_ptr, dat->data_d,dats[idx].curr_offset, dat->data_d, dat->data_d + dats[idx].bytes, end_ptr-base_ptr,delta);
          dats[idx].actually_uploaded = 1;
          cutilSafeCall(cudaMemcpyAsync(dat->data_d+dats[idx].curr_offset, dat->data + base_ptr, end_ptr - base_ptr, cudaMemcpyHostToDevice, stream_copy_up));
//          add_trans_entry(E_UP, upctr, dat, tile, dats[idx].curr_slot,
//              dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0],
//              dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1],
//              base_ptr, end_ptr, dats[idx].curr_offset, dats[idx].curr_offset+end_ptr-base_ptr);
        } else dats[idx].actually_uploaded = 0;
      //if not actually uploaded 
      } else if (dats[idx].actually_uploaded == 0 && (datasets_access_type[idx] > 0 || (datasets_access_type[idx] == 0 && upload_me(idx)))) {
          dats[idx].actually_uploaded = 1;
          cutilSafeCall(cudaMemcpyAsync(dat->data_d+dats[idx].curr_offset, dat->data + base_ptr, end_ptr - base_ptr, cudaMemcpyHostToDevice, stream_copy_up));
//          add_trans_entry(E_UP, upctr, dat, tile, dats[idx].curr_slot,
//              dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0],
//              dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1],
//              base_ptr, end_ptr, dats[idx].curr_offset, dats[idx].curr_offset+end_ptr-base_ptr);
      } //if partly uploaded
      else {
        //Missing some in the beginning
        if (dats[idx].curr_chunk[0] > dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0]) {
          long extra_data = slice_size * (dats[idx].curr_chunk[0] - dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0]);
          dats[idx].curr_offset -= extra_data;
          dats[idx].curr_size += extra_data;
          if (dats[idx].curr_offset < 0 || (dats[idx].curr_slot>0 && dats[idx].curr_offset < (dats[idx].last_offset + dats[idx].last_size)))
            printf("Error: missing left side of tile 0 overwriting previous tile or offset < 0\n");
          if (datasets_access_type[idx] > 0 || (datasets_access_type[idx] == 0 && upload_me(idx))) //read first
            cutilSafeCall(cudaMemcpyAsync(dat->data_d+dats[idx].curr_offset, dat->data + base_ptr, extra_data, cudaMemcpyHostToDevice, stream_copy_up));
        }
        //Missing some in the end
        if (dats[idx].curr_chunk[1] < dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1]) {
          //if the speculatively prefetched started before this one
          if (dats[idx].curr_chunk[0] < dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0]) {
            dats[idx].curr_offset += (dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0]-dats[idx].curr_chunk[0]) * slice_size;
            dats[idx].curr_size -= (dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0]-dats[idx].curr_chunk[0]) * slice_size;
          }
          long extra_data = slice_size * (dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1] - dats[idx].curr_chunk[1]);          
          if (dats[idx].curr_offset + (end_ptr - base_ptr) > (dats[idx].bytes/3) * (dats[idx].curr_slot+1)) 
            printf("Error: missing right side too large\n");
          if (datasets_access_type[idx] > 0 || (datasets_access_type[idx] == 0 && upload_me(idx))) //read first
            cutilSafeCall(cudaMemcpyAsync(dat->data_d+dats[idx].curr_offset+dats[idx].curr_size, dat->data + end_ptr - extra_data, extra_data, cudaMemcpyHostToDevice, stream_copy_up));
        }
      }
      dats[idx].last_size = dats[idx].curr_size;
      dats[idx].curr_size = end_ptr-base_ptr;
      dats[idx].curr_chunk[0] = dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0];
      dats[idx].curr_chunk[1] = dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1];
      dats[idx].copy_from = 0;
      dats[idx].copy_amount = 0;
    }
    cudaEventRecord(e_copyup, stream_copy_up);
//    check_trans(upctr,E_UP);

  }
#ifdef NOPREFETCH
  if (tile != total_tiles-1) {
  if (tile == 0) {cutilSafeCall(cudaStreamSynchronize(stream_copy_down)); /*remove_trans(downctr,E_DOWN);*/}
#endif
  upctr++;
  //upload next tile - leave a few rows spare if next tile's dependency range is bigger
  for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    ops_dat dat = item->dat;
    int idx = dat->index;

    //For edge thats I do not need to upload again
    //TODO: v1
    if (dat->size[dat->block->dims-1] == 1) continue;

    //Upload next tile
    int next_tile = mod(tile+1,total_tiles);
    long base_ptr, end_ptr, delta;
    ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, next_tile, total_tiles, next_tile == 0 ? 2 : 1, delta); //Right
    if (end_ptr-base_ptr == 0) {
      continue;
    }
    //
    dats[idx].curr_slot = mod(dats[idx].curr_slot+1,3);
    //Compute Full range, then right begin - full begin is the extra offset
    long base_ptr2, end_ptr2;
    ops_get_offsets_deprange(base_ptr2, end_ptr2, dat, dependency_ranges, next_tile, total_tiles, 2, delta); //Full
    long extra_offset = base_ptr - base_ptr2;
    dats[idx].last_offset = dats[idx].curr_offset;
    if (next_tile == 0) {
      long slice_size = (end_ptr - base_ptr) / (dependency_ranges[idx][next_tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1]-
          dependency_ranges[idx][next_tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0]);
      dats[idx].curr_offset = (dats[idx].bytes/3) * dats[idx].curr_slot + 2*slice_size;
    } else
      dats[idx].curr_offset = (dats[idx].bytes/3) * dats[idx].curr_slot;
    dats[idx].last_size = dats[idx].curr_size;
    dats[idx].curr_size = end_ptr - base_ptr2; //Full size
    if (dats[idx].curr_offset + end_ptr-base_ptr2 > (dats[idx].bytes/3)*(dats[idx].curr_slot+1)) printf("Error, out of bounds copy for %s: copying tile %d to slot %d: %p+%ld size %ld, but size is %ld\n",dat->name, next_tile, dats[idx].curr_slot, dat->data_d, dats[idx].curr_offset, end_ptr - base_ptr2, (dats[idx].bytes/3)*(dats[idx].curr_slot+1));
    if (datasets_access_type[idx] > 0 || (datasets_access_type[idx] == 0 && upload_me(idx))) { //read first 
      //printf("Prefetching tile %d to slot %d Copying %s from %p+%ld to %p+%ld (%p-%p), size %ld, delta %ld\n", next_tile, dats[idx].curr_slot, dat->name, dat->data, base_ptr, dat->data_d, dats[idx].curr_offset + extra_offset, dat->data_d, dat->data_d + dats[idx].bytes, end_ptr-base_ptr, 0);
      dats[idx].actually_uploaded = 1;
      cutilSafeCall(cudaMemcpyAsync(dat->data_d + dats[idx].curr_offset + extra_offset,
            dat->data + base_ptr, end_ptr - base_ptr, cudaMemcpyHostToDevice, stream_copy_up));
//      add_trans_entry(E_UP, upctr, dat, next_tile, 0,
//          dependency_ranges[idx][next_tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0],
//          dependency_ranges[idx][next_tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1],
//          base_ptr, end_ptr, dats[idx].curr_offset + extra_offset, dats[idx].curr_offset+end_ptr-base_ptr);
    } else dats[idx].actually_uploaded = 0;
    if (next_tile == 0) { //Speculative prefetch
      dats[idx].copy_from = 0;
      dats[idx].copy_amount = 0;
    } else {
      dats[idx].copy_from = dats[idx].last_offset + dats[idx].last_size - extra_offset;
      dats[idx].copy_amount = extra_offset;
    }
    dats[idx].curr_chunk[0] = dependency_ranges[dat->index][next_tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0];
    dats[idx].curr_chunk[1] = dependency_ranges[dat->index][next_tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1];
  }
//  check_trans(upctr,E_UP); 
#ifdef NOPREFETCH
    }
#endif

  compctr++;
  //alter base_offset so that it is offset by the dependency range for the current tile
  for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    ops_dat dat = item->dat;
      long base_ptr, end_ptr, delta;
      ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, tile, total_tiles, 2, delta); //Full
      dat->base_offset = dats[dat->index].base_offset - base_ptr +
#ifdef NOPREFETCH
           (tile == total_tiles-1 ? dats[dat->index].curr_offset : dats[dat->index].last_offset);
#else
           dats[dat->index].last_offset;
#endif
/*      long base_ptr, end_ptr, delta;
      ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, tile, total_tiles, 2, delta); //Full
      int curr_slot = 
#ifdef NOPREFETCH
        (tile == total_tiles-1) ? dats[dat->index].curr_slot : mod(dats[dat->index].curr_slot-1,3);
#else
        mod(dats[idx].curr_slot-1,3);
#endif
    add_trans_entry(E_COMP, compctr, dat, tile, curr_slot,
        dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0],
        dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1],
        base_ptr, end_ptr, dat->base_offset+base_ptr, dat->base_offset+end_ptr-base_ptr);*/
  }

  //Before actually starting the computations, make sure previous copies up finished
  cudaEventSynchronize(e_copyup);
  cudaEventDestroy(e_copyup);
//  int prev_up_idx = 
//#ifdef NOPREFETCH
//    (tile == total_tiles-1) ? upctr : (upctr - 1);
//#else
//    upctr - 1;
//#endif
//  remove_trans(prev_up_idx, E_UP);
//  check_trans(compctr,E_COMP); 
}
void ops_finish_tile(int tile, int total_tiles, std::vector<std::vector<int> > &tiled_ranges, std::vector<std::vector<int> > &dependency_ranges, std::vector<int> &datasets_access_type) {
  if (ops_managed) {ops_finish_tile_managed(tile, total_tiles, tiled_ranges, dependency_ranges, datasets_access_type); return;}
  cudaEvent_t e_copydown;
  cudaEventCreate(&e_copydown);
  cudaEventRecord(e_copydown, stream_copy_down);

  //Wait for compute to finish before downloading results
  cutilSafeCall(cudaStreamSynchronize(stream_compute));
//  remove_trans(compctr,E_COMP);
  
  downctr++;
  //TODO: do not copy back read-only data
  ops_dat_entry *item, *tmp_item;
  for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    ops_dat dat = item->dat;
    int idx = dat->index;
    //Restore properties
    if (tile == total_tiles-1)
      dat->base_offset = dats[dat->index].base_offset;

    //Skip edge dats, those are managed by the first tile
    if (tile > 0 && dat->size[dat->block->dims-1] == 1) continue;

    //Copy over the right edge of this tile in the previous slot, to the left of the next slot
    if (dats[idx].copy_amount > 0) {
      if (dats[idx].copy_from < 0 || dats[idx].copy_amount < 0 || dats[idx].copy_from + dats[idx].copy_amount > dats[idx].bytes) printf("Error: right edge to start overreach %s: from %ld to %ld, size %ld\n",dats[idx].dat->name,dats[idx].copy_from,dats[idx].copy_from+dats[idx].copy_amount,dats[idx].bytes);
      long toptr = dats[idx].curr_offset; 
      //printf("Copying %s end->start from %ld size %ld to %ld\n",dat->name, dats[idx].copy_from, dats[idx].copy_amount, toptr);
      cutilSafeCall(cudaMemcpyAsync(dat->data_d + toptr, dat->data_d+dats[idx].copy_from, dats[idx].copy_amount, cudaMemcpyDeviceToDevice, stream_compute));
      dats[idx].copy_from = 0;
      dats[idx].copy_amount = 0;
    }
    //Determine data to be copied off to the CPU
    long base_ptr, end_ptr, delta;
    ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, tile, total_tiles, (tile == total_tiles-1 ? 2 : 0),delta); //Left or full if last tile
    //Where the data being processed starts
    long base_ptr_gpu = 
#ifdef NOPREFETCH
      tile == total_tiles-1 ? dats[idx].curr_offset : dats[idx].last_offset;
#else
      dats[idx].last_offset;
#endif
    if (datasets_access_type[idx] == 0 && !upload_me(idx) && ops_cyclic) continue;
    if (datasets_access_type[idx] != 1  && end_ptr-base_ptr>0) { //not read only
      //printf("Tile %d copying back %s to %p+%ld from %p+%ld, size %ld\n", tile, dat->name, dat->data, base_ptr, dat->data_d, base_ptr_gpu,end_ptr-base_ptr);
      cutilSafeCall(cudaMemcpyAsync(dat->data + base_ptr, dat->data_d+base_ptr_gpu, end_ptr - base_ptr, cudaMemcpyDeviceToHost, stream_copy_down));
//      int curr_slot = 
//#ifdef NOPREFETCH
//        (tile == total_tiles-1) ? dats[idx].curr_slot : mod(dats[idx].curr_slot-1,3);
//#else
//        mod(dats[idx].curr_slot-1,3);
//#endif
//      add_trans_entry(E_DOWN, downctr, dat, tile, curr_slot,
//          dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0],
//          dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1],
//          base_ptr, end_ptr, base_ptr_gpu, base_ptr_gpu+end_ptr-base_ptr);
    }
  }
//  check_trans(downctr,E_DOWN); 
  
  //Wait for previous round of copies to finish
  cudaEventSynchronize(e_copydown);
  cudaEventDestroy(e_copydown);
//  remove_trans(downctr-1,E_DOWN);
}

