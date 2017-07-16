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

struct datasets {
  long bytes;
  ops_dat dat;
  int size[OPS_MAX_DIM];
  int base_offset;
  int max_width;
  int curr_slot;
  int curr_chunk[2];
  long curr_offset;
  long last_offset;
  long curr_size;
  long copy_from;
  long copy_amount;
};

std::vector<datasets> dats(0);

cudaStream_t stream_copy_up = 0;
cudaStream_t stream_copy_down = 0;
cudaStream_t stream_compute = 0;
cudaStream_t stream = 0;

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
    delta = (dats[dat->index].max_width - dependency_ranges[dat->index][1 * 2 * OPS_MAX_DIM + 2 * d + 0] + dependency_ranges[dat->index][0 * 2 * OPS_MAX_DIM + 2 * d + 0])*prod;
  else delta = 0;

  if (end_ptr < base_ptr) {printf("WARNING: overreaching depranges! Please check, shouldn't happen\n%s %ld-%ld, dep range: %d-%d prev %d - %d next start %d\n",dat->name, base_ptr, end_ptr, dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * d + 0], dependency_ranges[dat->index][tile * 2 * OPS_MAX_DIM + 2 * d + 1],dependency_ranges[dat->index][mod(tile-1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 0],dependency_ranges[dat->index][mod(tile-1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 1], dependency_ranges[dat->index][mod(tile+1,num_tiles) * 2 * OPS_MAX_DIM + 2 * d + 0]); end_ptr = base_ptr;} //zero ranges
}

void ops_prepare_tile(int tile, int total_tiles, std::vector<std::vector<int> > &tiled_ranges, std::vector<std::vector<int> > &dependency_ranges) {

  if (tile == 0) {
    //TODO: v1 - this doesn't really do anything.
    cutilSafeCall(cudaStreamSynchronize(stream_copy_up));
    cutilSafeCall(cudaStreamSynchronize(stream_compute));

    //First time ever - initialise
    if (dats.size()==0) {
      dats.resize(dependency_ranges.size());
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
      cutilSafeCall(cudaStreamCreateWithFlags(&stream_copy_down,cudaStreamNonBlocking));
      cutilSafeCall(cudaStreamCreateWithFlags(&stream_copy_up,cudaStreamNonBlocking));
      int leastPriority, greatestPriority;
      cudaDeviceGetStreamPriorityRange ( &leastPriority, &greatestPriority );
      cutilSafeCall(cudaStreamCreateWithPriority(&stream_compute,cudaStreamNonBlocking,greatestPriority));
      stream = stream_compute;
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
      if (maxsize > 1 && dats[idx].bytes == 0) maxsize += 13;

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
        dats[idx].curr_size = 0;
        dats[idx].copy_from = 0;
        dats[idx].copy_amount = 0;
      }
    }
  }
 
  //Create event that we can sync on an the end to make sure previous copies have finished 
  cudaEvent_t e_copyup;
  cudaEventCreate(&e_copyup);
  if (tile != 0) //TODO: v1 previous tile stack's last tile prefetching next tile stack's first tile
    cudaEventRecord(e_copyup, stream_copy_up);

  //Copy required data to GPU
  //TODO: assert that only last dim is tiled
  //TODO: SoA: set dat->size[dat->block->dims-1]
  ops_dat_entry *item, *tmp_item;
  if (tile == 0) { //TODO: v1, will need to check if it was uploaded predictively okay
    cudaStreamSynchronize(stream_copy_down);
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_dat dat = item->dat;
      int idx = dat->index;

      //Determine data to be copied up to the GPU
      long base_ptr, end_ptr, delta;
      ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, tile, total_tiles, 2, delta); //Full
      printf("Tile 0 Copying %s from %p+%ld to %p+%ld (%p-%p), size %ld\n", dat->name, dat->data, base_ptr, dat->data_d,delta, dat->data_d, dat->data_d + dats[idx].bytes, end_ptr-base_ptr);
      cutilSafeCall(cudaMemcpyAsync(dat->data_d+delta, dat->data + base_ptr, end_ptr - base_ptr, cudaMemcpyHostToDevice, stream_copy_up));
      dats[idx].curr_slot = 0;
      dats[idx].curr_offset = delta;
      dats[idx].last_offset = 0;
      dats[idx].curr_size = end_ptr-base_ptr;
      dats[idx].curr_chunk[0] = dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0];
      dats[idx].curr_chunk[1] = dependency_ranges[idx][tile * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1];
      dats[idx].copy_from = 0;
      dats[idx].copy_amount = 0;
    }
    cudaEventRecord(e_copyup, stream_copy_up);
  }
  if (tile != total_tiles - 1) { //TODO: v1 upload next tile - leave a few rows spare if next tile's dependency range is bigger
    for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
      tmp_item = TAILQ_NEXT(item, entries);
      ops_dat dat = item->dat;
      int idx = dat->index;

      //For edge thats I do not need to upload again
      //TODO: v1
      if (dat->size[dat->block->dims-1] == 1) continue;

      //Upload next tile
      long base_ptr, end_ptr, delta;
      ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, mod(tile+1,total_tiles), total_tiles, 1, delta); //Right
      if (dats[idx].curr_slot < 2) {
        printf("Prefetching tile %d to slot %d Copying %s from %p+%ld to %p+%ld (%p-%p), size %ld\n", tile+1, dats[idx].curr_slot+1, dat->name, dat->data, base_ptr, dat->data_d, dats[idx].curr_offset + dats[idx].curr_size, dat->data_d, dat->data_d + dats[idx].bytes, end_ptr-base_ptr);
        cutilSafeCall(cudaMemcpyAsync(dat->data_d + dats[idx].curr_offset + dats[idx].curr_size,
                dat->data + base_ptr, end_ptr - base_ptr, cudaMemcpyHostToDevice, stream_copy_up));
        dats[idx].curr_slot++; //Smaller than two, so just increment
        dats[idx].last_offset = dats[idx].curr_offset;
        dats[idx].curr_offset += dats[idx].curr_size; //end of previous
        dats[idx].curr_size = end_ptr-base_ptr;
        dats[idx].copy_from = 0;
        dats[idx].copy_amount = 0;
      } else { //Going to first slot, need extra offset, and copy of previous tile's overlapping dependency range
        //Compute Full range, then right begin - full begin is the extra offset
        long base_ptr2, end_ptr2;
        ops_get_offsets_deprange(base_ptr2, end_ptr2, dat, dependency_ranges, mod(tile+1,total_tiles), total_tiles, 2, delta); //Full
        long extra_offset = base_ptr - base_ptr2;
        printf("Prefetching tile %d to slot %d Copying %s from %p+%ld to %p+%ld (%p-%p), size %ld\n", tile+1, 0, dat->name, dat->data, base_ptr, dat->data_d, extra_offset, dat->data_d, dat->data_d + dats[idx].bytes, end_ptr-base_ptr);
        cutilSafeCall(cudaMemcpyAsync(dat->data_d + extra_offset,
                dat->data + base_ptr, end_ptr - base_ptr, cudaMemcpyHostToDevice, stream_copy_up));
        dats[idx].copy_from = dats[idx].curr_offset + dats[idx].curr_size - extra_offset;
        dats[idx].copy_amount = extra_offset;
        dats[idx].curr_slot = 0;
        dats[idx].last_offset = dats[idx].curr_offset;
        dats[idx].curr_offset = 0;
        dats[idx].curr_size = end_ptr - base_ptr2; //specify full size with extra on left

        //Do an extra check for potential overlap
        ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, 0, total_tiles, 2, delta);
        if (dats[idx].curr_size + extra_offset > delta + end_ptr-base_ptr) printf("Warning - potential race condition %s delta %ld (%d-%d)\n",dat->name, delta,dependency_ranges[dat->index][0 * 2 * OPS_MAX_DIM + 2 * (dat->block->dims-1) + 1], dependency_ranges[dat->index][1 * 2 * OPS_MAX_DIM + 2 * (dat->block->dims-1) + 0]);
      }
      dats[idx].curr_chunk[0] = dependency_ranges[dat->index][mod(tile+1,total_tiles) * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 0];
      dats[idx].curr_chunk[1] = dependency_ranges[dat->index][mod(tile+1,total_tiles) * 2 * OPS_MAX_DIM + 2 * (dat->block->dims - 1) + 1];
    }
  }

  //alter base_offset so that it is offset by the dependency range for the current tile
  for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    ops_dat dat = item->dat;
    if ((tile < total_tiles-1 && dats[dat->index].curr_slot == 1) || (tile == total_tiles-1 && dats[dat->index].curr_slot == 0) || dat->size[dat->block->dims-1] == 1) { //TODO: v1 upload next handling
      long base_ptr, end_ptr, delta;
      ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, tile, total_tiles, 2, delta); //Full
      //printf("New base offset for %s: %ld->%ld\n",dat->name, dat->base_offset, dats[dat->index].base_offset -  base_ptr);
      dat->base_offset = dats[dat->index].base_offset -  base_ptr + delta; //TODO: v1 little extra for first tile for safety
    }
  }

  //Before actually starting the computations, make sure previous copies up finished
  cudaEventSynchronize(e_copyup);
  cudaEventDestroy(e_copyup);
}
void ops_finish_tile(int tile, int total_tiles, std::vector<std::vector<int> > &tiled_ranges, std::vector<std::vector<int> > &dependency_ranges) {
  cudaEvent_t e_copydown;
  cudaEventCreate(&e_copydown);
  cudaEventRecord(e_copydown, stream_copy_down);

  //Wait for compute to finish before downloading results
  cutilSafeCall(cudaStreamSynchronize(stream_compute));

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

    //Copy over the right edge of this tile in the last slot, to the left of the first slot
    if (dats[idx].curr_slot == 0 && dats[idx].copy_amount > 0) {
      cutilSafeCall(cudaMemcpyAsync(dat->data_d, dat->data_d+dats[idx].copy_from, dats[idx].copy_amount, cudaMemcpyDeviceToDevice, stream_compute));
      dats[idx].copy_from = 0;
      dats[idx].copy_amount = 0;
    }
    //Determine data to be copied off to the CPU
    long base_ptr, end_ptr, delta;
    ops_get_offsets_deprange(base_ptr, end_ptr, dat, dependency_ranges, tile, total_tiles, (tile == total_tiles-1 ? 2 : 0),delta); //Left or full if last tile
    //Where the data being processed starts
    long base_ptr_gpu = tile == total_tiles-1 ? dats[idx].curr_offset : dats[idx].last_offset;
    //if we are not downloading from slot 0, then last_offset does not contain the left part of the tile
    if (!(tile < total_tiles-1 && dats[idx].curr_slot == 1) && !(tile == total_tiles-1 && dats[idx].curr_slot==0)) {
      long base_ptr2, end_ptr2;
      ops_get_offsets_deprange(base_ptr2, end_ptr2, dat, dependency_ranges, tile, total_tiles, 1, delta); //Right
      base_ptr_gpu -= (base_ptr2-base_ptr);
    }
    printf("Tile %d copying back %s to %p+%ld from %p+%ld, size %ld\n", tile, dat->name, dat->data, base_ptr, dat->data_d, base_ptr_gpu,end_ptr-base_ptr);
    cutilSafeCall(cudaMemcpyAsync(dat->data + base_ptr, dat->data_d+base_ptr_gpu, end_ptr - base_ptr, cudaMemcpyDeviceToHost, stream_copy_down));
  }

  //Wait for previous round of copies to finish
  cudaEventSynchronize(e_copydown);
  cudaEventDestroy(e_copydown);
  if (((double*)dats[27].dat->data)[202] == 0.0 && tile > 1) {printf("Zero detected for tile %d\n",tile-1);}
//Problem if there is no sync on stream_copy_up here - i.e. ths unload overlaps with next load
}

