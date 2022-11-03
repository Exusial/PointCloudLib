import math
import numpy as np
import jittor as jt
from jittor import nn
from jittor.contrib import concat 

def optimal_block(batch_size):
    return 2 ** int(math.log(batch_size))

class FurthestPointSampler(nn.Module):
    cuda_src='''
        __device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                                int idx1, int idx2) {
            const float v1 = dists[idx1], v2 = dists[idx2];
            const int i1 = dists_i[idx1], i2 = dists_i[idx2];
            dists[idx1] = max(v1, v2);
            dists_i[idx1] = v2 > v1 ? i2 : i1;
        }

        __global__ void furthest_point_sampling_kernel (
            int b, int n, int m, int block_size,
            const float *__restrict__ dataset,
            float *__restrict__ temp, 
            int *__restrict__ idxs) {

            if (m <= 0) return;

            extern __shared__ int dists_i[];
            float *dists =  (float *) &dists_i[block_size];

            int batch_index = blockIdx.x;
            dataset += batch_index * n * 3;
            temp += batch_index * n;
            idxs += batch_index * m;

            int tid = threadIdx.x;
            const int stride = block_size;

            int old = 0;
            if (threadIdx.x == 0) idxs[0] = old;

            // initialize temp with INF
            for (int k = tid; k < n; k += stride)
                temp[k] = 1e10;

            __syncthreads();
            for (int j = 1; j < m; j++) {
                int besti = 0;
                float best = -1;
                float x1 = dataset[old * 3 + 0];
                float y1 = dataset[old * 3 + 1];
                float z1 = dataset[old * 3 + 2];
                for (int k = tid; k < n; k += stride) {
                    float x2, y2, z2;
                    x2 = dataset[k * 3 + 0];
                    y2 = dataset[k * 3 + 1];
                    z2 = dataset[k * 3 + 2];
                    float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
                    if (mag <= 1e-3) continue;

                    float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

                    float d2 = min(d, temp[k]);
                    temp[k] = d2;
                    besti = d2 > best ? k : besti;
                    best = d2 > best ? d2 : best;
                }
                dists[tid] = best;
                dists_i[tid] = besti;
                __syncthreads();

                if (block_size >= 512) {
                    if (tid < 256) {
                        __update(dists, dists_i, tid, tid + 256);
                    }
                    __syncthreads();
                }
                if (block_size >= 256) {
                    if (tid < 128) {
                        __update(dists, dists_i, tid, tid + 128);
                    }
                    __syncthreads();
                }
                if (block_size >= 128) {
                    if (tid < 64) {
                        __update(dists, dists_i, tid, tid + 64);
                    }
                    __syncthreads();
                }
                if (block_size >= 64) {
                    if (tid < 32) {
                        __update(dists, dists_i, tid, tid + 32);
                    }
                    __syncthreads();
                }
                if (block_size >= 32) {
                    if (tid < 16) {
                        __update(dists, dists_i, tid, tid + 16);
                    }
                    __syncthreads();
                }
                if (block_size >= 16) {
                    if (tid < 8) {
                        __update(dists, dists_i, tid, tid + 8);
                    }
                    __syncthreads();
                }
                if (block_size >= 8) {
                    if (tid < 4) {
                        __update(dists, dists_i, tid, tid + 4);
                    }
                    __syncthreads();
                }
                if (block_size >= 4) {
                    if (tid < 2) {
                        __update(dists, dists_i, tid, tid + 2);
                    }
                    __syncthreads();
                }
                if (block_size >= 2) {
                    if (tid < 1) {
                        __update(dists, dists_i, tid, tid + 1);
                    }
                    __syncthreads();
                }

                old = dists_i[0];
                if (tid == 0) idxs[j] = old;
            }
        }

        int block_size = #block_size;

        float *temp;
        cudaMallocManaged(&temp, in0_shape0 * in0_shape1 * sizeof(float));

        furthest_point_sampling_kernel<<<in0_shape0, block_size, 2*block_size*sizeof(int)>>>(
            in0_shape0,
            in0_shape1,
            out_shape1,
            block_size,
            in0_p,
            temp,
            out_p
        );
        cudaDeviceSynchronize();
        cudaFree(temp);
    '''
    def __init__(self, n_samples=None):
        super().__init__()
        self.n_samples = n_samples

    def execute(self, x, n_samples=None):
        '''
        Parameters
        ----------
        x: jt.Var, (B, N, 3)

        Returns
        -------
        y: jt.Var, (B, n_samples, 3)
        '''
        batch_size, n_points, n_coords = x.shape
        if n_samples:
            self.n_samples = n_samples
        assert self.n_samples
        assert self.n_samples <= n_points
        assert n_coords == 3
        assert x.dtype == 'float32'

        block_size = optimal_block(batch_size)

        cuda_src = self.cuda_src.replace('#block_size', str(block_size))

        idxs_shape = [batch_size, self.n_samples]
        idxs = jt.code(idxs_shape, 'int32', [x,], cuda_src=cuda_src)
        
        y = x.reindex([batch_size, self.n_samples, 3], [
            'i0',               # Batchid
            '@e0(i0, i1)',      # Nid
            'i2'
        ], extras=[idxs])

        return y, idxs

class RandomSampler(nn.Module):
    def __init__(self, n_samples=None):
        super().__init__()
        self.n_samples = n_samples
    
    def execute(self, p, n_samples=None):
        if not n_samples:
            n_samples = self.n_samples
        B, N, _ = p.shape
        idx = jt.randint(0, N, shape=(B, n_samples, ))
        y = p.reindex([B, n_samples, 3], ['i0', '@e0(i0,i1)', 'i2'], extras=[idx])
        return y, idx