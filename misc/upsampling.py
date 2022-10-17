import jittor as jt
from jittor import nn
import numpy as np

# borrowed from OpenPoints
class ThreeNN(jt.Function):
    def __init__(self):
        super().__init__()
        self.cuda_header = '''
        #define TOTAL_THREADS 1024
        #define THREADS_PER_BLOCK 256 
        #define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
        __global__ void three_nn_kernel_fast(int b, int n, int m, const float *__restrict__ unknown, 
            const float *__restrict__ known, float *__restrict__ dist2, int *__restrict__ idx) {
            // unknown: (B, N, 3)
            // known: (B, M, 3)
            // output: 
            //      dist2: (B, N, 3)
            //      idx: (B, N, 3)
            
            int bs_idx = blockIdx.y;
            int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (bs_idx >= b || pt_idx >= n) return;

            unknown += bs_idx * n * 3 + pt_idx * 3;
            known += bs_idx * m * 3;
            dist2 += bs_idx * n * 3 + pt_idx * 3;
            idx += bs_idx * n * 3 + pt_idx * 3;

            float ux = unknown[0];
            float uy = unknown[1];
            float uz = unknown[2];

            double best1 = 1e40, best2 = 1e40, best3 = 1e40;
            int besti1 = 0, besti2 = 0, besti3 = 0;
            for (int k = 0; k < m; ++k) {
                float x = known[k * 3 + 0];
                float y = known[k * 3 + 1];
                float z = known[k * 3 + 2];
                float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
                if (d < best1) {
                    best3 = best2; besti3 = besti2;
                    best2 = best1; besti2 = besti1;
                    best1 = d; besti1 = k;
                } 
                else if (d < best2) {
                    best3 = best2; besti3 = besti2;
                    best2 = d; besti2 = k;
                } 
                else if (d < best3) {
                    best3 = d; besti3 = k;
                }
            }
            dist2[0] = best1; dist2[1] = best2; dist2[2] = best3;
            idx[0] = besti1; idx[1] = besti2; idx[2] = besti3;
        }


        void three_nn_kernel_launcher_fast(int b, int n, int m, const float *unknown, 
            const float *known, float *dist2, int *idx) {
            // unknown: (B, N, 3)
            // known: (B, M, 3)
            // output: 
            //      dist2: (B, N, 3)
            //      idx: (B, N, 3)

            cudaError_t err;
            dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
            dim3 threads(THREADS_PER_BLOCK);

            three_nn_kernel_fast<<<blocks, threads>>>(b, n, m, unknown, known, dist2, idx);

            err = cudaGetLastError();
            if (cudaSuccess != err) {
                fprintf(stderr, "CUDA kernel failed");
                exit(-1);
            }
        }
        '''
        self.forward_src = '''
        @alias(unknown, in0)
        @alias(known, in1)
        @alias(dist, out0)
        @alias(idx, out1)
        three_nn_kernel_launcher_fast(unknown_shape0, unknown_shape1, known_shape1, unknown_p, known_p, dist_p, idx_p);
        '''
    def execute(ctx, unknown, known):
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        B, N, _ = unknown.size()
        dist2, idx = jt.code([(B,N,3), (B,N,3)], [unknown.dtype, jt.int32], [unknown, known],
        cuda_header=ctx.cuda_header, cuda_src=ctx.forward_src)
        return jt.sqrt(dist2), idx

    def grad(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply

# borrowed from OpenPoints
class ThreeInterpolate(jt.Function):
    def __init__(self):
        super().__init__()
        self.cuda_header = '''
        #undef out
        #define TOTAL_THREADS 1024
        #define THREADS_PER_BLOCK 256 
        #define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
        __global__ void three_interpolate_kernel_fast(int b, int c, int m, int n, const float *__restrict__ points, 
        const int *__restrict__ idx, const float *__restrict__ weight, float *__restrict__ out) {
            // points: (B, C, M)
            // idx: (B, N, 3)
            // weight: (B, N, 3)
            // output:
            //      out: (B, C, N)

            int bs_idx = blockIdx.z;
            int c_idx = blockIdx.y;
            int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

            weight += bs_idx * n * 3 + pt_idx * 3;
            points += bs_idx * c * m + c_idx * m;
            idx += bs_idx * n * 3 + pt_idx * 3;
            out += bs_idx * c * n + c_idx * n;

            out[pt_idx] = weight[0] * points[idx[0]] + weight[1] * points[idx[1]] + weight[2] * points[idx[2]];
        }

        void three_interpolate_kernel_launcher_fast(int b, int c, int m, int n, 
            const float *points, const int *idx, const float *weight, float *out) {
            // points: (B, C, M)
            // idx: (B, N, 3)
            // weight: (B, N, 3)
            // output:
            //      out: (B, C, N)

            cudaError_t err;
            dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
            dim3 threads(THREADS_PER_BLOCK);
            three_interpolate_kernel_fast<<<blocks, threads>>>(b, c, m, n, points, idx, weight, out);

            err = cudaGetLastError();
            if (cudaSuccess != err) {
                fprintf(stderr, "CUDA kernel failed");
                exit(-1);
            }
        }


        __global__ void three_interpolate_grad_kernel_fast(int b, int c, int n, int m, const float *__restrict__ grad_out, 
            const int *__restrict__ idx, const float *__restrict__ weight, float *__restrict__ grad_points) {
            // grad_out: (B, C, N)
            // weight: (B, N, 3)
            // output:
            //      grad_points: (B, C, M)

            int bs_idx = blockIdx.z;
            int c_idx = blockIdx.y;
            int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;
            
            grad_out += bs_idx * c * n + c_idx * n + pt_idx;
            weight += bs_idx * n * 3 + pt_idx * 3;
            grad_points += bs_idx * c * m + c_idx * m;
            idx += bs_idx * n * 3 + pt_idx * 3;


            atomicAdd(grad_points + idx[0], grad_out[0] * weight[0]);
            atomicAdd(grad_points + idx[1], grad_out[0] * weight[1]);
            atomicAdd(grad_points + idx[2], grad_out[0] * weight[2]);
        }

        void three_interpolate_grad_kernel_launcher_fast(int b, int c, int n, int m, const float *grad_out, 
            const int *idx, const float *weight, float *grad_points) {
            // grad_out: (B, C, N)
            // weight: (B, N, 3)
            // output:
            //      grad_points: (B, C, M)

            cudaError_t err;
            dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
            dim3 threads(THREADS_PER_BLOCK);
            three_interpolate_grad_kernel_fast<<<blocks, threads>>>(b, c, n, m, grad_out, idx, weight, grad_points);

            err = cudaGetLastError();
            if (cudaSuccess != err) {
                fprintf(stderr, "CUDA kernel failed");
                exit(-1);
            }
        }
        void three_interpolate_wrapper_fast(int b, int c, int m, int n,
                                float* points,
                                int* idx,
                                float* weight,
                                float* out) {
            three_interpolate_kernel_launcher_fast(b, c, m, n, points, idx, weight, out);
        }
        void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m,
                                float* grad_out,
                                int* idx,
                                float* weight,
                                float* grad_points) {
            three_interpolate_grad_kernel_launcher_fast(b, c, n, m, grad_out, idx, weight, grad_points);
        }
        '''
        self.forward_src = '''
        @alias(feature, in0)
        @alias(idx, in1)
        @alias(weights, in2)
        @alias(output, out0)
        three_interpolate_wrapper_fast(feature_shape0, feature_shape1, feature_shape2, idx_shape1, feature_p, idx_p, weights_p, output_p);
        '''
        self.backward_src = '''
        @alias(grad_out, in0)
        @alias(idx, in1)
        @alias(weights, in2)
        @alias(output, out0)
        three_interpolate_grad_kernel_launcher_fast(grad_out_shape0, grad_out_shape1, output_shape2, grad_out_shape2, grad_out_p, idx_p, weights_p, output_p);
        '''

    def execute(self, features, idx, weight):
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)
        self.idx = idx
        self.weight = weight 
        self.m = m
        output = jt.code([B,c,n],features.dtype,[features,idx,weight],cuda_header=self.cuda_header,cuda_src=self.forward_src)
        return output

    def grad(self, grad_out):
        """
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = self.idx, self.weight,self.m
        B, c, n = grad_out.size()
        grad_features = jt.code([B,c,m],grad_out.dtype,[grad_out, idx, weight], cuda_header=self.cuda_header,cuda_src=self.backward_src)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply

def three_interpolation(unknown_xyz, known_xyz, know_feat):
    """
    input: known_xyz: (m, 3), unknown_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    dist, idx = three_nn(unknown_xyz, known_xyz)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = jt.sum(dist_recip, dim=2, keepdims=True)
    weight = dist_recip / norm
    interpolated_feats = three_interpolate(know_feat, idx, weight)
    return interpolated_feats
