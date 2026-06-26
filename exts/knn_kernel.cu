// knn_kernel.cu
#include <torch/extension.h>
#include <vector>
#include <cfloat>  // FLT_MAX

template <typename scalar_t>
__global__ void knn_idx_kernel(
    const scalar_t* __restrict__ pix,    // (M,2)
    const scalar_t* __restrict__ nodes,  // (Ns,2)
    long* __restrict__ idx_out,          // (M,k)
    int M,
    int Ns,
    int k
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    scalar_t px = pix[m * 2 + 0];
    scalar_t py = pix[m * 2 + 1];

    // k 很小（比如 6），直接用寄存器维护 top-k
    const int KMAX = 8;  // 安全上限，>= 你实际用的 k
    scalar_t best_dist[KMAX];
    long     best_idx[KMAX];

    // 初始化为 +inf
    for (int i = 0; i < k; ++i) {
        best_dist[i] = FLT_MAX;
        best_idx[i]  = 0;   // 默认 0，后面如果 Ns<k，可以给一个合法下标
    }

    // 全局遍历所有节点，维护 top-k（最小距离）
    for (int n = 0; n < Ns; ++n) {
        scalar_t nx = nodes[n * 2 + 0];
        scalar_t ny = nodes[n * 2 + 1];
        scalar_t dx = px - nx;
        scalar_t dy = py - ny;
        scalar_t dist2 = dx * dx + dy * dy;

        // 找当前 best[] 里距离最大的那个
        int max_j = 0;
        scalar_t max_d = best_dist[0];
        for (int j = 1; j < k; ++j) {
            if (best_dist[j] > max_d) {
                max_d = best_dist[j];
                max_j = j;
            }
        }
        // 如果新点更近，就替换
        if (dist2 < max_d) {
            best_dist[max_j] = dist2;
            best_idx[max_j]  = n;
        }
    }

    // 写回输出
    for (int j = 0; j < k; ++j) {
        idx_out[m * k + j] = best_idx[j];
    }
}

torch::Tensor knn_idx_forward_cuda(
    torch::Tensor pix,   // (M,2), float
    torch::Tensor nodes, // (Ns,2), float
    int64_t k
) {
    TORCH_CHECK(pix.is_cuda(),  "pix must be a CUDA tensor");
    TORCH_CHECK(nodes.is_cuda(),"nodes must be a CUDA tensor");
    TORCH_CHECK(pix.size(1) == 2,  "pix must have shape (M,2)");
    TORCH_CHECK(nodes.size(1) == 2,"nodes must have shape (Ns,2)");
    TORCH_CHECK(k > 0, "k must be > 0");

    pix   = pix.contiguous();
    nodes = nodes.contiguous();

    int M  = pix.size(0);
    int Ns = nodes.size(0);
    int kk = static_cast<int>(k);

    auto opts = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(pix.device());
    auto out = torch::empty({M, k}, opts);

    int threads = 256;
    int blocks  = (M + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(pix.scalar_type(), "knn_idx_forward_cuda", [&](){
        knn_idx_kernel<scalar_t><<<blocks, threads>>>(
            pix.data_ptr<scalar_t>(),
            nodes.data_ptr<scalar_t>(),
            out.data_ptr<long>(),
            M, Ns, kk
        );
    });

    return out;
}
