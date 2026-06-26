// knn_bind.cpp
#include <torch/extension.h>

// 声明 CUDA 实现
torch::Tensor knn_idx_forward_cuda(
    torch::Tensor pix,
    torch::Tensor nodes,
    int64_t k
);

// Python 包装函数
torch::Tensor knn_idx_forward(
    torch::Tensor pix,
    torch::Tensor nodes,
    int64_t k
) {
    return knn_idx_forward_cuda(pix, nodes, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_idx", &knn_idx_forward,
          "KNN index (CUDA) forward, inputs: pix(M,2), nodes(Ns,2), k -> (M,k)");
}
