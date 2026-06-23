#include <torch/extension.h>

// Defined in attention.cu 
at:::Tensor fused_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_attention", &fused_attention, "Fused online-softmax attention (CUDA)");
}

