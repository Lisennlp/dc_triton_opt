#include <torch/extension.h>

torch::Tensor forward_hpg4_bm32_ref(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor pre_w1,
    torch::Tensor pre_w2,
    torch::Tensor pre_dd,
    torch::Tensor post_w1,
    torch::Tensor post_w2,
    torch::Tensor post_dd,
    double scaling,
    int64_t window);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_hpg4_bm32_ref",
        &forward_hpg4_bm32_ref,
        "DC HPG=4 BM=32 scalar CUDA reference forward");
}

