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

torch::Tensor forward_hpg4_wide_ref(
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
    int64_t window,
    int64_t chunk_size);

torch::Tensor forward_hpg4_wide_opt(
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
    int64_t window,
    int64_t chunk_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_hpg4_bm32_ref",
        &forward_hpg4_bm32_ref,
        "DC HPG=4 BM=32 scalar CUDA reference forward");
    m.def(
        "forward_hpg4_wide_ref",
        &forward_hpg4_wide_ref,
        "DC HPG=4 KL=256 scalar CUDA reference forward");
    m.def(
        "forward_hpg4_wide_opt",
        &forward_hpg4_wide_opt,
        "DC HPG=4 KL=256 tensor-core CUDA experimental forward");
}
