/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Plugin host binding for kv_quant_sparse_attn_sharedkv (A5 / ascend950, FP8 KV cache).
 * Mirrors csrc/plugin/la.cpp: allocates outputs, then dlsym-calls the
 * aclnnKvQuantSparseAttnSharedkv symbol produced by the opbuild pipeline
 * (csrc/ops/kv_quant_sparse_attn_sharedkv). Ported from vllm-ascend's
 * npu_kv_quant_sparse_attn_sharedkv_npu (csrc/torch_binding.cpp).
 */

#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "kv_quant_sparse_attn_sharedkv.h"

using namespace at;
namespace {
constexpr std::string_view KV_QUANT_SAS_NAME = "aclnnKvQuantSparseAttnSharedkv";

inline at::Tensor opt_tensor(const c10::optional<at::Tensor> &t)
{
    return c10::value_or_else(t, [] { return at::Tensor(); });
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> kv_quant_sparse_attn_sharedkv(
    const at::Tensor &q, int64_t kv_quant_mode, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata, int64_t tile_size,
    int64_t rope_head_dim, double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
    int64_t ori_win_left, int64_t ori_win_right, std::string layout_q, std::string layout_kv,
    bool return_softmax_lse)
{
    for (size_t i = 0; i < q.sizes().size(); i++) {
        TORCH_CHECK(q.size(i) > 0, "kv_quant_sparse_attn_sharedkv: query shape[", i, "] must be > 0, got ",
                    q.size(i));
    }

    at::Tensor attn_out = at::empty(q.sizes(), q.options().dtype(q.dtype()));
    at::Tensor softmax_lse;
    if (return_softmax_lse) {
        std::vector<int64_t> lseSizes(q.sizes().begin(), q.sizes().end());
        lseSizes.back() = 1;
        softmax_lse = at::empty(lseSizes, q.options().dtype(c10::ScalarType::Float));
    } else {
        softmax_lse = at::empty({0}, q.options().dtype(c10::ScalarType::Float));
    }

    int64_t oriKvStride0 = 0;
    int64_t cmpKvStride0 = 0;
    if (ori_kv.has_value() && ori_kv.value().defined()) {
        oriKvStride0 = ori_kv.value().stride(0);
    }
    if (cmp_kv.has_value() && cmp_kv.value().defined()) {
        cmpKvStride0 = cmp_kv.value().stride(0);
    }

    // EXEC_NPU_CMD's ConvertTypes takes lvalue refs, so bind the optional-converted
    // tensors to locals first (mirrors la.cpp's const at::Tensor& = value_or_else(...)).
    const at::Tensor ori_kv_t = opt_tensor(ori_kv);
    const at::Tensor cmp_kv_t = opt_tensor(cmp_kv);
    const at::Tensor ori_sparse_indices_t = opt_tensor(ori_sparse_indices);
    const at::Tensor cmp_sparse_indices_t = opt_tensor(cmp_sparse_indices);
    const at::Tensor ori_block_table_t = opt_tensor(ori_block_table);
    const at::Tensor cmp_block_table_t = opt_tensor(cmp_block_table);
    const at::Tensor cu_seqlens_q_t = opt_tensor(cu_seqlens_q);
    const at::Tensor cu_seqlens_ori_kv_t = opt_tensor(cu_seqlens_ori_kv);
    const at::Tensor cu_seqlens_cmp_kv_t = opt_tensor(cu_seqlens_cmp_kv);
    const at::Tensor seqused_q_t = opt_tensor(seqused_q);
    const at::Tensor seqused_kv_t = opt_tensor(seqused_kv);
    const at::Tensor sinks_t = opt_tensor(sinks);
    const at::Tensor metadata_t = opt_tensor(metadata);

    EXEC_NPU_CMD<KV_QUANT_SAS_NAME>(
        q, ori_kv_t, cmp_kv_t, ori_sparse_indices_t, cmp_sparse_indices_t, ori_block_table_t,
        cmp_block_table_t, cu_seqlens_q_t, cu_seqlens_ori_kv_t, cu_seqlens_cmp_kv_t, seqused_q_t,
        seqused_kv_t, sinks_t, metadata_t, kv_quant_mode, tile_size, rope_head_dim, softmax_scale,
        cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv,
        oriKvStride0, cmpKvStride0, return_softmax_lse, attn_out, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(attn_out, softmax_lse);
}
