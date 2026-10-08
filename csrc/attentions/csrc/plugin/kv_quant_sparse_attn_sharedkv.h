/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Plugin host binding for kv_quant_sparse_attn_sharedkv (A5 / ascend950, FP8 KV cache).
 * Calls the aclnnKvQuantSparseAttnSharedkv symbol produced by the opbuild pipeline.
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_IMPL_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

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
    bool return_softmax_lse);

#endif  // KV_QUANT_SPARSE_ATTN_SHAREDKV_IMPL_H
