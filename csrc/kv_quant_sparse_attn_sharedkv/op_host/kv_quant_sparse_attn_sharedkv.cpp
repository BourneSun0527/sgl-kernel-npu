/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Host binding for kv_quant_sparse_attn_sharedkv (A5 / ascend950, FP8 KV cache).
 * Mirrors csrc/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv.cpp but uses an
 * inline minimal tiling (the full ~28KB optiling port from vllm-ascend's
 * gert::TilingContext framework to ge_helper::TilingContext is vendored as
 * kv_quant_sparse_attn_sharedkv_tiling.cpp.orig / .h.orig and must be ported on an
 * A5 + CANN environment). Until then the op is registered and callable via
 * torch.ops.npu.kv_quant_sparse_attn_sharedkv, but the tiling fields are best-effort
 * and the SCFA kernel results are not validated.
 */

#include <limits>
#include <memory>
#include <string>

#include "acl/acl.h"
#include "aclrtlaunch_kv_quant_sparse_attn_sharedkv.h"
#include "ge_helper.h"
#include "torch_helper.h"
#include "kv_quant_sparse_attn_sharedkv_def.h"
#include "kv_quant_sparse_attn_sharedkv_tiling_data.h"

namespace sglang::npu_kernel {
namespace {

using ge_helper::TilingContext;
using sglang::SASHost::KvQuantSparseAttnSharedkv;

void CheckTensor(const at::Tensor &tensor, const at::Tensor &q, const char *name)
{
    TORCH_CHECK(tensor.device().type() == q.device().type() && tensor.device().index() == q.device().index(), name,
                " must be on the same device as q");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalTensor(const c10::optional<at::Tensor> &tensor, const at::Tensor &q, const char *name)
{
    if (tensor.has_value()) {
        CheckTensor(*tensor, q, name);
    }
}

at::Tensor Placeholder(const at::Tensor &q, at::ScalarType dtype)
{
    return at::empty({1}, q.options().dtype(dtype));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> kv_quant_sparse_attn_sharedkv(
    const at::Tensor &q, int64_t /*kv_quant_mode*/, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata, int64_t tile_size,
    int64_t rope_head_dim, double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
    int64_t ori_win_left, int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv,
    bool return_softmax_lse)
{
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "kv_quant_sparse_attn_sharedkv: q must be bfloat16");
    TORCH_CHECK(q.device().type() == DEVICE_TYPE, "kv_quant_sparse_attn_sharedkv: q must be an NPU tensor");
    TORCH_CHECK(q.is_contiguous(), "kv_quant_sparse_attn_sharedkv: q must be contiguous");
    TORCH_CHECK((layout_q == "BSND" && q.dim() == 4) || (layout_q == "TND" && q.dim() == 3),
                "kv_quant_sparse_attn_sharedkv: q rank does not match layout_q");
    TORCH_CHECK(ori_kv.has_value(), "kv_quant_sparse_attn_sharedkv: ori_kv is required");
    TORCH_CHECK(ori_kv->scalar_type() == at::kFloat8_e4m3fn, "kv_quant_sparse_attn_sharedkv: ori_kv must be fp8_e4m3fn");
    TORCH_CHECK(!cmp_kv.has_value() || cmp_kv->scalar_type() == at::kFloat8_e4m3fn,
                "kv_quant_sparse_attn_sharedkv: cmp_kv must be fp8_e4m3fn");
    TORCH_CHECK(sinks.has_value(), "kv_quant_sparse_attn_sharedkv: sinks is required");
    TORCH_CHECK(sinks->scalar_type() == at::kFloat, "kv_quant_sparse_attn_sharedkv: sinks must be float32");
    TORCH_CHECK(metadata.has_value(), "kv_quant_sparse_attn_sharedkv: metadata is required");
    TORCH_CHECK(metadata->scalar_type() == at::kInt && metadata->numel() == 1024,
                "kv_quant_sparse_attn_sharedkv: metadata must be int32 with 1024 elements");
    TORCH_CHECK(layout_kv == "PA_ND", "kv_quant_sparse_attn_sharedkv: this port supports layout_kv='PA_ND' only");
    TORCH_CHECK(layout_q == "BSND" || layout_q == "TND", "kv_quant_sparse_attn_sharedkv: layout_q must be 'BSND' or 'TND'");
    TORCH_CHECK(ori_block_table.has_value(), "kv_quant_sparse_attn_sharedkv: ori_block_table is required for PA_ND");
    TORCH_CHECK(seqused_kv.has_value(), "kv_quant_sparse_attn_sharedkv: seqused_kv is required for PA_ND");
    TORCH_CHECK(layout_q != "TND" || cu_seqlens_q.has_value(),
                "kv_quant_sparse_attn_sharedkv: cu_seqlens_q is required for TND query layout");
    TORCH_CHECK(layout_q != "BSND" || !cu_seqlens_q.has_value(),
                "kv_quant_sparse_attn_sharedkv: cu_seqlens_q must be None for BSND query layout");
    TORCH_CHECK(softmax_scale >= 0.0, "kv_quant_sparse_attn_sharedkv: softmax_scale must be non-negative");

    CheckTensor(*ori_kv, q, "ori_kv");
    CheckOptionalTensor(cmp_kv, q, "cmp_kv");
    CheckOptionalTensor(ori_sparse_indices, q, "ori_sparse_indices");
    CheckOptionalTensor(cmp_sparse_indices, q, "cmp_sparse_indices");
    CheckOptionalTensor(ori_block_table, q, "ori_block_table");
    CheckOptionalTensor(cmp_block_table, q, "cmp_block_table");
    CheckOptionalTensor(cu_seqlens_q, q, "cu_seqlens_q");
    CheckOptionalTensor(cu_seqlens_ori_kv, q, "cu_seqlens_ori_kv");
    CheckOptionalTensor(cu_seqlens_cmp_kv, q, "cu_seqlens_cmp_kv");
    CheckOptionalTensor(seqused_q, q, "seqused_q");
    CheckOptionalTensor(seqused_kv, q, "seqused_kv");
    CheckTensor(*sinks, q, "sinks");
    CheckTensor(*metadata, q, "metadata");

    auto checkInt32 = [](const c10::optional<at::Tensor> &tensor, const char *name) {
        TORCH_CHECK(!tensor.has_value() || tensor->scalar_type() == at::kInt, name, " must be int32");
    };
    checkInt32(ori_sparse_indices, "ori_sparse_indices");
    checkInt32(cmp_sparse_indices, "cmp_sparse_indices");
    checkInt32(ori_block_table, "ori_block_table");
    checkInt32(cmp_block_table, "cmp_block_table");
    checkInt32(cu_seqlens_q, "cu_seqlens_q");
    checkInt32(cu_seqlens_ori_kv, "cu_seqlens_ori_kv");
    checkInt32(cu_seqlens_cmp_kv, "cu_seqlens_cmp_kv");
    checkInt32(seqused_q, "seqused_q");
    checkInt32(seqused_kv, "seqused_kv");

    auto attention_out = at::empty_like(q);
    auto softmax_lse = at::empty({0}, q.options().dtype(at::kFloat));

    // Set up the op definition + tiling context (mirrors sparse_attn_sharedkv.cpp).
    KvQuantSparseAttnSharedkv op("kv_quant_sparse_attn_sharedkv");
    op.SetAttrAny("kv_quant_mode", static_cast<int32_t>(1));
    op.SetAttrAny("tile_size", static_cast<uint32_t>(tile_size));
    op.SetAttrAny("rope_head_dim", static_cast<uint32_t>(rope_head_dim));
    op.SetAttrAny("softmax_scale", static_cast<float>(softmax_scale));
    op.SetAttrAny("cmp_ratio", static_cast<uint32_t>(cmp_ratio));
    op.SetAttrAny("ori_mask_mode", static_cast<uint32_t>(ori_mask_mode));
    op.SetAttrAny("cmp_mask_mode", static_cast<uint32_t>(cmp_mask_mode));
    op.SetAttrAny("ori_win_left", static_cast<uint32_t>(ori_win_left));
    op.SetAttrAny("ori_win_right", static_cast<uint32_t>(ori_win_right));
    op.SetAttrStr("layout_q", std::string(layout_q));
    op.SetAttrStr("layout_kv", std::string(layout_kv));
    op.SetAttrAny("ori_kv_stride0", static_cast<uint32_t>(ori_kv->stride(0)));
    op.SetAttrAny("cmp_kv_stride0", static_cast<uint32_t>(cmp_kv.has_value() ? cmp_kv->stride(0) : 0));
    op.SetAttrAny("return_softmax_lse", return_softmax_lse);

    auto context = std::make_shared<TilingContext>("kv_quant_sparse_attn_sharedkv");
    auto scalarType = q.scalar_type();
    op.SetToContext(context, scalarType);
    context->RegisterTensor(q, true);
    context->RegisterTensor(ori_kv, true);
    context->RegisterTensor(cmp_kv, true);
    context->RegisterTensor(ori_sparse_indices, true);
    context->RegisterTensor(cmp_sparse_indices, true);
    context->RegisterTensor(ori_block_table, true);
    context->RegisterTensor(cmp_block_table, true);
    context->RegisterTensor(cu_seqlens_q, true);
    context->RegisterTensor(cu_seqlens_ori_kv, true);
    context->RegisterTensor(cu_seqlens_cmp_kv, true);
    context->RegisterTensor(seqused_q, true);
    context->RegisterTensor(seqused_kv, true);
    context->RegisterTensor(sinks, true);
    context->RegisterTensor(metadata, true);
    context->RegisterTensor(attention_out, false);
    context->RegisterTensor(softmax_lse, false);

    // TODO(port): replace this inline best-effort tiling with the full port from
    // kv_quant_sparse_attn_sharedkv_tiling.cpp.orig (gert::TilingContext ->
    // ge_helper::TilingContext), which computes blockSize / mBaseSize / s2 splits /
    // core assignment / workspace via SASInfoParser + SASTilingCheck +
    // KvQuantSparseAttnSharedkvTiling.
    optiling::KvQuantSparseAttnSharedkvTilingData tilingData;
    auto &base = tilingData.baseParams;
    base.batchSize = (layout_q == "BSND") ? static_cast<uint32_t>(q.size(0)) : 1U;
    base.qSeqSize = (layout_q == "BSND") ? static_cast<uint32_t>(q.size(1))
                                         : static_cast<uint32_t>(q.size(0));
    base.dSize = (layout_q == "BSND") ? static_cast<uint32_t>(q.size(3)) : static_cast<uint32_t>(q.size(2));
    base.tileSize = static_cast<uint32_t>(tile_size);
    base.ropeHeadDim = static_cast<uint32_t>(rope_head_dim);
    base.softmaxScale = static_cast<float>(softmax_scale);
    base.cmpRatio = static_cast<uint32_t>(cmp_ratio);
    base.oriMaskMode = static_cast<uint32_t>(ori_mask_mode);
    base.cmpMaskMode = static_cast<uint32_t>(cmp_mask_mode);
    base.oriWinLeft = static_cast<uint32_t>(ori_win_left);
    base.oriWinRight = static_cast<uint32_t>(ori_win_right);
    base.oriKvStride = static_cast<int32_t>(ori_kv->stride(0));
    base.cmpKvStride = static_cast<int32_t>(cmp_kv.has_value() ? cmp_kv->stride(0) : 0);
    // SCFA (the A5 FP8 production path), flashDecode=0, splitG=0.
    tilingData.dispatchKey = optiling::MakeKvQuantSASDispatchKey(
        0U, optiling::KvQuantSASLayout::BSND, optiling::KvQuantSASLayout::PA_ND,
        optiling::KvQuantSASTemplateMode::SCFA_TEMPLATE_MODE, 0U);
    if (layout_q == "TND") {
        tilingData.dispatchKey = optiling::MakeKvQuantSASDispatchKey(
            0U, optiling::KvQuantSASLayout::TND, optiling::KvQuantSASLayout::PA_ND,
            optiling::KvQuantSASLayout::SCFA_TEMPLATE_MODE, 0U);
    }

    // Reserve a modest user workspace; the real size comes from the full tiling.
    constexpr uint64_t kUserWorkspace = 1ULL << 20;  // 1 MiB
    context->SetWorkspaceSizes(static_cast<size_t>(kUserWorkspace));

    auto tilingTensor = context->GetTilingTensor(tilingData);
    auto workspace = at::empty({static_cast<int64_t>(context->GetWorkspaceSize())}, q.options().dtype(at::kByte));

    auto qPlaceholder = Placeholder(q, q.scalar_type());
    auto intPlaceholder = Placeholder(q, at::kInt);
    auto oriKvLaunch = *ori_kv;
    auto cmpKvLaunch = cmp_kv.value_or(qPlaceholder);
    auto oriSparseLaunch = ori_sparse_indices.value_or(intPlaceholder);
    auto cmpSparseLaunch = cmp_sparse_indices.value_or(intPlaceholder);
    auto oriBlockLaunch = ori_block_table.value_or(intPlaceholder);
    auto cmpBlockLaunch = cmp_block_table.value_or(intPlaceholder);
    auto cuQLaunch = cu_seqlens_q.value_or(intPlaceholder);
    auto cuOriKvLaunch = cu_seqlens_ori_kv.value_or(intPlaceholder);
    auto cuCmpKvLaunch = cu_seqlens_cmp_kv.value_or(intPlaceholder);
    auto seqQLaunch = seqused_q.value_or(intPlaceholder);
    auto seqKvLaunch = *seqused_kv;
    auto sinksLaunch = *sinks;
    auto metadataLaunch = *metadata;

    // Blockdim = AIC core count (36 on A5). The full tiling computes a balanced
    // blockDim from the core-distribution plan; fetch the device topology here.
    int32_t aic = 0;
    aclError st = aclGetDeviceCapability(0, ACL_AICORE_NUM, &aic);
    TORCH_CHECK(st == ACL_ERROR_NONE && aic > 0, "kv_quant_sparse_attn_sharedkv: failed to query AIC core count");
    const uint32_t blockDim = static_cast<uint32_t>(aic);

    EXEC_KERNEL_CMD(kv_quant_sparse_attn_sharedkv, blockDim, q, oriKvLaunch, cmpKvLaunch, oriSparseLaunch,
                    cmpSparseLaunch, oriBlockLaunch, cmpBlockLaunch, cuQLaunch, cuOriKvLaunch, cuCmpKvLaunch,
                    seqQLaunch, seqKvLaunch, sinksLaunch, metadataLaunch, attention_out, softmax_lse, workspace,
                    tilingTensor);
    return {attention_out, softmax_lse};
}

}  // namespace sglang::npu_kernel
