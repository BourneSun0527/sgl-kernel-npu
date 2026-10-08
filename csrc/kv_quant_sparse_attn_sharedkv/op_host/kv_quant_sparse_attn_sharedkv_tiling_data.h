/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef SGL_KERNEL_NPU_KV_QUANT_SPARSE_ATTN_SHAREDKV_TILING_DATA_H
#define SGL_KERNEL_NPU_KV_QUANT_SPARSE_ATTN_SHAREDKV_TILING_DATA_H

#include <cstdint>

namespace optiling {

// Mirrors vllm-ascend's KvQuantSparseAttnSharedkvBaseParams (plain struct form,
// matching the sgl-kernel-npu convention where tiling data is a trivially-copyable
// struct serialized by ge_helper::TilingTensorCache rather than CANN's
// BEGIN_TILING_DATA_DEF/REGISTER_TILING_DATA_CLASS macros).
struct KvQuantSparseAttnSharedkvBaseParams {
    uint32_t batchSize = 0;
    uint32_t qSeqSize = 0;
    uint32_t kvSeqSize = 0;
    uint32_t paOriBlockSize = 0;
    uint32_t paCmpBlockSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t nNumOfQInOneGroup = 0;
    uint32_t sparseBlockCount = 0;
    float softmaxScale = 1.0F;
    int32_t oriKvStride = 0;
    int32_t cmpKvStride = 0;
    uint32_t tileSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t cmpRatio = 0;
    uint32_t oriMaskMode = 0;
    uint32_t cmpMaskMode = 0;
    int32_t oriWinLeft = 0;
    int32_t oriWinRight = 0;
    uint32_t sparseBlockSize = 0;
    bool hasOriSparseIndices = false;
    uint32_t oriSparseIndexWidth = 0;
    uint32_t dSize = 0;
    uint32_t dSizeVInput = 0;
};

struct KvQuantSparseAttnSharedkvTilingData {
    KvQuantSparseAttnSharedkvBaseParams baseParams{};
    // Runtime dispatch key selecting the AICore kernel template instantiation
    // (flashDecode | qLayout | kvLayout | templateMode | splitG), set by the host
    // tiling pass and switched on by the extern "C" kernel entry. Encoded with
    // MakeKvQuantSASDispatchKey below.
    uint32_t dispatchKey = 0;
};

// Template-mode values (must match kv_quant_sparse_attn_sharedkv_common.h /
// kv_quant_sparse_attn_sharedkv_template_tiling_key.h).
enum class KvQuantSASTemplateMode : uint32_t {
    SWA_TEMPLATE_MODE = 0,
    CFA_TEMPLATE_MODE = 1,
    SCFA_TEMPLATE_MODE = 2
};

enum class KvQuantSASLayout : uint32_t { BSND = 0, TND = 1, PA_ND = 2 };

// dispatchKey layout (from high to low):
//   [31:24] flashDecode (0/1)
//   [23:16] qLayout    (BSND=0 / TND=1)
//   [15:8]  kvLayout   (PA_ND=2)
//   [7:4]   templateMode (SWA=0 / CFA=1 / SCFA=2)
//   [3:0]   splitG     (0/1)
constexpr uint32_t MakeKvQuantSASDispatchKey(uint32_t flashDecode, KvQuantSASLayout qLayout, KvQuantSASLayout kvLayout,
                                             KvQuantSASTemplateMode mode, uint32_t splitG)
{
    return ((flashDecode & 0xFFU) << 24U) | (static_cast<uint32_t>(qLayout) << 16U) |
           (static_cast<uint32_t>(kvLayout) << 8U) | ((static_cast<uint32_t>(mode) & 0xFU) << 4U) | (splitG & 0xFU);
}

}  // namespace optiling
#endif
