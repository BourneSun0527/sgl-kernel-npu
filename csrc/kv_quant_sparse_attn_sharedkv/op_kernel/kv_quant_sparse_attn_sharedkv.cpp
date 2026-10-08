/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * AICore entry for kv_quant_sparse_attn_sharedkv (A5 / ascend950, FP8 KV cache).
 * Adapted from vllm-ascend's templated TilingKey kernel to the sgl-kernel-npu
 * extern "C" + runtime-dispatch launch model (see csrc/sparse_attn_sharedkv/
 * op_kernel/sparse_attn_sharedkv.cpp). The arch35 SCFA device implementation
 * (BaseApi::KvQuantSparseAttnSharedkvScfa and the SCFA block classes) is vendored
 * unchanged under arch35/.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
// The tiling struct + template-mode/layout constants must be visible to the arch35
// SCFA headers (which reference KvQuantSparseAttnSharedkvTilingData and the
// SAS_LAYOUT / SASTemplateMode enums), so include them before the arch35 headers.
#include "kv_quant_sparse_attn_sharedkv_tiling_data.h"
#include "kv_quant_sparse_attn_sharedkv_common.h"
#include "kv_quant_sparse_attn_sharedkv_template_tiling_key.h"
#include "arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h"

using namespace AscendC;
using namespace optiling;

// NOTE(A5-validation): the vendored arch35 SCFA classes were written for vllm-ascend's
// CANN TilingKey compile-time-instantiation + aclnn launch model (TEMPLATES_DEF /
// ARGS_TRAITS / GET_TILING_DATA_WITH_STRUCT macros from ascendc/host_api/tiling/
// template_argument.h). sgl-kernel-npu uses extern "C" + ACLRT_LAUNCH_KERNEL with a
// runtime dispatchKey, so this entry explicitly instantiates the template combos the
// host tiling selects. Whether the arch35 code + CANN template-argument macros compile
// unchanged under this model MUST be verified on an A5 + CANN toolchain; if the macros
// assume the aclnn framework, the arch35 classes need adapting to match the non-quant
// sgl-kernel-npu arch32 port (namespace SASKernel, plain reinterpret_cast tiling).
// Copy the tiling struct from GM into a local instance. AscendC forbids both a
// gm->local reinterpret_cast and a gm->local struct copy-ctor, so the fields are
// copied scalar-by-scalar (each read is a GM load). This is what CANN's
// GET_TILING_DATA_WITH_STRUCT macro does internally; the arch35 SCFA Init expects a
// plain (local) pointer, unlike the non-quant arch32 port whose Init takes __gm__.
__aicore__ inline void KvQuantCopyTilingFromGm(const __gm__ optiling::KvQuantSparseAttnSharedkvTilingData *src,
                                                optiling::KvQuantSparseAttnSharedkvTilingData *dst)
{
    dst->baseParams.batchSize = src->baseParams.batchSize;
    dst->baseParams.qSeqSize = src->baseParams.qSeqSize;
    dst->baseParams.kvSeqSize = src->baseParams.kvSeqSize;
    dst->baseParams.paOriBlockSize = src->baseParams.paOriBlockSize;
    dst->baseParams.paCmpBlockSize = src->baseParams.paCmpBlockSize;
    dst->baseParams.oriMaxBlockNumPerBatch = src->baseParams.oriMaxBlockNumPerBatch;
    dst->baseParams.cmpMaxBlockNumPerBatch = src->baseParams.cmpMaxBlockNumPerBatch;
    dst->baseParams.nNumOfQInOneGroup = src->baseParams.nNumOfQInOneGroup;
    dst->baseParams.sparseBlockCount = src->baseParams.sparseBlockCount;
    dst->baseParams.softmaxScale = src->baseParams.softmaxScale;
    dst->baseParams.oriKvStride = src->baseParams.oriKvStride;
    dst->baseParams.cmpKvStride = src->baseParams.cmpKvStride;
    dst->baseParams.tileSize = src->baseParams.tileSize;
    dst->baseParams.ropeHeadDim = src->baseParams.ropeHeadDim;
    dst->baseParams.cmpRatio = src->baseParams.cmpRatio;
    dst->baseParams.oriMaskMode = src->baseParams.oriMaskMode;
    dst->baseParams.cmpMaskMode = src->baseParams.cmpMaskMode;
    dst->baseParams.oriWinLeft = src->baseParams.oriWinLeft;
    dst->baseParams.oriWinRight = src->baseParams.oriWinRight;
    dst->baseParams.sparseBlockSize = src->baseParams.sparseBlockSize;
    dst->baseParams.hasOriSparseIndices = src->baseParams.hasOriSparseIndices;
    dst->baseParams.oriSparseIndexWidth = src->baseParams.oriSparseIndexWidth;
    dst->baseParams.dSize = src->baseParams.dSize;
    dst->baseParams.dSizeVInput = src->baseParams.dSizeVInput;
    dst->dispatchKey = src->dispatchKey;
}

#define KVQ_SAS_OP_IMPL(FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, SPLIT_G)                              \
    do {                                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,                                \
            BaseApi::SCFABlockCube<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, true,               \
                                   static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T),         \
                                   static_cast<SASTemplateMode>(TEMPLATE_MODE), SPLIT_G>,                          \
            BaseApi::SCFABlockCubeDummy<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, true,           \
                                        static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T),  \
                                        static_cast<SASTemplateMode>(TEMPLATE_MODE), SPLIT_G>>::type;              \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                               \
            BaseApi::SCFABlockVecDummy<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, true,            \
                                       static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T),    \
                                       static_cast<SASTemplateMode>(TEMPLATE_MODE), SPLIT_G>,                      \
            BaseApi::SCFABlockVec<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, true,                 \
                                  static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T),          \
                                  static_cast<SASTemplateMode>(TEMPLATE_MODE), SPLIT_G>>::type;                    \
        BaseApi::KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType> op;                                   \
        auto gmTiling = reinterpret_cast<const __gm__ optiling::KvQuantSparseAttnSharedkvTilingData *>(tiling);    \
        optiling::KvQuantSparseAttnSharedkvTilingData tilingData;                                                \
        KvQuantCopyTilingFromGm(gmTiling, &tilingData);                                                           \
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,            \
                cuSeqlensQ, seqUsedQ, seqUsedKV, sinks, metadata, attentionOut, user, &tilingData, &tPipe);        \
        op.Process();                                                                                             \
    } while (0)

extern "C" __global__ __aicore__ void kv_quant_sparse_attn_sharedkv(
    GM_ADDR query, GM_ADDR oriKV, GM_ADDR cmpKV, GM_ADDR oriSparseIndices, GM_ADDR cmpSparseIndices,
    GM_ADDR oriBlockTable, GM_ADDR cmpBlockTable, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensOriKv,
    GM_ADDR cuSeqlensCmpKv, GM_ADDR seqUsedQ, GM_ADDR seqUsedKV, GM_ADDR sinks, GM_ADDR metadata,
    GM_ADDR attentionOut, GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;
    GM_ADDR user = (workspace != nullptr) ? GetUserWorkspace(workspace) : nullptr;
    auto tilingData = reinterpret_cast<const __gm__ KvQuantSparseAttnSharedkvTilingData *>(tiling);

    // Runtime dispatch on the host-set dispatchKey (see MakeKvQuantSASDispatchKey:
    // flashDecode<<24 | qLayout<<16 | kvLayout<<8 | mode<<4 | splitG). KV layout is
    // always PA_ND(=2); flashDecode/splitG are 0/1; templateMode in {SWA=0,CFA=1,SCFA=2}.
    switch (tilingData->dispatchKey) {
        // SCFA (the A5 FP8 production path) — BSND
        case 0x00000220: KVQ_SAS_OP_IMPL(0, SAS_LAYOUT_BSND, SAS_LAYOUT_PA_ND, SCFA_TEMPLATE, 0); break;
        case 0x01000220: KVQ_SAS_OP_IMPL(1, SAS_LAYOUT_BSND, SAS_LAYOUT_PA_ND, SCFA_TEMPLATE, 0); break;
        case 0x00000221: KVQ_SAS_OP_IMPL(0, SAS_LAYOUT_BSND, SAS_LAYOUT_PA_ND, SCFA_TEMPLATE, 1); break;
        case 0x01000221: KVQ_SAS_OP_IMPL(1, SAS_LAYOUT_BSND, SAS_LAYOUT_PA_ND, SCFA_TEMPLATE, 1); break;
        // SCFA — TND
        case 0x00010220: KVQ_SAS_OP_IMPL(0, SAS_LAYOUT_TND, SAS_LAYOUT_PA_ND, SCFA_TEMPLATE, 0); break;
        case 0x01010220: KVQ_SAS_OP_IMPL(1, SAS_LAYOUT_TND, SAS_LAYOUT_PA_ND, SCFA_TEMPLATE, 0); break;
        case 0x00010221: KVQ_SAS_OP_IMPL(0, SAS_LAYOUT_TND, SAS_LAYOUT_PA_ND, SCFA_TEMPLATE, 1); break;
        case 0x01010221: KVQ_SAS_OP_IMPL(1, SAS_LAYOUT_TND, SAS_LAYOUT_PA_ND, SCFA_TEMPLATE, 1); break;
        // CFA — BSND
        case 0x00000210: KVQ_SAS_OP_IMPL(0, SAS_LAYOUT_BSND, SAS_LAYOUT_PA_ND, CFA_TEMPLATE, 0); break;
        case 0x01000210: KVQ_SAS_OP_IMPL(1, SAS_LAYOUT_BSND, SAS_LAYOUT_PA_ND, CFA_TEMPLATE, 0); break;
        // SWA — BSND
        case 0x00000200: KVQ_SAS_OP_IMPL(0, SAS_LAYOUT_BSND, SAS_LAYOUT_PA_ND, SWA_TEMPLATE, 0); break;
        case 0x01000200: KVQ_SAS_OP_IMPL(1, SAS_LAYOUT_BSND, SAS_LAYOUT_PA_ND, SWA_TEMPLATE, 0); break;
        default: break;
    }
}
