/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_attn_sharedkv_common.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_COMMON_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "kv_quant_sparse_attn_sharedkv_metadata.h"

using namespace AscendC;

enum class SAS_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_ND = 2
};

enum class SASTemplateMode {
    SWA_TEMPLATE_MODE = 0,
    CFA_TEMPLATE_MODE = 1,
    SCFA_TEMPLATE_MODE = 2
};

// ================================Util functions==================================
// Scalar helpers mirroring csrc/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_common.h.
// The vendored arch35 SCFA code calls unqualified scalar Min/Align that vllm-ascend's CANN
// exposed but CANN 9.1.0 does not (9.1.0 only has the tensor Min). Provide them here so
// the arch35 headers resolve. Kept in the global namespace like the non-quant port.
template <typename T1, typename T2>
__aicore__ inline T1 SASAlign(T1 num, T2 rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd * rnd);
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 num, T2 rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? a : b;
}

// 1-arg Align: align up to the 32-byte block. NOTE(A5-validation): vllm-ascend's CANN
// exposed a default-base 1-arg Align(); the exact default (32 here) must be confirmed
// against the kernel's s2AlignedSize usage on A5.
template <typename T>
__aicore__ inline T Align(T num)
{
    return (num + static_cast<T>(31)) / static_cast<T>(32) * static_cast<T>(32);
}
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_COMMON_H