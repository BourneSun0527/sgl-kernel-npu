/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Host-side (CPU) reimplementation of the vllm-ascend AICPU op
 * `KvQuantSparseAttnSharedkvMetadata`. The I/O shim (CpuKernelContext /
 * GetAttrValue / Tensor) is replaced by plain pointers + at::Tensor allocation,
 * exactly as done for the non-quant `SparseAttnSharedkvMetadataHost`.
 *
 * STATUS: input validation (Prepare / CheckSingleParam / CheckExistence /
 * CheckFeature / ParamsInit / ProcessSocVersion) and the seq-size getters are
 * ported faithfully from the AICPU source. The scheduling algorithm body
 * (BalanceSchedule -> CalcCostInfo / AssignBlocksToCore / SplitFD -> GenMetaData)
 * is the ~780-line core in
 *   kv_quant_sparse_attn_sharedkv_metadata_aicpu.cpp.orig (lines 299-1091)
 * and is left as TODO stubs: it must be ported line-by-line on an A5 + CANN
 * environment where it can be validated end-to-end against the AICore kernel
 * that consumes this metadata table. Until then Run() returns a zeroed table.
 */

#include "kv_quant_sparse_attn_sharedkv_metadata.h"

#include <algorithm>
#include <climits>
#include <string>

#include "acl/acl.h"

namespace sgl_kernel_npu {

using namespace optiling;  // FA_*_INDEX / FD_*_INDEX / SAS_META_SIZE / SasMetaData

bool KvQuantSparseAttnSharedkvMetadataHost::Run(const int32_t *actSeqLenQ, const int32_t *actSeqLenOriKv,
                                                const int32_t *actSeqLenCmpKv, const int32_t *seqUsedQ,
                                                const int32_t *seqUsedKv, int32_t batchSize, int32_t querySeqSize,
                                                int32_t queryHeadNum, int32_t kvSeqSize, int32_t kvHeadNum,
                                                int32_t headDim, int32_t oriTopK, int32_t cmpTopK, int32_t cmpRatio,
                                                int32_t oriMaskMode, int32_t cmpMaskMode, int64_t winLeft,
                                                int64_t winRight, const std::string &socVersion,
                                                const std::string &layoutQuery, const std::string &layoutKv,
                                                bool hasOriKv, bool hasCmpKv, uint32_t aicCoreNum,
                                                uint32_t aivCoreNum, int32_t *metaData)
{
    actSeqLenQ_ = actSeqLenQ;
    actSeqLenOriKv_ = actSeqLenOriKv;
    actSeqLenCmpKv_ = actSeqLenCmpKv;
    seqUsedQ_ = seqUsedQ;
    seqUsedKv_ = seqUsedKv;
    metaData_ = metaData;
    batchSize_ = batchSize;
    querySeqSize_ = querySeqSize;
    queryHeadNum_ = queryHeadNum;
    kvSeqSize_ = kvSeqSize;
    kvHeadNum_ = kvHeadNum;
    headDim_ = headDim;
    oriTopK_ = oriTopK;
    cmpTopK_ = cmpTopK;
    cmpRatio_ = cmpRatio;
    oriMaskMode_ = oriMaskMode;
    cmpMaskMode_ = cmpMaskMode;
    winLeft_ = winLeft;
    winRight_ = winRight;
    socVersion_ = socVersion;
    layoutQuery_ = layoutQuery;
    layoutKv_ = layoutKv;
    hasOriKv_ = hasOriKv;
    hasCmpKv_ = hasCmpKv;
    aicCoreNum_ = aicCoreNum;
    aivCoreNum_ = aivCoreNum;

    if (metaData_ == nullptr || aicCoreNum_ == 0U || aivCoreNum_ == 0U) {
        return false;
    }
    // Zero the output table so a partially-ported GenMetaData cannot leave stale data.
    std::fill(metaData_, metaData_ + optiling::SAS_META_SIZE, 0);

    if (!Prepare()) {
        return false;
    }
    SplitResult splitRes{aicCoreNum_, aivCoreNum_};
    // TODO(port): port BalanceSchedule() + GenMetaData() from
    // kv_quant_sparse_attn_sharedkv_metadata_aicpu.cpp.orig:299-1091.
    return BalanceSchedule(splitRes) && GenMetaData(splitRes);
}

// ---------------- Prepare: validation + init (ported faithfully) ---------------

bool KvQuantSparseAttnSharedkvMetadataHost::Prepare()
{
    return ParamsCheck() && ParamsInit();
}

bool KvQuantSparseAttnSharedkvMetadataHost::CheckSingleParam()
{
    if (metaData_ == nullptr) {
        return false;
    }
    if (aicCoreNum_ == 0U || aivCoreNum_ == 0U) {
        return false;
    }
    if (batchSize_ < 0) {
        return false;
    }
    if (querySeqSize_ < 0) {
        return false;
    }
    if (queryHeadNum_ != 64 && queryHeadNum_ != 128) {
        return false;
    }
    if (kvHeadNum_ != 1) {
        return false;
    }
    if (oriMaskMode_ != static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
        oriMaskMode_ != static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL) &&
        oriMaskMode_ != static_cast<int32_t>(SparseMode::BAND)) {
        return false;
    }
    if (winLeft_ < 0) {
        return false;
    }
    if (layoutQuery_ != "TND" && layoutQuery_ != "BSND") {
        return false;
    }
    if (layoutKv_ != "PA_ND" && layoutKv_ != "TND" && layoutKv_ != "BSND") {
        return false;
    }
    if (layoutQuery_ == "TND" && layoutKv_ == "BSND") {
        return false;
    }
    if (layoutQuery_ == "BSND" && layoutKv_ == "TND") {
        return false;
    }
    return true;
}

bool KvQuantSparseAttnSharedkvMetadataHost::CheckExistence()
{
    auto isInvalid = [](const int32_t *p) { return p == nullptr; };
    if (layoutQuery_ == "TND") {
        if (isInvalid(actSeqLenQ_)) {
            return false;
        }
    }
    if (isInvalid(seqUsedKv_)) {
        return false;
    }
    return true;
}

bool KvQuantSparseAttnSharedkvMetadataHost::CheckConsistency()
{
    return true;
}

bool KvQuantSparseAttnSharedkvMetadataHost::CheckFeature()
{
    if (hasCmpKv_) {
        if (cmpRatio_ != 4 && cmpRatio_ != 128) {
            return false;
        }
        if (cmpTopK_ != 0 && cmpTopK_ != 512 && cmpTopK_ != 1024) {
            return false;
        }
    }
    return true;
}

bool KvQuantSparseAttnSharedkvMetadataHost::ParamsCheck()
{
    return CheckSingleParam() && CheckExistence() && CheckConsistency() && CheckFeature();
}

ValidSocVersion KvQuantSparseAttnSharedkvMetadataHost::ProcessSocVersion()
{
    if (socVersion_.find("Ascend950") != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    }
    return ValidSocVersion::ASCEND910;
}

int32_t KvQuantSparseAttnSharedkvMetadataHost::GetQueryBatchSize()
{
    if (seqUsedQ_ != nullptr) {
        return batchSize_;  // batch inferred upstream; seqUsedQ carries per-batch q lens
    }
    if (layoutQuery_ == "TND" && actSeqLenQ_ != nullptr) {
        return batchSize_;  // (actSeqLenQ dims - 1) already resolved into batchSize_ by caller
    }
    return batchSize_;
}

int32_t KvQuantSparseAttnSharedkvMetadataHost::GetKvBatchSize()
{
    return batchSize_;
}

bool KvQuantSparseAttnSharedkvMetadataHost::ParamsInit()
{
    batchSize_ = GetQueryBatchSize();
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    if (mode == SparseMode::DEFAULT_MASK) {
        preToken_ = INT64_MAX;
        nextToken_ = INT64_MAX;
        attentionMode_ = 0;
    } else if (mode == SparseMode::RIGHT_DOWN_CAUSAL) {
        preToken_ = INT64_MAX;
        nextToken_ = 0;
        attentionMode_ = 1;
    } else {  // BAND
        preToken_ = (winLeft_ > -1) ? winLeft_ : INT64_MAX;
        nextToken_ = 0;
        attentionMode_ = 1;
    }
    isS1G_ = (layoutQuery_ == "BSND" || layoutQuery_ == "TND");
    groupSize_ = queryHeadNum_ / kvHeadNum_;
    if (queryHeadNum_ == 128) {
        isN128_ = true;
    }
    if (hasCmpKv_) {
        if (cmpTopK_ > 0) {
            isSCFA_ = true;
        } else {
            isCFA_ = true;
        }
    }
    ValidSocVersion validSocVersion = ProcessSocVersion();
    if (validSocVersion == ValidSocVersion::ASCEND910) {
        uint32_t mBaseBlockLen = 128U;
        uint32_t s1BlockLen = mBaseBlockLen / groupSize_;
        if (isSCFA_) {
            s1BlockLen = 1U;
        }
        mBaseSize_ = groupSize_ * s1BlockLen;
        s2BaseSize_ = 512U;
    } else if (validSocVersion == ValidSocVersion::ASCEND950) {
        if (isN128_) {
            mBaseSize_ = groupSize_;
            aicCoreNum_ /= 2U;
            aivCoreNum_ /= 2U;
        } else {
            if (isSCFA_) {
                mBaseSize_ = 1U * groupSize_;
            } else {
                mBaseSize_ = 64U / groupSize_ * groupSize_;
            }
        }
        s2BaseSize_ = 128U;
    }
    return true;
}

uint32_t KvQuantSparseAttnSharedkvMetadataHost::GetS1SeqSize(uint32_t bIdx)
{
    if (seqUsedQ_ != nullptr) {
        return static_cast<uint32_t>(seqUsedQ_[bIdx]);
    }
    if (layoutQuery_ == "TND" && actSeqLenQ_ != nullptr) {
        return static_cast<uint32_t>(actSeqLenQ_[bIdx + 1U] - actSeqLenQ_[bIdx]);
    }
    return static_cast<uint32_t>(querySeqSize_);
}

uint32_t KvQuantSparseAttnSharedkvMetadataHost::GetS2SeqSize(uint32_t bIdx)
{
    if (seqUsedKv_ != nullptr) {
        return static_cast<uint32_t>(seqUsedKv_[bIdx]);
    }
    if (layoutKv_ == "TND" && actSeqLenOriKv_ != nullptr) {
        return static_cast<uint32_t>(actSeqLenOriKv_[bIdx + 1U] - actSeqLenOriKv_[bIdx]);
    }
    return static_cast<uint32_t>(kvSeqSize_);
}

// ---- Scheduling algorithm body (ported from .orig lines 299-1084) ------------

void KvQuantSparseAttnSharedkvMetadataHost::CalcSplitInfo(SplitContext &splitContext)
{
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        uint32_t s2Size = GetS2SeqSize(bIdx);
        splitInfo.s1GBaseNum[bIdx] = (s1Size * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (s1Size * groupSize_) % mBaseSize_;
        splitInfo.s2BaseNum[bIdx] = (s2Size + s2BaseSize_ - 1U) / s2BaseSize_;
        splitInfo.s2TailSize[bIdx] = s2Size % s2BaseSize_;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

int64_t KvQuantSparseAttnSharedkvMetadataHost::CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t KvQuantSparseAttnSharedkvMetadataHost::CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return nextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + nextToken_;
        default:
            return nextToken_;
    }
}

int64_t KvQuantSparseAttnSharedkvMetadataHost::WinCalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t winAlignCoefM = 16U;
    uint32_t winAlignCoefS2 = 64U;
    uint32_t winAlignBasicM = (basicM + winAlignCoefM - 1U) >> 4U;
    uint32_t winAlignBasicS2 = (basicS2 + winAlignCoefS2 - 1U) >> 6U;
    return static_cast<int64_t>(6U * winAlignBasicM + 10U * winAlignBasicS2);
}

int64_t KvQuantSparseAttnSharedkvMetadataHost::CmpCalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t cmpAlignCoefM = 16U;
    uint32_t cmpAlignCoefS2 = 64U;
    uint32_t cmpAlignBasicM = (basicM + cmpAlignCoefM - 1U) >> 4U;
    uint32_t cmpAlignBasicS2 = (basicS2 + cmpAlignCoefS2 - 1U) >> 6U;
    return static_cast<int64_t>(6U * cmpAlignBasicM + 10U * cmpAlignBasicS2);
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
                                                          uint32_t s1GTailSize, uint32_t winS2TailSize,
                                                          uint32_t cmpS2TailSize)
{
    typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK] = WinCalcCost(s1NormalSize, s2NormalSize);
    typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : WinCalcCost(s1GTailSize, s2NormalSize);
    typeCost_[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] = (winS2TailSize == 0U) ? 0U : WinCalcCost(s1NormalSize, winS2TailSize);
    typeCost_[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] =
        (s1GTailSize == 0U || winS2TailSize == 0U) ? 0U : WinCalcCost(s1GTailSize, winS2TailSize);
    if (hasCmpKv_) {
        typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] = CmpCalcCost(s1NormalSize, s2NormalSize);
        typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, s2NormalSize);
        typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] = (cmpS2TailSize == 0U) ? 0U : CmpCalcCost(s1NormalSize, cmpS2TailSize);
        typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] =
            (s1GTailSize == 0U || cmpS2TailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, cmpS2TailSize);
    }
}

Range<int64_t> KvQuantSparseAttnSharedkvMetadataHost::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache)
{
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(0, 0);
    }
    if (!attentionMode_) {
        return std::make_pair(0, static_cast<int64_t>(batchCache.s2Size) - 1);
    }
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken =
        std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
                 static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) -
        1;
    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (isS1G_) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(groupSize_);
        s1LastToken = s1GLastToken / static_cast<int64_t>(groupSize_);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }
    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;
    return std::make_pair(s2FirstToken, s2LastToken);
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext,
                                                            BatchCache &batchCache)
{
    (void)splitContext;
    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.s2Size = GetS2SeqSize(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcWinS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo)
{
    if (s1GCache.winS2Start >= s1GCache.winS2End) {
        s1GCache.winS1GBlock = 0;
        s1GCache.winS1GCost = 0;
        s1GCache.winS1GLastBlockCost = 0;
        s1GCache.winS1GNormalBlockCost = 0;
    } else {
        s1GCache.winS1GBlock = s1GCache.winS2End - s1GCache.winS2Start;
        uint32_t curWinTailS2Num = (s1GCache.winS2TailSize != 0U) ? 1U : 0U;
        uint32_t curWinNormalS2Num = s1GCache.winS1GBlock - curWinTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) && splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.winS1GCost = typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK] * curWinNormalS2Num +
                                  typeCost_[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] * curWinTailS2Num;
            s1GCache.winS1GLastBlockCost = curWinTailS2Num > 0U ? typeCost_[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK]
                                                                : typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK];
            s1GCache.winS1GNormalBlockCost = typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK];
        } else {
            s1GCache.winS1GCost = typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK] * curWinNormalS2Num +
                                  typeCost_[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] * curWinTailS2Num;
            s1GCache.winS1GLastBlockCost = curWinTailS2Num > 0U ? typeCost_[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK]
                                                                : typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK];
            s1GCache.winS1GNormalBlockCost = typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK];
        }
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcCmpS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo)
{
    if (s1GCache.cmpS2Start >= s1GCache.cmpS2End) {
        s1GCache.cmpS1GBlock = 0;
        s1GCache.cmpS1GCost = 0;
        s1GCache.cmpS1GLastBlockCost = 0;
        s1GCache.cmpS1GNormalBlockCost = 0;
    } else {
        s1GCache.cmpS1GBlock = s1GCache.cmpS2End - s1GCache.cmpS2Start;
        uint32_t curCmpTailS2Num = (s1GCache.cmpS2TailSize != 0U) ? 1U : 0U;
        uint32_t curCmpNormalS2Num = s1GCache.cmpS1GBlock - curCmpTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) && splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.cmpS1GCost = typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
                                  typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
            s1GCache.cmpS1GLastBlockCost = curCmpTailS2Num > 0U ? typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK]
                                                                : typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
            s1GCache.cmpS1GNormalBlockCost = typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
        } else {
            s1GCache.cmpS1GCost = typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
                                  typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
            s1GCache.cmpS1GLastBlockCost = curCmpTailS2Num > 0U ? typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK]
                                                                : typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
            s1GCache.cmpS1GNormalBlockCost = typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
        }
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcBlockRangeAndTailSize(Range<int64_t> &oriS2TokenRange,
                                                                      const BatchCache &batchCache, S1GCache &s1GCache)
{
    int64_t oriS2FirstToken = oriS2TokenRange.first;
    int64_t oriS2LastToken = oriS2TokenRange.second;
    if (oriS2FirstToken >= static_cast<int64_t>(batchCache.s2Size) || oriS2LastToken < 0 ||
        oriS2LastToken < oriS2FirstToken) {
        oriS2FirstToken = 0;
        oriS2LastToken = 0;
        s1GCache.winS2Start = 0;
        s1GCache.winS2End = 0;
        s1GCache.winS2TailSize = 0;
    } else {
        oriS2FirstToken = Clip(oriS2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
        oriS2LastToken = Clip(oriS2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
        s1GCache.winS2Start = 0;
        s1GCache.winS2End = (oriS2LastToken - oriS2FirstToken) / s2BaseSize_ + 1U;
        s1GCache.winS2TailSize = (oriS2LastToken - oriS2FirstToken + 1) % s2BaseSize_;
    }
    s1GCache.cmpS2Start = s1GCache.winS2End;
    uint32_t cmpS2LastTokenSize = hasCmpKv_ ? (oriS2LastToken + 1) / cmpRatio_ : 0;
    uint32_t actCmpS2LastTokenSize = 0;
    if (isCFA_) {
        actCmpS2LastTokenSize = cmpS2LastTokenSize;
    } else if (isSCFA_) {
        actCmpS2LastTokenSize = std::min(cmpS2LastTokenSize, static_cast<uint32_t>(cmpTopK_));
    }
    s1GCache.cmpS2End = (actCmpS2LastTokenSize == 0)
                            ? s1GCache.cmpS2Start
                            : s1GCache.cmpS2Start + (actCmpS2LastTokenSize - 1) / s2BaseSize_ + 1U;
    s1GCache.cmpS2TailSize = actCmpS2LastTokenSize % s2BaseSize_;
}

void KvQuantSparseAttnSharedkvMetadataHost::GatherWinAndCmpCache(S1GCache &s1GCache)
{
    s1GCache.s2Start = (s1GCache.winS1GBlock > 0) ? s1GCache.winS2Start : s1GCache.cmpS2Start;
    if (s1GCache.cmpS1GBlock > 0) {
        s1GCache.s1GLastBlockCost = s1GCache.cmpS1GLastBlockCost;
        s1GCache.s2End = s1GCache.cmpS2End;
    } else {
        s1GCache.s1GLastBlockCost = s1GCache.winS1GLastBlockCost;
        s1GCache.s2End = s1GCache.winS2End;
    }
    s1GCache.s1GBlock = s1GCache.winS1GBlock + s1GCache.cmpS1GBlock;
    s1GCache.s1GCost = s1GCache.winS1GCost + s1GCache.cmpS1GCost;
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext,
                                                         const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.winS1GNormalBlockCost = 0;
        s1GCache.winS1GLastBlockCost = 0;
        s1GCache.cmpS1GNormalBlockCost = 0;
        s1GCache.cmpS1GLastBlockCost = 0;
        s1GCache.s1GBlock = 0;
        s1GCache.s2Start = 0;
        s1GCache.cmpS2Start = 0;
        s1GCache.s2End = 0;
        return;
    }
    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;
    auto oriS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache);
    CalcBlockRangeAndTailSize(oriS2TokenRange, batchCache, s1GCache);
    CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[s1GCache.bIdx], s1GCache.winS2TailSize,
                  s1GCache.cmpS2TailSize);
    CalcWinS1GCache(s1GCache, splitInfo);
    CalcCmpS1GCache(s1GCache, splitInfo);
    GatherWinAndCmpCache(s1GCache);
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext,
                                                          CostInfo &costInfo)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;
    if (GetS1SeqSize(bIdx) == 0U || GetS2SeqSize(bIdx) == 0U) {
        return;
    }
    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
        if (s1GCache.s1GCost > costInfo.maxS1GCost) {
            costInfo.maxS1GCost = s1GCache.s1GCost;
        }
        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcCostInfo(SplitContext &splitContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    CostInfo &costInfo = splitContext.costInfo;
    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * kvHeadNum_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * kvHeadNum_;
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::UpdateCursor(const SplitContext &splitContext,
                                                         AssignContext &assignContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;
    bool UpdateS1G = false;
    bool UpdateBatch = false;
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) {
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }
    if (assignContext.curBN2Idx == batchSize_ * kvHeadNum_) {
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }
    if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = (supportFd_) ? assignContext.s1GCache.winS2Start : 0;
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::AssignByBatch(const SplitContext &splitContext,
                                                          AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo &costInfo = splitContext.costInfo;
    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;
        if (assignContext.curBN2Idx == batchSize_ * kvHeadNum_) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }
        if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost
                                    ? assignContext.bN2Cost - assignContext.s1GCache.s1GCost
                                    : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock
                                     ? assignContext.bN2Block - assignContext.s1GCache.s1GBlock
                                     : 0U;
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

int64_t KvQuantSparseAttnSharedkvMetadataHost::CalcCurBlockCost(AssignContext &assignContext)
{
    int64_t curCost = 0;
    if (assignContext.curS2Idx < assignContext.s1GCache.cmpS2Start) {
        curCost = assignContext.s1GCache.winS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GCache.cmpS2Start - 1U)) {
            curCost = assignContext.s1GCache.winS1GLastBlockCost;
        }
    } else {
        curCost = assignContext.s1GCache.cmpS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
            curCost = assignContext.s1GCache.cmpS1GLastBlockCost;
        }
    }
    return curCost;
}

void KvQuantSparseAttnSharedkvMetadataHost::AssignByBlock(const SplitContext &splitContext,
                                                          AssignContext &assignContext)
{
    if (assignContext.isFinished || !supportFd_) {
        return;
    }
    int64_t curCost = CalcCurBlockCost(assignContext);
    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + curCost)) {
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
        curCost = CalcCurBlockCost(assignContext);
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::ForceAssign(const SplitContext &splitContext,
                                                        AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    int64_t curCost = CalcCurBlockCost(assignContext);
    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext);
}

bool KvQuantSparseAttnSharedkvMetadataHost::IsNeedRecordFDInfo(const AssignContext &assignContext,
                                                               const SplitResult &splitRes)
{
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void KvQuantSparseAttnSharedkvMetadataHost::RecordFDInfo(const SplitContext &splitContext,
                                                         const AssignContext &assignContext, SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / kvHeadNum_;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U)
                                ? (s1Size * groupSize_ - splitS1GIdx * mBaseSize_)
                                : mBaseSize_;
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    result.fdRes.fdBN2Idx[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdMIdx[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdWorkspaceIdx[result.numOfFdHead] = assignContext.preFdDataNum;
    result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize;
    result.numOfFdHead++;
}

void KvQuantSparseAttnSharedkvMetadataHost::AssignBlocksToCore(const SplitContext &splitContext,
                                                               AssignContext &assignContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] =
        assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;
    int64_t avgCost = assignContext.unassignedCost / (aicCoreNum_ - assignContext.curCoreIdx);
    assignContext.coreCache = {};
    if (!supportFd_) {
        assignContext.coreCache.costLimit = std::max(avgCost, costInfo.maxS1GCost);
    } else {
        assignContext.coreCache.costLimit = avgCost;
    }
    AssignByBatch(splitContext, assignContext);
    AssignByRow(splitContext, assignContext);
    AssignByBlock(splitContext, assignContext);
    if (assignContext.coreCache.block == 0 && supportFd_) {
        ForceAssign(splitContext, assignContext);
    }
    result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
    result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
    result.s2End[assignContext.curCoreIdx] = assignContext.curS2Idx;
    result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
    assignContext.unassignedCost -= assignContext.coreCache.cost;
    result.maxS2GBaseNum = std::max(assignContext.coreCache.block, result.maxS2GBaseNum);
    if (IsNeedRecordFDInfo(assignContext, result)) {
        RecordFDInfo(splitContext, assignContext, result);
        assignContext.preFdDataNum += assignContext.curKvSplitPart;
        assignContext.curKvSplitPart = 1U;
    }
    if (assignContext.curS2Idx > assignContext.s1GCache.s2Start && assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
        assignContext.curKvSplitPart++;
    }
}

void KvQuantSparseAttnSharedkvMetadataHost::CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext,
                                                          SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    if (aicCoreNum_ == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;
    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    for (uint32_t i = 0; i < aicCoreNum_; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlocksToCore(splitContext, assignContext, result);
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void KvQuantSparseAttnSharedkvMetadataHost::SplitFD(SplitResult &splitRes)
{
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_;
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad;
        uint32_t curAveMSize = (splitRes.fdRes.fdMSize[i] + curFDVectorNum - 1U) / curFDVectorNum;
        curFDVectorNum = (splitRes.fdRes.fdMSize[i] + curAveMSize - 1U) / curAveMSize;
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
            splitRes.fdRes.fdMNum[curCoreIndex] =
                (vid < curFDVectorNum - 1) ? curAveMSize : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
            curCoreIndex++;
        }
    }
    splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}

bool KvQuantSparseAttnSharedkvMetadataHost::BalanceSchedule(SplitResult &splitRes)
{
    SplitContext splitContext(batchSize_);
    CalcSplitInfo(splitContext);
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = batchSize_ * kvHeadNum_;
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);
    splitRes.maxCost = INT64_MAX;
    splitRes.usedCoreNum = 1U;
    CalcSplitPlan(splitRes.maxCost, splitContext, splitRes);
    if (splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U);
    return true;
}

bool KvQuantSparseAttnSharedkvMetadataHost::GenMetaData(SplitResult &splitRes)
{
    auto *metaDataPtr = reinterpret_cast<optiling::detail::SasMetaData *>(metaData_);
    // FA Metadata Generate
    if (isN128_) {
        for (size_t i = 0; i < aicCoreNum_; i++) {
            if (i >= splitRes.usedCoreNum) {
                metaDataPtr->faMetadata[2 * i][FA_CORE_ENABLE_INDEX] = 0;
                metaDataPtr->faMetadata[2 * i + 1][FA_CORE_ENABLE_INDEX] = 0;
                metaDataPtr->faMetadata[2 * i][FA_S2_MAX_NUM] = splitRes.maxS2GBaseNum;
                metaDataPtr->faMetadata[2 * i + 1][FA_S2_MAX_NUM] = splitRes.maxS2GBaseNum;
                continue;
            }
            metaDataPtr->faMetadata[2 * i][FA_CORE_ENABLE_INDEX] = 1;
            metaDataPtr->faMetadata[2 * i + 1][FA_CORE_ENABLE_INDEX] = 1;
            metaDataPtr->faMetadata[2 * i][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
            metaDataPtr->faMetadata[2 * i][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            metaDataPtr->faMetadata[2 * i][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
            metaDataPtr->faMetadata[2 * i + 1][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
            metaDataPtr->faMetadata[2 * i + 1][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            metaDataPtr->faMetadata[2 * i + 1][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
            metaDataPtr->faMetadata[2 * i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
            metaDataPtr->faMetadata[2 * i][FA_M_END_INDEX] = splitRes.gS1End[i];
            metaDataPtr->faMetadata[2 * i][FA_S2_END_INDEX] = splitRes.s2End[i];
            metaDataPtr->faMetadata[2 * i + 1][FA_BN2_END_INDEX] = splitRes.bN2End[i];
            metaDataPtr->faMetadata[2 * i + 1][FA_M_END_INDEX] = splitRes.gS1End[i];
            metaDataPtr->faMetadata[2 * i + 1][FA_S2_END_INDEX] = splitRes.s2End[i];
            metaDataPtr->faMetadata[2 * i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
            metaDataPtr->faMetadata[2 * i + 1][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
            metaDataPtr->faMetadata[2 * i][FA_S2_MAX_NUM] = splitRes.maxS2GBaseNum;
            metaDataPtr->faMetadata[2 * i + 1][FA_S2_MAX_NUM] = splitRes.maxS2GBaseNum;
        }
    } else {
        for (size_t i = 0; i < aicCoreNum_; ++i) {
            if (i >= splitRes.usedCoreNum) {
                metaDataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 0;
                continue;
            }
            metaDataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 1;
            metaDataPtr->faMetadata[i][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
            metaDataPtr->faMetadata[i][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            metaDataPtr->faMetadata[i][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
            metaDataPtr->faMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
            metaDataPtr->faMetadata[i][FA_M_END_INDEX] = splitRes.gS1End[i];
            metaDataPtr->faMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];
            metaDataPtr->faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
        }
    }
    // FD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metaDataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 0;
            continue;
        }
        metaDataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 1;
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metaDataPtr->fdMetadata[i][FD_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metaDataPtr->fdMetadata[i][FD_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metaDataPtr->fdMetadata[i][FD_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metaDataPtr->fdMetadata[i][FD_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metaDataPtr->fdMetadata[i][FD_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metaDataPtr->fdMetadata[i][FD_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}

// Declared in the header but never called/defined in the AICPU source; kept as a no-op.
void KvQuantSparseAttnSharedkvMetadataHost::SplitCore() {}

}  // namespace sgl_kernel_npu

namespace sglang {
namespace npu_kernel {

namespace {
struct HostTopology {
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    std::string socVersion;
};

HostTopology ResolveHostTopology()
{
    static HostTopology cached = []() {
        int32_t device = 0;
        if (aclrtGetDevice(&device) != ACL_SUCCESS) {
            device = 0;  // fall back to device 0
        }
        int64_t aic = 0;
        int64_t aiv = 0;
        TORCH_CHECK(aclGetDeviceCapability(device, ACL_DEVICE_INFO_AI_CORE_NUM, &aic) == ACL_SUCCESS,
                    "kv_quant_sparse_attn_sharedkv_metadata_host: aclGetDeviceCapability(AI_CORE_NUM) failed");
        TORCH_CHECK(aclGetDeviceCapability(device, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv) == ACL_SUCCESS,
                    "kv_quant_sparse_attn_sharedkv_metadata_host: aclGetDeviceCapability(VECTOR_CORE_NUM) failed");
        TORCH_CHECK(aic > 0 && aiv > 0,
                    "kv_quant_sparse_attn_sharedkv_metadata_host: invalid device core counts "
                    "(aic=", aic, ", aiv=", aiv, ")");
        const char *soc = aclrtGetSocName();
        return HostTopology{static_cast<uint32_t>(aic), static_cast<uint32_t>(aiv),
                            (soc != nullptr) ? std::string(soc) : std::string()};
    }();
    return cached;
}
}  // namespace

at::Tensor kv_quant_sparse_attn_sharedkv_metadata_host(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, int64_t /*kv_quant_mode*/,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_kv,
    int64_t ori_topk, int64_t cmp_topk, int64_t /*tile_size*/, int64_t /*rope_head_dim*/, int64_t cmp_ratio,
    int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    c10::string_view layout_q, c10::string_view layout_kv, bool has_ori_kv, bool has_cmp_kv)
{
    auto opts = at::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true);
    at::Tensor metaDataHost = at::zeros({static_cast<int64_t>(optiling::SAS_META_SIZE)}, opts);

    auto rawPtr = [](const c10::optional<at::Tensor> &t, const char *name) -> const int32_t * {
        if (!t.has_value()) {
            return nullptr;
        }
        const at::Tensor &tensor = *t;
        TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
        TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor (host metadata path)");
        return static_cast<const int32_t *>(tensor.const_data_ptr());
    };
    const int32_t *cuQ = rawPtr(cu_seqlens_q, "cu_seqlens_q");
    const int32_t *cuOriKv = rawPtr(cu_seqlens_ori_kv, "cu_seqlens_ori_kv");
    const int32_t *cuCmpKv = rawPtr(cu_seqlens_cmp_kv, "cu_seqlens_cmp_kv");
    const int32_t *seqQ = rawPtr(seqused_q, "seqused_q");
    const int32_t *seqKv = rawPtr(seqused_kv, "seqused_kv");

    const HostTopology &topo = ResolveHostTopology();
    sgl_kernel_npu::KvQuantSparseAttnSharedkvMetadataHost scheduler;
    bool ok = scheduler.Run(cuQ, cuOriKv, cuCmpKv, seqQ, seqKv, static_cast<int32_t>(batch_size),
                            static_cast<int32_t>(max_seqlen_q), static_cast<int32_t>(num_heads_q),
                            static_cast<int32_t>(max_seqlen_kv), static_cast<int32_t>(num_heads_kv),
                            static_cast<int32_t>(head_dim), static_cast<int32_t>(ori_topk),
                            static_cast<int32_t>(cmp_topk), static_cast<int32_t>(cmp_ratio),
                            static_cast<int32_t>(ori_mask_mode), static_cast<int32_t>(cmp_mask_mode), ori_win_left,
                            ori_win_right, topo.socVersion, std::string(layout_q), std::string(layout_kv), has_ori_kv,
                            has_cmp_kv, topo.aicCoreNum, topo.aivCoreNum, metaDataHost.data_ptr<int32_t>());
    TORCH_CHECK(ok, "kv_quant_sparse_attn_sharedkv_metadata_host: scheduling failed (invalid params)");
    return metaDataHost.to(at::Device("npu"), /*non_blocking=*/true);
}

}  // namespace npu_kernel
}  // namespace sglang
