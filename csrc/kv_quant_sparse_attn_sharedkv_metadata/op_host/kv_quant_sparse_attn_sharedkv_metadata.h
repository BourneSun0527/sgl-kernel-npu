/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Host-side (CPU) reimplementation of the vllm-ascend AICPU op
 * `KvQuantSparseAttnSharedkvMetadata` (csrc/attention/
 * kv_quant_sparse_attn_sharedkv_metadata), following the same porting playbook used
 * for the non-quant `SparseAttnSharedkvMetadataHost`: only the CpuKernelContext /
 * GetAttrValue / Tensor I/O shim is replaced by plain pointers plus at::Tensor
 * allocation; the scheduling algorithm body is preserved.
 */

#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_HOST_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_HOST_H

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>

// Fixed output buffer layout. The consumer AICore kernel indexes the flat
// int32[1024] buffer with these exact strides (GetAttrAbsIndex), so they must not
// change. The kv_quant (A5) layout uses 9 ints per FA core (the extra FA_S2_MAX_NUM
// slot) vs 8 in the non-quant variant.
namespace optiling {
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t SAS_META_SIZE = 1024;
using SAS_METADATA_T = int32_t;

constexpr uint32_t FA_METADATA_SIZE = 9;
constexpr uint32_t FD_METADATA_SIZE = 8;

constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr uint32_t FA_S2_MAX_NUM = 8;

constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

namespace detail {
struct SasMetaData {
    uint32_t faMetadata[AIC_CORE_NUM][FA_METADATA_SIZE];
    uint32_t fdMetadata[AIV_CORE_NUM][FD_METADATA_SIZE];
};
}  // namespace detail
static_assert(SAS_META_SIZE * sizeof(SAS_METADATA_T) >= sizeof(detail::SasMetaData),
              "metadata buffer must hold the full SasMetaData layout");
}  // namespace optiling

namespace sgl_kernel_npu {

constexpr int64_t FA_TOLERANCE_RATIO = 2;

enum BlockType : uint32_t { WIN_NORMAL_BLOCK = 0, WIN_TAIL_BLOCK, CMP_NORMAL_BLOCK, CMP_TAIL_BLOCK, BLOCK_MAX_TYPE };

enum class SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
};

enum class ValidSocVersion { ASCEND910 = 0, ASCEND950, RESERVED_VERSION = 99999 };

template <class T>
using Range = std::pair<T, T>;

template <class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template <typename T>
T Clip(T value, T minValue, T maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

template <typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

struct FlashDecodeResult {
    uint32_t fdUsedVecNum{0U};
    std::vector<uint32_t> fdBN2Idx{};
    std::vector<uint32_t> fdMIdx{};
    std::vector<uint32_t> fdWorkspaceIdx{};
    std::vector<uint32_t> fdS2SplitNum{};
    std::vector<uint32_t> fdMSize{};
    std::vector<uint32_t> fdIdx{};
    std::vector<uint32_t> fdMStart{};
    std::vector<uint32_t> fdMNum{};

    FlashDecodeResult(uint32_t aicNum, uint32_t aivNum)
        : fdBN2Idx(aicNum),
          fdMIdx(aicNum),
          fdWorkspaceIdx(aicNum),
          fdS2SplitNum(aicNum),
          fdMSize(aicNum),
          fdIdx(aivNum),
          fdMStart(aivNum),
          fdMNum(aivNum)
    {}
};

struct SplitResult {
    uint32_t usedCoreNum{0U};
    std::vector<uint32_t> bN2End{};
    std::vector<uint32_t> gS1End{};
    std::vector<uint32_t> s2End{};
    std::vector<uint32_t> firstFdDataWorkspaceIdx{};
    int64_t maxCost{0};
    uint32_t numOfFdHead{0U};
    uint32_t maxS2SplitNum{0U};
    uint32_t maxS2GBaseNum{0U};
    FlashDecodeResult fdRes{0U, 0U};

    SplitResult(uint32_t aicNum, uint32_t aivNum)
        : bN2End(aicNum), gS1End(aicNum), s2End(aicNum), firstFdDataWorkspaceIdx(aicNum), fdRes(aicNum, aivNum)
    {}
};

struct SplitInfo {
    std::vector<uint32_t> s1GBaseNum{};
    std::vector<uint32_t> s2BaseNum{};
    std::vector<uint32_t> s1GTailSize{};
    std::vector<uint32_t> s2TailSize{};
    bool isKvSeqAllZero{true};

    explicit SplitInfo(uint32_t batchSize)
        : s1GBaseNum(batchSize), s2BaseNum(batchSize), s1GTailSize(batchSize), s2TailSize(batchSize)
    {}
};

struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch{};
    std::vector<uint32_t> bN2BlockOfEachBatch{};
    std::vector<int64_t> bN2LastBlockCostOfEachBatch{};
    uint32_t totalBlockNum{0U};
    int64_t totalCost{0};
    int64_t maxS1GCost{0};

    explicit CostInfo(uint32_t batchSize)
        : bN2CostOfEachBatch(batchSize), bN2BlockOfEachBatch(batchSize), bN2LastBlockCostOfEachBatch(batchSize)
    {}
};

struct SplitContext {
    SplitInfo splitInfo{0U};
    CostInfo costInfo{0U};

    explicit SplitContext(uint32_t batchSize) : splitInfo(batchSize), costInfo(batchSize) {}
};

struct BatchCache {
    uint32_t bIdx{0U};
    uint32_t s1Size{0U};
    uint32_t s2Size{0U};
    int64_t preTokenLeftUp{0};
    int64_t nextTokenLeftUp{0};
    BlockCost<int64_t> typeCost{};
};

struct S1GCache {
    uint32_t bIdx{0U};
    uint32_t s1GIdx{0U};
    uint32_t s2Start{0U};
    uint32_t s2End{0U};
    uint32_t winS2Start{0U};
    uint32_t winS2End{0U};
    uint32_t cmpS2Start{0U};
    uint32_t cmpS2End{0U};
    int64_t s1GCost{0};
    int64_t s1GLastBlockCost{0};
    uint32_t s1GBlock{0U};
    int64_t s1GNormalBlockCost{0};
    uint32_t winS1GBlock{0U};
    int64_t winS1GCost{0};
    int64_t winS1GLastBlockCost{0};
    int64_t winS1GNormalBlockCost{0};
    uint32_t cmpS1GBlock{0U};
    int64_t cmpS1GCost{0};
    int64_t cmpS1GLastBlockCost{0};
    int64_t cmpS1GNormalBlockCost{0};
    int64_t cmpS2TailSize{0};
    int64_t winS2TailSize{0};
};

struct CoreCache {
    int64_t costLimit{0};
    int64_t cost{0};
    uint32_t block{0U};
};

struct AssignContext {
    uint32_t curBIdx{0U};
    uint32_t curBN2Idx{0U};
    uint32_t curS1GIdx{0U};
    uint32_t curS2Idx{0U};
    uint32_t curCoreIdx{0U};
    int64_t unassignedCost{0};
    uint32_t usedCoreNum{0U};
    uint32_t curKvSplitPart{1U};
    uint32_t preFdDataNum{0U};

    int64_t bN2Cost{0};
    uint32_t bN2Block{0U};
    bool isFinished{false};
    BatchCache batchCache{};
    S1GCache s1GCache{};
    CoreCache coreCache{};
};

// Pure-host core-distribution scheduler for the kv_quant (A5 FP8) sparse-attention
// variant. Inputs are CPU int32 arrays (null when the optional tensor is absent);
// output is written into the caller-provided int32[1024] buffer (zero-initialized).
class KvQuantSparseAttnSharedkvMetadataHost
{
public:
    KvQuantSparseAttnSharedkvMetadataHost() = default;
    ~KvQuantSparseAttnSharedkvMetadataHost() = default;

    bool Run(const int32_t *actSeqLenQ, const int32_t *actSeqLenOriKv, const int32_t *actSeqLenCmpKv,
             const int32_t *seqUsedQ, const int32_t *seqUsedKv, int32_t batchSize, int32_t querySeqSize,
             int32_t queryHeadNum, int32_t kvSeqSize, int32_t kvHeadNum, int32_t headDim, int32_t oriTopK,
             int32_t cmpTopK, int32_t cmpRatio, int32_t oriMaskMode, int32_t cmpMaskMode, int64_t winLeft,
             int64_t winRight, const std::string &socVersion, const std::string &layoutQuery,
             const std::string &layoutKv, bool hasOriKv, bool hasCmpKv, uint32_t aicCoreNum, uint32_t aivCoreNum,
             int32_t *metaData);

private:
    bool Prepare();
    int32_t GetQueryBatchSize();
    int32_t GetKvBatchSize();
    bool CheckSingleParam();
    bool CheckExistence();
    bool CheckConsistency();
    bool CheckFeature();
    bool ParamsCheck();
    bool ParamsInit();
    bool BalanceSchedule(SplitResult &splitRes);
    bool GenMetaData(SplitResult &splitRes);
    ValidSocVersion ProcessSocVersion();

    uint32_t GetS1SeqSize(uint32_t bIdx);
    uint32_t GetS2SeqSize(uint32_t bIdx);
    int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    Range<int64_t> CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache);
    int64_t WinCalcCost(uint32_t basicM, uint32_t basicS2);
    int64_t CmpCalcCost(uint32_t basicM, uint32_t basicS2);
    void CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t winS2TailSize,
                       uint32_t cmpS2TailSize);

    void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
    void CalcBlockRangeAndTailSize(Range<int64_t> &oriS2TokenRange, const BatchCache &batchCache, S1GCache &s1GCache);
    void CalcWinS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo);
    void CalcCmpS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo);
    void GatherWinAndCmpCache(S1GCache &s1GCache);
    void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache,
                      S1GCache &s1GCache);

    void CalcSplitInfo(SplitContext &splitContext);
    void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo);
    void CalcCostInfo(SplitContext &splitContext);

    void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext);
    int64_t CalcCurBlockCost(AssignContext &assignContext);
    void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext);
    void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignBlocksToCore(const SplitContext &splitContext, AssignContext &assignContext, SplitResult &result);

    bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes);
    void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result);

    void SplitFD(SplitResult &splitRes);
    void CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext, SplitResult &result);
    void SplitCore();

    // Inputs (raw CPU pointers; null when the optional tensor was not provided).
    const int32_t *actSeqLenQ_{nullptr};
    const int32_t *actSeqLenOriKv_{nullptr};
    const int32_t *actSeqLenCmpKv_{nullptr};
    const int32_t *seqUsedQ_{nullptr};
    const int32_t *seqUsedKv_{nullptr};

    int32_t *metaData_{nullptr};

    // Attributes.
    int32_t batchSize_{0};
    int32_t querySeqSize_{0};
    int32_t queryHeadNum_{0};
    int32_t kvSeqSize_{0};
    int32_t kvHeadNum_{0};
    int32_t headDim_{0};
    int32_t oriTopK_{0};
    int32_t cmpTopK_{0};
    int32_t cmpRatio_{-1};
    int32_t oriMaskMode_{4};
    int32_t cmpMaskMode_{3};
    int64_t winLeft_{127};
    int64_t winRight_{0};
    std::string layoutQuery_{"BSND"};
    std::string layoutKv_{"PA_ND"};
    bool hasOriKv_{true};
    bool hasCmpKv_{true};
    uint32_t aicCoreNum_{24U};
    uint32_t aivCoreNum_{48U};

    std::string socVersion_{"ascend910B"};
    int64_t preToken_{0};
    int64_t nextToken_{0};
    uint32_t groupSize_{0};
    uint32_t mBaseSize_{0};
    uint32_t s2BaseSize_{0};
    bool isS1G_{true};
    bool isCFA_{false};
    bool isSCFA_{false};
    bool supportFd_{false};
    uint32_t sparseMode_{0};
    uint32_t attentionMode_{1};
    BlockCost<int64_t> typeCost_;
    bool isN128_{false};
};

}  // namespace sgl_kernel_npu

#endif  // KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_HOST_H
