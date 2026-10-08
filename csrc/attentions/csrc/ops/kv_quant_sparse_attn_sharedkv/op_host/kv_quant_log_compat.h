#ifndef KV_QUANT_LOG_COMPAT_H
#define KV_QUANT_LOG_COMPAT_H

// vllm-ascend's check/tiling use older CANN macro names (OP_LOGE/OP_LOGI/OP_CHECK_IF);
// the attentions shim (log/ops_log.h + error/ops_error.h) only provides the OPS_LOG_*/
// OP_CHECK/OPS_ERR_IF names. Map old -> new so the vendored code compiles unchanged.
// Must be included AFTER log/ops_log.h and error/ops_error.h.
#ifndef OP_LOGE
#define OP_LOGE(...) OPS_LOG_E(__VA_ARGS__)
#endif
#ifndef OP_LOGI
#define OP_LOGI(...) OPS_LOG_I(__VA_ARGS__)
#endif
#ifndef OP_LOGW
#define OP_LOGW(...) OPS_LOG_W(__VA_ARGS__)
#endif
#ifndef OP_LOGD
#define OP_LOGD(...) OPS_LOG_D(__VA_ARGS__)
#endif
#ifndef OP_CHECK_IF
#define OP_CHECK_IF(COND, LOG_FUNC, EXPR) OP_CHECK(COND, LOG_FUNC, EXPR)
#endif
#ifndef OP_ERR_IF
#define OP_ERR_IF(COND, LOG_FUNC, EXPR) OPS_ERR_IF(COND, LOG_FUNC, EXPR)
#endif

#endif  // KV_QUANT_LOG_COMPAT_H
