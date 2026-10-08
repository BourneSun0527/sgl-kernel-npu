# opbuild CMakeLists for kv_quant_sparse_attn_sharedkv (A5 / ascend950, FP8 KV cache).
# Mirrors laser_attention/op_host/CMakeLists.txt; the op_host sources (def/proto/tiling/
# check) are vendored unchanged from vllm-ascend and use the CANN op framework
# (OP_ADD / gert::TilingContext / BEGIN_TILING_DATA_DEF) natively supported by opbuild.
add_ops_compile_options(
        OP_NAME KvQuantSparseAttnSharedkv
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
)

set(kv_quant_sparse_attn_sharedkv_depends kv_quant_sparse_attn_sharedkv PARENT_SCOPE)

target_sources(op_host_aclnn PRIVATE
        kv_quant_sparse_attn_sharedkv_def.cpp
)

target_sources(optiling PRIVATE
        kv_quant_sparse_attn_sharedkv_check_consistancy.cpp
        kv_quant_sparse_attn_sharedkv_check_existance.cpp
        kv_quant_sparse_attn_sharedkv_check_feature.cpp
        kv_quant_sparse_attn_sharedkv_check_single_para.cpp
        kv_quant_sparse_attn_sharedkv_check.cpp
        kv_quant_sparse_attn_sharedkv_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            kv_quant_sparse_attn_sharedkv_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        kv_quant_sparse_attn_sharedkv_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_CANN_PACKAGE_PATH}/include
)
