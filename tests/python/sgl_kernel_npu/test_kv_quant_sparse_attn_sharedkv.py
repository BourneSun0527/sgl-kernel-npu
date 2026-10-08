"""Smoke tests for the kv_quant (A5 / ascend950, FP8 KV cache) sparse-attention pair.

These ops are A5-only. The metadata op is a pure-host scheduler that returns a
device int32[1024] core-distribution table; the attention op is the AICore SCFA
kernel that consumes it. This file covers:

  * op registration (both ops are importable via torch.ops.npu.*),
  * metadata output contract (int32, 1024 elements, on NPU),
  * metadata input-contract rejection (invalid cmp_topk / cmp_ratio),
  * a guarded attention-op launch smoke test (the SCFA kernel + tiling need real
    FP8 KV-cache tensors, so the call is only exercised on A5 with valid inputs).

Run:
    python tests/python/sgl_kernel_npu/test_kv_quant_sparse_attn_sharedkv.py
"""

import unittest

import sgl_kernel_npu  # noqa: F401  (loads libsgl_kernel_npu.so -> registers torch.ops.npu.*)
import attentions  # noqa: F401  (loads libPTAExtensionOPS.so -> registers torch.ops.attentions.*)
import torch
import torch_npu

META_OP = torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host

# The main attention op is registered in the attentions namespace (built via the opbuild
# pipeline). Guard it so the module still imports for the metadata tests if attentions
# is not installed.
try:
    ATTN_OP = torch.ops.attentions.kv_quant_sparse_attn_sharedkv
except Exception:  # op not registered
    ATTN_OP = None

_META_SLOTS = 1024  # SAS_META_SIZE


def _is_a5():
    try:
        props = torch.npu.get_device_properties(0)
        # Ascend950 family (Ascend950 / Ascend950DT / Ascend950PR / ...) is A5. Cube-core
        # count varies by SKU (32 or 36), so detect by SoC name, not a fixed count.
        return "Ascend950" in str(getattr(props, "name", ""))
    except Exception:
        return False


@unittest.skipUnless(_is_a5(), "kv_quant_sparse_attn_sharedkv is A5 (Ascend950) only")
class KvQuantSparseAttnSharedkvMetadataHostTest(unittest.TestCase):
    """Output-contract + input-rejection checks for the host metadata scheduler."""

    def test_output_contract(self):
        # decode: batch=4, seqused_kv gives per-batch kv lens.
        seqused_kv = torch.tensor([128, 256, 512, 1024], dtype=torch.int32)
        out = META_OP(
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=512,
            kv_quant_mode=1,
            cu_seqlens_q=None,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            batch_size=4,
            max_seqlen_q=1,
            max_seqlen_kv=1024,
            ori_topk=0,
            cmp_topk=512,
            tile_size=64,
            rope_head_dim=64,
            cmp_ratio=4,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="BSND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=True,
        )
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.numel(), _META_SLOTS)
        self.assertEqual(out.device.type, "npu")

    def test_rejects_invalid_cmp_topk(self):
        seqused_kv = torch.tensor([128, 256], dtype=torch.int32)
        with self.assertRaises(Exception):
            META_OP(
                64, 1, 512, 1, None, None, None, None, seqused_kv, 2, 1, 1024, 0, 7, 64, 64, 4, 4, 3, 127, 0,
                "BSND", "PA_ND", True, True,
            )

    def test_rejects_invalid_cmp_ratio(self):
        seqused_kv = torch.tensor([128], dtype=torch.int32)
        with self.assertRaises(Exception):
            META_OP(
                64, 1, 512, 1, None, None, None, None, seqused_kv, 1, 1, 128, 0, 512, 64, 64, 7, 4, 3, 127, 0,
                "BSND", "PA_ND", True, True,
            )


@unittest.skipUnless(_is_a5() and ATTN_OP is not None,
                     "kv_quant_sparse_attn_sharedkv main op is disabled until the arch35 kernel port")
class KvQuantSparseAttnSharedkvTest(unittest.TestCase):
    """Attention-op registration + a guarded launch smoke test."""

    def test_op_is_registered(self):
        # The op handle resolves (schema + PrivateUse1 impl are wired in the build).
        self.assertTrue(callable(ATTN_OP))

    def test_launch_smoke(self):
        # Minimal end-to-end launch: produces an output shaped like q.
        # NOTE: the host tiling is a best-effort inline port (see op_host/...cpp);
        # this only asserts the call path runs, NOT numerical correctness.
        import torch.nn.functional as F

        b, s1, n, d = 1, 1, 64, 512
        kv_block_size = 128
        kv_blocks = 8
        q = torch.randn(b, s1, n, d, dtype=torch.bfloat16, device="npu")
        # FP8_E4M3FN KV cache (per-tile quant, mode=1): ori [blocks, block, 1, d+rope].
        # NPU's randn does not support fp8_e4m3fn directly, so generate bf16 then cast.
        ori_kv = torch.randn(kv_blocks, kv_block_size, 1, d + 64, dtype=torch.bfloat16,
                             device="npu").to(torch.float8_e4m3fn)
        cmp_kv = torch.randn(kv_blocks, kv_block_size, 1, (d + 64) // 4, dtype=torch.bfloat16,
                             device="npu").to(torch.float8_e4m3fn)
        ori_block_table = torch.arange(kv_blocks, dtype=torch.int32, device="npu").reshape(1, kv_blocks)
        cmp_block_table = torch.arange(kv_blocks, dtype=torch.int32, device="npu").reshape(1, kv_blocks)
        seqused_kv = torch.tensor([kv_block_size * kv_blocks], dtype=torch.int32, device="npu")
        sinks = torch.zeros(b, dtype=torch.float32, device="npu")

        metadata = META_OP(
            64, 1, 512, 1, None, None, None, None, seqused_kv.cpu(), b, s1, kv_block_size * kv_blocks,
            0, 512, 64, 64, 4, 4, 3, 127, 0, "BSND", "PA_ND", True, True,
        )

        out, lse = ATTN_OP(
            q, 1,
            ori_kv=ori_kv, cmp_kv=cmp_kv,
            ori_sparse_indices=None, cmp_sparse_indices=None,
            ori_block_table=ori_block_table, cmp_block_table=cmp_block_table,
            cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
            seqused_q=None, seqused_kv=seqused_kv, sinks=sinks, metadata=metadata,
            tile_size=64, rope_head_dim=64, softmax_scale=1.0, cmp_ratio=4,
            ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
            layout_q="BSND", layout_kv="PA_ND", return_softmax_lse=False,
        )
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.dtype, q.dtype)


if __name__ == "__main__":
    unittest.main()
