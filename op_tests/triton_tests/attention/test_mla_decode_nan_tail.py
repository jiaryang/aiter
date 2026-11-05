# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A NaN in the unused tail of the last KV block must not reach decode output.

The score mask makes P = 0 on those rows, but wmma(P, V) still multiplies them.
0 * NaN is NaN, so the V load has to substitute 0 past seq_len.
"""

import pytest
import torch

import aiter.ops.triton.attention.mla as mla_module
from aiter.ops.triton.attention.mla import mla_decode_fwd
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx1250", reason="gfx1250 Gluon MLA decode"
)

LORA, ROPE, HEADS = 512, 64, 8


def _run(block_size, kv_dtype, seq_lens, max_seqlen_kv=0):
    dev = "cuda"
    bs = len(seq_lens)
    blocks_per_seq = [-(-s // block_size) for s in seq_lens]
    num_blocks = sum(blocks_per_seq)
    block_tables = torch.zeros((bs, max(blocks_per_seq)), dtype=torch.int32, device=dev)
    kv = torch.full((num_blocks, block_size, 1, LORA + ROPE), float("nan"), device=dev)
    q = torch.randn((bs, HEADS, LORA + ROPE), dtype=torch.bfloat16, device=dev)
    scale = (LORA + ROPE) ** -0.5
    used = 0
    refs = []
    for i, (seq_len, n_blocks) in enumerate(zip(seq_lens, blocks_per_seq)):
        ids = torch.arange(used, used + n_blocks, device=dev)
        used += n_blocks
        block_tables[i, :n_blocks] = ids.to(torch.int32)
        tokens = torch.randn((seq_len, LORA + ROPE), device=dev)
        flat = kv[ids].view(n_blocks * block_size, LORA + ROPE)
        flat[:seq_len] = tokens
        kv[ids] = flat.view(n_blocks, block_size, 1, LORA + ROPE)
        k = tokens.to(kv_dtype).float()
        probs = torch.softmax((q[i].float() @ k.T) * scale, dim=-1)
        refs.append(probs @ k[:, :LORA])
    kv = kv.to(kv_dtype)
    out = torch.empty((bs, HEADS, LORA), dtype=torch.bfloat16, device=dev)
    mla_decode_fwd(
        q=q,
        kv_buffer=kv,
        out=out,
        cu_seqlens_q=torch.arange(bs + 1, dtype=torch.int32, device=dev),
        seqused_k=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        max_seqlen_kv=max_seqlen_kv,
        block_tables=block_tables,
        softmax_scale=scale,
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
        causal=True,
        q_descale=None,
        kv_descale=(
            torch.ones((), dtype=torch.float32, device=dev)
            if kv_dtype != torch.bfloat16
            else None
        ),
    )
    torch.cuda.synchronize()
    assert not torch.isnan(out).any()
    for i, ref in enumerate(refs):
        denom = ref.abs().max().clamp_min(1e-6)
        rel = (out[i].float() - ref).abs().max() / denom
        assert rel < 0.05, f"seq {seq_lens[i]} rel {rel}"


@pytest.mark.parametrize(
    "block_size,kv_dtype",
    [(64, e4m3_dtype), (64, torch.bfloat16), (128, e4m3_dtype)],
)
def test_partial_block_nan_tail_stays_out_of_the_output(block_size, kv_dtype):
    # Page sizes 64 and 128 take the pipelined kernel. 40 ends mid-block; 64
    # fills a page-64 block and ends mid-block on page 128.
    _run(block_size, kv_dtype, [40, 64])


@pytest.mark.parametrize(
    "kv_dtype,num_stages",
    [(e4m3_dtype, 3), (e4m3_dtype, 4), (torch.bfloat16, 3)],
)
def test_kv_ring_nan_tail_stays_out_of_the_output(monkeypatch, kv_dtype, num_stages):
    # num_stages > 2 (the KV ring) only comes from a tuned override, which is
    # read only when max_seqlen_kv is known, and dropped if it overflows LDS.
    kv_elem_size = torch.empty((), dtype=kv_dtype).element_size()
    assert (
        mla_module._pipelined_unshuffled_lds_bytes(
            64, 16, LORA, ROPE, kv_elem_size, 2, num_stages
        )
        <= 256 * 1024
    )
    monkeypatch.setattr(
        mla_module,
        "_tuned_decode_config",
        lambda *args, **kwargs: {"num_stages": num_stages},
    )
    # One tile (the prime re-reads it), and several tiles ending mid-block.
    seq_lens = [40, 64 * 5 + 7, 64 * 9 + 63]
    _run(64, kv_dtype, seq_lens, max_seqlen_kv=max(seq_lens))
