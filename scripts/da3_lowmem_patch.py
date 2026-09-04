"""Opt-in low-GPU-memory patch for Depth Anything 3 inference.

Enabled by ``DA3_LOWMEM_OFFLOAD=1`` in ``run_da3_numeric_inference.py``.

DA3 keeps every retained backbone layer output (4 layers x 2048 x tokens,
fp32) on the GPU for all views at once and then runs the DPT head over them.
On an 11 GB RTX 2080 Ti that caps a 504-res scene at roughly 60-70 views.
This patch keeps the network math unchanged and only changes *where* the
retained tensors live:

1. ``DinoVisionTransformer.get_intermediate_layers`` — same block loop, same
   reference-view reordering, same LayerNorm; each retained layer output is
   normed on the GPU exactly as before and then copied to pinned host memory.
   Camera tokens are cloned so the pre-norm activations can be freed instead
   of being kept alive by a view.
2. ``DualDPT.forward`` — the same ``chunk_size=8`` chunk loop as upstream, but
   each chunk is copied back to the GPU right before ``_forward_impl``.

Everything else (attention, MLPs, head convolutions, camera decoder, output
processing, GLB export) runs untouched.
"""
from __future__ import annotations

import torch


def apply() -> None:
    from depth_anything_3.model import dualdpt
    from depth_anything_3.model.dinov2 import vision_transformer as vt

    VT = vt.DinoVisionTransformer

    def _to_pinned_cpu(tensor: torch.Tensor) -> torch.Tensor:
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
        host.copy_(tensor, non_blocking=False)
        return host

    def _norm_like_upstream(self, out_x: torch.Tensor) -> torch.Tensor:
        # Mirrors DinoVisionTransformer.get_intermediate_layers.
        if out_x.shape[-1] == self.embed_dim:
            normed = self.norm(out_x)
        elif out_x.shape[-1] == (self.embed_dim * 2):
            normed = torch.cat(
                [out_x[..., : self.embed_dim], self.norm(out_x[..., self.embed_dim :])],
                dim=-1,
            )
        else:
            raise ValueError(f"Invalid output shape: {out_x.shape}")
        return normed[..., 1 + self.num_register_tokens :, :]

    def get_intermediate_layers(self, x, n=1, export_feat_layers=[], **kwargs):
        # Same control flow as DinoVisionTransformer._get_intermediate_layers_not_chunked
        # followed by get_intermediate_layers, with retained outputs offloaded.
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        total_block_len = len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        outputs: list[torch.Tensor] = []
        camera_tokens: list[torch.Tensor] = []
        aux_output: list[torch.Tensor] = []
        local_x = x
        b_idx = None

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            if (
                self.alt_start != -1
                and (i == self.alt_start - 1)
                and x.shape[1] >= vt.THRESH_FOR_REF_SELECTION
                and kwargs.get("cam_token", None) is None
            ):
                strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
                vt.logger.info(f"Selecting reference view using strategy: {strategy}")
                b_idx = vt.select_reference_view(x, strategy=strategy)
                x = vt.reorder_by_reference(x, b_idx)
                local_x = vt.reorder_by_reference(local_x, b_idx)

            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    vt.logger.info("Using camera conditions provided by the user")
                    cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                if (
                    x.shape[1] >= vt.THRESH_FOR_REF_SELECTION
                    and self.alt_start != -1
                    and b_idx is not None
                ):
                    out_x = vt.restore_original_order(out_x, b_idx)
                camera_tokens.append(out_x[:, :, 0].clone())
                outputs.append(_to_pinned_cpu(_norm_like_upstream(self, out_x)))
                del out_x
            if i in export_feat_layers:
                aux_output.append(x)

        aux_output = [self.norm(out) for out in aux_output]
        aux_output = [out[..., 1 + self.num_register_tokens :, :] for out in aux_output]
        return tuple(zip(outputs, camera_tokens)), aux_output

    def dualdpt_forward(self, feats, H, W, patch_start_idx, chunk_size: int = 8):
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        device = next(self.parameters()).device
        if chunk_size is None or chunk_size >= S:
            out_dict = self._forward_impl(
                [feat.to(device, non_blocking=True) for feat in feats], H, W, patch_start_idx
            )
            out_dict = {k: v.reshape(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return dualdpt.Dict(out_dict)
        out_dicts = []
        for s0 in range(0, B * S, chunk_size):
            s1 = min(s0 + chunk_size, B * S)
            out_dict = self._forward_impl(
                [feat[s0:s1].to(device, non_blocking=True) for feat in feats],
                H,
                W,
                patch_start_idx,
            )
            out_dicts.append(out_dict)
        out_dict = {
            k: torch.cat([out_dict[k] for out_dict in out_dicts], dim=0)
            for k in out_dicts[0].keys()
        }
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return dualdpt.Dict(out_dict)

    VT.get_intermediate_layers = get_intermediate_layers
    dualdpt.DualDPT.forward = dualdpt_forward
    print("[da3] low-memory patch applied: backbone features offloaded to pinned CPU, DPT head chunks re-uploaded", flush=True)
