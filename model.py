# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import _C


# ======================================================================================
# Utils
# ======================================================================================

def norm2_distance(fm_ref: torch.Tensor, fm_tar: torch.Tensor) -> torch.Tensor:
    """Compute per-edge L2 distance (squared) across channel dim (dim=1)."""
    diff = fm_ref - fm_tar
    return (diff * diff).sum(dim=1)


def batch_index_opr(data: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Gather along last dim (dim=2) for 3D tensor data: [B, C, L], index: [B, L].
    Returns: [B, C, L]
    """
    with torch.no_grad():
        c = data.shape[1]
        index = index.unsqueeze(1).expand(-1, c, -1).long()
    return torch.gather(data, dim=2, index=index)


def build_act_layer(act_layer: str) -> nn.Module:
    act_layer = str(act_layer)
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    if act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    if act_layer == "GELU":
        return nn.GELU()
    raise NotImplementedError(f"build_act_layer does not support {act_layer}")


class ToChannelsFirst(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) -> (B, C, H, W)
        return x.permute(0, 3, 1, 2)


class ToChannelsLast(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H, W, C)
        return x.permute(0, 2, 3, 1)


def build_norm_layer(
    dim: int,
    norm_layer: str,
    in_format: str = "channels_last",
    out_format: str = "channels_last",
    eps: float = 1e-6,
) -> nn.Sequential:
    """
    Create normalization wrapper that handles (channels_first/channels_last) conversion.
    - BN: expects channels_first (B,C,H,W)
    - LN: expects channels_last (B,H,W,C) when applied spatially
    """
    layers: list[nn.Module] = []
    norm_layer = str(norm_layer)

    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(ToChannelsFirst())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == "channels_last":
            layers.append(ToChannelsLast())
        return nn.Sequential(*layers)

    if norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(ToChannelsLast())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first":
            layers.append(ToChannelsFirst())
        return nn.Sequential(*layers)

    raise NotImplementedError(f"build_norm_layer does not support {norm_layer}")


def _same_padding_1d(kernel_size: int, dilation: int) -> int:
    """
    "Same length" padding for odd kernel_size Conv1d.
    For odd k: pad = dilation * (k-1)/2.
    """
    k = int(kernel_size)
    d = int(dilation)
    if k % 2 == 0:
        raise ValueError(f"kernel_size must be odd for same padding, got {k}")
    return d * (k - 1) // 2


# ======================================================================================
# Tree Scan Core ops (CUDA extensions)
# ======================================================================================

class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index: torch.Tensor, max_adj_per_vertex: int):
        sorted_index, sorted_parent, sorted_child = _C.bfs_forward(edge_index, max_adj_per_vertex)
        return sorted_index, sorted_parent, sorted_child


class _Refine(Function):
    @staticmethod
    def forward(
        ctx,
        feature_in: torch.Tensor,
        edge_weight: torch.Tensor,
        sorted_index: torch.Tensor,
        sorted_parent: torch.Tensor,
        sorted_child: torch.Tensor,
        edge_coef: torch.Tensor,
    ):
        feature_aggr, feature_aggr_up = _C.tree_scan_refine_forward(
            feature_in, edge_weight, sorted_index, sorted_parent, sorted_child, edge_coef
        )
        ctx.save_for_backward(
            feature_in, edge_weight, sorted_index, sorted_parent, sorted_child,
            feature_aggr, feature_aggr_up, edge_coef
        )
        return feature_aggr

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (
            feature_in, edge_weight, sorted_index, sorted_parent, sorted_child,
            feature_aggr, feature_aggr_up, edge_coef
        ) = ctx.saved_tensors

        grad_feature = _C.tree_scan_refine_backward_feature(
            feature_in, edge_weight, sorted_index, sorted_parent, sorted_child,
            feature_aggr, feature_aggr_up, grad_output, edge_coef
        )
        grad_edge_weight = _C.tree_scan_refine_backward_edge_weight(
            feature_in, edge_weight, sorted_index, sorted_parent, sorted_child,
            feature_aggr, feature_aggr_up, grad_output, edge_coef
        )
        return grad_feature, grad_edge_weight, None, None, None, None


class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index: torch.Tensor, edge_weight: torch.Tensor, vertex_index: int):
        return _C.mst_forward(edge_index, edge_weight, vertex_index)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        return None, None, None


mst = _MST.apply


# ======================================================================================
# MST builder (spatial 4-neighborhood)
# ======================================================================================

class MinimumSpanningTree(nn.Module):
    def __init__(self, distance_func: str, mapping_func=None):
        super().__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func

    @staticmethod
    def _build_matrix_index(fm: torch.Tensor) -> torch.Tensor:
        batch, height, width = fm.shape[0], fm.shape[2], fm.shape[3]
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width  # [H, W]

        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], dim=2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], dim=2)
        index = torch.cat([row_index.reshape(1, -1, 2), col_index.reshape(1, -1, 2)], dim=1)
        return index.expand(batch, -1, -1)

    def _build_feature_weight(self, fm: torch.Tensor) -> torch.Tensor:
        batch = fm.shape[0]
        weight_row = norm2_distance(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight = torch.cat([weight_row.reshape(batch, -1), weight_col.reshape(batch, -1)], dim=1)
        return self.mapping_func(weight) if self.mapping_func is not None else weight

    def _build_feature_weight_cosine(self, fm: torch.Tensor, max_tree: bool) -> torch.Tensor:
        batch, dim = fm.shape[0], fm.shape[1]
        w_row = torch.cosine_similarity(
            fm[:, :, :-1, :].reshape(batch, dim, -1),
            fm[:, :, 1:, :].reshape(batch, dim, -1),
            dim=1,
        )
        w_col = torch.cosine_similarity(
            fm[:, :, :, :-1].reshape(batch, dim, -1),
            fm[:, :, :, 1:].reshape(batch, dim, -1),
            dim=1,
        )
        weight = torch.cat([w_row, w_col], dim=1)

        if self.mapping_func is None:
            return weight

        return self.mapping_func(weight if max_tree else -weight)

    @torch.no_grad()
    def forward(self, guide_in: torch.Tensor, max_tree: bool = False) -> torch.Tensor:
        index = self._build_matrix_index(guide_in)
        if self.distance_func == "Cosine":
            weight = self._build_feature_weight_cosine(guide_in, max_tree=max_tree)
        else:
            weight = self._build_feature_weight(guide_in)
        return mst(index, weight, guide_in.shape[2] * guide_in.shape[3])


# ======================================================================================
# Spectral Tree Branch (Stage1 upgraded)
# ======================================================================================

class SpectralTreeBranch(nn.Module):
    """
    Build an r-neighbor graph on the CHANNEL axis (treated as vertices),
    edge weights from cosine similarity of each channel's spatial vector,
    then run MST + BFS + tree_scan_refine to propagate features along the spectral tree.

    Input / Output: channels_last features [B, H, W, C]
      - vertices: C (channels)
      - feature channels in refine: L = H*W (spatial positions)

    Causal definition (per your setup):
      - spe_last_root_only=True => ONLY keep the "root" channel (last channel) as representation,
        then broadcast to all C to keep shapes unchanged.
      - spe_last_root_only=False => original non-causal (all-vertex outputs kept).
    """
    def __init__(
        self,
        r: int = 3,
        weight_mapping: str = "exp_neg_cos",   # for MST edge weight (min tree)
        refine_weight_mode: str = "sigmoid_cos",  # how to map cos->refine weights
        refine_temp: float = 2.0,
        refine_eps: float = 1e-6,
    ):
        super().__init__()
        assert r >= 1
        self.r = int(r)
        self.weight_mapping = str(weight_mapping)
        self.refine_weight_mode = str(refine_weight_mode)
        self.refine_temp = float(refine_temp)
        self.refine_eps = float(refine_eps)

    @staticmethod
    def _build_spectral_edges(num_channels: int, r: int, device: torch.device) -> torch.Tensor:
        src_list = []
        dst_list = []
        for d in range(1, r + 1):
            if num_channels - d <= 0:
                continue
            src = torch.arange(0, num_channels - d, device=device, dtype=torch.int32)
            dst = src + d
            src_list.append(src)
            dst_list.append(dst)

        if len(src_list) == 0:
            src = torch.zeros(1, device=device, dtype=torch.int32)
            dst = torch.zeros(1, device=device, dtype=torch.int32)
            edge = torch.stack([src, dst], dim=1).unsqueeze(0)  # [1,1,2]
            return edge

        src_all = torch.cat(src_list, dim=0)
        dst_all = torch.cat(dst_list, dim=0)
        edge = torch.stack([src_all, dst_all], dim=1).unsqueeze(0)  # [1,E,2]
        return edge

    def _edge_weights_from_feat(self, feat_bcl: torch.Tensor, edge_index_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feat_bcl: [B, C, L] (float)
        edge_index_1: [1, E, 2] int32
        Return:
          - w_mst: [B, E] float (for MST; smaller => more similar)
          - cos  : [B, E] float (raw cosine in [-1,1])
        """
        B, C, L = feat_bcl.shape
        v = F.normalize(feat_bcl, p=2, dim=2, eps=1e-6)  # [B,C,L]

        src = edge_index_1[0, :, 0].long()  # [E]
        dst = edge_index_1[0, :, 1].long()  # [E]
        v_src = v[:, src, :]  # [B,E,L]
        v_dst = v[:, dst, :]  # [B,E,L]
        cos = (v_src * v_dst).sum(dim=2)  # [B,E] in [-1,1]

        if self.weight_mapping == "exp_neg_cos":
            w_mst = torch.exp(-cos)                  # cos high => small weight
        elif self.weight_mapping == "neg_cos":
            w_mst = -cos
        elif self.weight_mapping == "one_minus_cos":
            w_mst = 1.0 - cos
        else:
            raise NotImplementedError(f"Unknown weight_mapping: {self.weight_mapping}")

        return w_mst, cos

    def _tree_edge_cosine(self, v_norm_bcl: torch.Tensor, tree_edges: torch.Tensor) -> torch.Tensor:
        """
        v_norm_bcl: [B,C,L] normalized
        tree_edges: [B,C-1,2] int32/long
        Return cos_tree: [B,C-1] in [-1,1]
        """
        u = tree_edges[:, :, 0].long()
        v = tree_edges[:, :, 1].long()
        vu = torch.gather(v_norm_bcl, 1, u.unsqueeze(-1).expand(-1, -1, v_norm_bcl.shape[2]))  # [B,C-1,L]
        vv = torch.gather(v_norm_bcl, 1, v.unsqueeze(-1).expand(-1, -1, v_norm_bcl.shape[2]))  # [B,C-1,L]
        return (vu * vv).sum(dim=2)  # [B,C-1]

    def _cos_to_refine_weight(self, cos: torch.Tensor) -> torch.Tensor:
        """
        cos: [B, E] in [-1,1]
        Return w_ref: [B, E] positive, larger => stronger propagation
        """
        if self.refine_weight_mode == "sigmoid_cos":
            return torch.sigmoid(self.refine_temp * cos)  # (0,1)
        if self.refine_weight_mode == "exp_cos":
            return torch.exp(self.refine_temp * cos).clamp(min=self.refine_eps)
        if self.refine_weight_mode == "one_plus_cos":
            return (1.0 + cos).clamp(min=self.refine_eps)
        raise NotImplementedError(f"Unknown refine_weight_mode: {self.refine_weight_mode}")

    @staticmethod
    def _accumulate_vertex_weights(C: int, tree_edges: torch.Tensor, edge_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Build vertex-wise weight wv from tree edge weights.
        tree_edges: [B,C-1,2]
        edge_w    : [B,C-1]
        Return wv : [B,C]
        """
        B = tree_edges.shape[0]
        u = tree_edges[:, :, 0].long()
        v = tree_edges[:, :, 1].long()

        wv_sum = torch.zeros((B, C), device=tree_edges.device, dtype=edge_w.dtype)
        deg = torch.zeros((B, C), device=tree_edges.device, dtype=edge_w.dtype)

        wv_sum.scatter_add_(1, u, edge_w)
        wv_sum.scatter_add_(1, v, edge_w)
        deg.scatter_add_(1, u, torch.ones_like(edge_w))
        deg.scatter_add_(1, v, torch.ones_like(edge_w))

        wv = wv_sum / (deg + eps)
        return wv

    def forward(self, x_hwc: torch.Tensor, *, spe_last_root_only: bool = False) -> torch.Tensor:
        """
        x_hwc: [B, H, W, C]
        return: [B, H, W, C] (spectral-tree refined)
        """
        B, H, W, C = x_hwc.shape
        device = x_hwc.device
        L = H * W

        feat_bcl = x_hwc.permute(0, 3, 1, 2).reshape(B, C, L).contiguous()  # [B,C,L]
        edge_index_1 = self._build_spectral_edges(C, self.r, device=device)  # [1,E,2]
        edge_index = edge_index_1.expand(B, -1, -1).contiguous()             # [B,E,2]

        w_mst, _ = self._edge_weights_from_feat(feat_bcl, edge_index_1)      # [B,E]
        tree = mst(edge_index, w_mst, C)                                     # [B,C-1,2]

        bfs = _BFS.apply
        refine = _Refine.apply
        max_adj = max(2 * self.r, 2)
        sorted_index, sorted_parent, sorted_child = bfs(tree, max_adj)

        v_norm = F.normalize(feat_bcl, p=2, dim=2, eps=1e-6)                 # [B,C,L]
        cos_tree = self._tree_edge_cosine(v_norm, tree)                      # [B,C-1]
        edge_ref_w = self._cos_to_refine_weight(cos_tree)                    # [B,C-1]
        wv = self._accumulate_vertex_weights(C, tree, edge_ref_w, eps=self.refine_eps)  # [B,C]
        wv = wv.clamp(min=self.refine_eps)

        feature_in = feat_bcl.transpose(1, 2).contiguous()                   # [B,L,C]

        ew = wv.unsqueeze(1).expand(B, L, C).contiguous()                    # [B,L,C]
        ew = batch_index_opr(ew, sorted_index)

        edge_coef = torch.ones_like(sorted_index, dtype=feature_in.dtype)
        feature_out = refine(feature_in, ew, sorted_index, sorted_parent, sorted_child, edge_coef)  # [B,L,C]

        out_bcl = feature_out.transpose(1, 2).contiguous()                   # [B,C,L]

        # spectral causal: keep last channel as root, broadcast to all C
        if spe_last_root_only:
            ridx = C - 1
            root = out_bcl[:, ridx:ridx + 1, :]                              # [B,1,L]
            out_bcl = root.expand(-1, C, -1).contiguous()                    # [B,C,L]

        out_hwc = out_bcl.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        return out_hwc


# ======================================================================================
# Spectral State Fusion (SSF)
# ======================================================================================

class SpectralStateFusion(nn.Module):
    """
    Spectral State Fusion.
    Input/Output: [B, L, C], where C = K * n_state.
    A multi-layer dilated Conv1d stack mixes information across spectral channels.
    """
    def __init__(
        self,
        n_state: int = 128,
        # --- conv stack knobs ---
        kernel_size: int = 3,
        conv_layers: int = 3,
        conv_dilations: Optional[Sequence[int]] = None,
        conv_dilation_base: int = 2,
        conv_act: str = "SiLU",
        conv_dropout: float = 0.0,
        bias: bool = True,
        zero_init_last: bool = False,
        # --- normalization / MLP / gating ---
        use_group_ln: bool = True,
        ln_eps: float = 1e-6,
        use_group_mlp: bool = True,
        mlp_ratio: float = 0.5,
        mlp_dropout: float = 0.0,
        use_group_mix: bool = True,
        group_mix_kernel: int = 3,
        min_groups_to_apply: int = 2,
        use_residual_gate: bool = True,
        gate_mode: str = "channel",
        gate_init: float = -2.0,
        gate_clamp: float = 6.0,
        skip_if_K_gt: Optional[int] = None,
        # --- unused, kept for signature compatibility ---
        dilation: int = 2,
        padding: int = 2,
    ):
        super().__init__()
        self.n_state = int(n_state)

        # -------------------------
        # LN per group
        # -------------------------
        self.use_group_ln = bool(use_group_ln)
        self.group_ln = nn.LayerNorm(self.n_state, eps=float(ln_eps)) if self.use_group_ln else None

        # -------------------------
        # Multi-layer dilated conv stack (default 3 layers)
        # -------------------------
        k = int(kernel_size)
        if k % 2 == 0:
            raise ValueError(f"SpectralStateFusion expects odd kernel_size, got {k}")

        Lc = int(conv_layers)
        if Lc < 1:
            raise ValueError(f"conv_layers must be >= 1, got {Lc}")

        if conv_dilations is not None:
            dil_list = [int(d) for d in conv_dilations]
            if len(dil_list) != Lc:
                raise ValueError(f"len(conv_dilations) must equal conv_layers ({Lc}), got {len(dil_list)}")
        else:
            # auto: [1, base, base^2, ...] (or [dilation] if Lc==1 keeps old-ish default)
            base = int(conv_dilation_base)
            if base < 1:
                raise ValueError(f"conv_dilation_base must be >= 1, got {base}")
            if Lc == 1:
                dil_list = [int(dilation)]
            else:
                dil_list = [1]
                for i in range(1, Lc):
                    dil_list.append(dil_list[-1] * base)

        self.conv_layers = Lc
        self.conv_dilations = tuple(dil_list)
        self.conv_act_name = str(conv_act)
        self.conv_dropout_p = float(conv_dropout)

        conv_blocks: list[nn.Module] = []
        act = build_act_layer(self.conv_act_name)
        for i, dil in enumerate(dil_list):
            pad_i = _same_padding_1d(k, dil)
            conv_i = nn.Conv1d(
                self.n_state,
                self.n_state,
                kernel_size=k,
                dilation=int(dil),
                padding=int(pad_i),
                bias=bool(bias),
            )
            conv_blocks.append(conv_i)

            # activation/dropout between convs (not after last conv by default)
            if i != len(dil_list) - 1:
                conv_blocks.append(act)
                if self.conv_dropout_p > 0:
                    conv_blocks.append(nn.Dropout(p=self.conv_dropout_p))

        self.convL = nn.Sequential(*conv_blocks)

        if bool(zero_init_last):
            # optional: make the stack start near-identity (i.e., conv path initially ~0)
            # We only zero-init the LAST conv layer.
            last_conv = None
            for m in reversed(self.convL):
                if isinstance(m, nn.Conv1d):
                    last_conv = m
                    break
            if last_conv is not None:
                nn.init.zeros_(last_conv.weight)
                if last_conv.bias is not None:
                    nn.init.zeros_(last_conv.bias)

        # -------------------------
        # Group MLP (same as yours)
        # -------------------------
        self.use_group_mlp = bool(use_group_mlp)
        if self.use_group_mlp:
            hidden = max(16, int(self.n_state * float(mlp_ratio)))
            self.mlp = nn.Sequential(
                nn.Linear(self.n_state, hidden, bias=True),
                nn.SiLU(inplace=True),
                nn.Dropout(p=float(mlp_dropout)),
                nn.Linear(hidden, self.n_state, bias=True),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            self.mlp = None

        # -------------------------
        # Cross-group mixing along K (same as yours)
        # -------------------------
        self.use_group_mix = bool(use_group_mix)
        self.min_groups_to_apply = int(min_groups_to_apply)
        self.skip_if_K_gt = skip_if_K_gt if skip_if_K_gt is None else int(skip_if_K_gt)

        k2 = int(group_mix_kernel)
        pad2 = k2 // 2
        self.group_mix_dw = nn.Conv1d(
            in_channels=self.n_state,
            out_channels=self.n_state,
            kernel_size=k2,
            padding=pad2,
            groups=self.n_state,
            bias=False,
        )
        self.group_mix_pw = nn.Conv1d(
            in_channels=self.n_state,
            out_channels=self.n_state,
            kernel_size=1,
            padding=0,
            groups=1,
            bias=True,
        )
        nn.init.zeros_(self.group_mix_dw.weight)
        nn.init.zeros_(self.group_mix_pw.weight)
        nn.init.zeros_(self.group_mix_pw.bias)

        # -------------------------
        # Residual gate (same as yours)
        # -------------------------
        self.use_residual_gate = bool(use_residual_gate)
        self.gate_mode = str(gate_mode).lower()
        self.gate_clamp = float(gate_clamp)

        if self.use_residual_gate:
            if self.gate_mode == "scalar":
                self.g = nn.Parameter(torch.tensor(float(gate_init)))
            elif self.gate_mode == "channel":
                self.g = nn.Parameter(torch.full((self.n_state,), float(gate_init)))
            else:
                raise ValueError(f"Unsupported gate_mode: {gate_mode}")

    def _gate(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.use_residual_gate:
            raise RuntimeError("_gate called but use_residual_gate=False")
        g = torch.clamp(self.g, -self.gate_clamp, self.gate_clamp)
        return torch.sigmoid(g).to(dtype=dtype, device=device)

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        if out.dim() != 3:
            raise ValueError(f"SpectralStateFusion expects [B,L,C], got {tuple(out.shape)}")

        B, L, C = out.shape
        if C % self.n_state != 0:
            raise ValueError(f"C={C} must be divisible by n_state={self.n_state}")
        K = C // self.n_state

        if self.skip_if_K_gt is not None and K > self.skip_if_K_gt:
            return out

        x0 = out
        x = out.view(B, L, K, self.n_state)

        if self.group_ln is not None:
            x = self.group_ln(x)

        # conv along spectral-state dim (n_state) over sequence length L (per group K)
        x_l = x.permute(0, 2, 3, 1).contiguous().view(B * K, self.n_state, L)
        x_l = self.convL(x_l)  # multi-layer dilated conv stack
        x = x_l.view(B, K, self.n_state, L).permute(0, 3, 1, 2).contiguous()

        if self.mlp is not None:
            x = x + self.mlp(x)

        if self.use_group_mix and K >= self.min_groups_to_apply:
            xg = x.permute(0, 1, 3, 2).contiguous().view(B * L, self.n_state, K)
            xg = self.group_mix_dw(xg)
            xg = self.group_mix_pw(xg)
            x = xg.view(B, L, self.n_state, K).permute(0, 1, 3, 2).contiguous()

        y = x.view(B, L, C)

        if self.use_residual_gate:
            gate = self._gate(dtype=y.dtype, device=y.device)
            if self.gate_mode == "scalar":
                return x0 + gate * (y - x0)
            g = gate.view(1, 1, 1, self.n_state).expand(B, L, K, self.n_state).reshape(B, L, C)
            return x0 + g * (y - x0)

        return y


# ======================================================================================
# Two-Branch Fusion (Stage1 upgraded)
# ======================================================================================

class Fusion2Branch(nn.Module):
    """
    Fuse two feature maps (channels_last): h_spa, h_spe in [B,H,W,C]
    Methods:
      - "add"       : y = h_spa + h_spe
      - "gate"      : y = h_spa + sigmoid(MLP([h_spa,h_spe])) * h_spe
      - "concat"    : y = Linear([h_spa,h_spe]) -> C
      - "crossgate" : y = (1 + beta*g_spe) * h_spa + alpha*g_spa * h_spe
    Output: [B,H,W,C]
    """
    def __init__(
        self,
        dim: int,
        method: str = "crossgate",
        gate_hidden_ratio: float = 0.25,
        gate_init_bias: float = -2.0,
        alpha_init: float = -4.0,
        beta_init: float = -4.0,
        clamp_scale: float = 6.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.method = str(method).lower()
        self.clamp_scale = float(clamp_scale)

        if self.method == "add":
            self.proj = None
            self.gate = None
            self.gate_spa = None
            self.gate_spe = None
            self.alpha = None
            self.beta = None

        elif self.method == "concat":
            self.proj = nn.Linear(2 * self.dim, self.dim, bias=True)
            self.gate = None
            self.gate_spa = None
            self.gate_spe = None
            self.alpha = None
            self.beta = None

        elif self.method == "gate":
            hidden = max(1, int(self.dim * gate_hidden_ratio))
            self.gate = nn.Sequential(
                nn.Linear(2 * self.dim, hidden, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, self.dim, bias=True),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, float(gate_init_bias))
            self.proj = None
            self.gate_spa = None
            self.gate_spe = None
            self.alpha = None
            self.beta = None

        elif self.method == "crossgate":
            hidden = max(1, int(self.dim * gate_hidden_ratio))

            self.gate_spa = nn.Sequential(
                nn.Linear(2 * self.dim, hidden, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, self.dim, bias=True),
            )
            nn.init.zeros_(self.gate_spa[-1].weight)
            nn.init.constant_(self.gate_spa[-1].bias, float(gate_init_bias))

            self.gate_spe = nn.Sequential(
                nn.Linear(2 * self.dim, hidden, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, self.dim, bias=True),
            )
            nn.init.zeros_(self.gate_spe[-1].weight)
            nn.init.constant_(self.gate_spe[-1].bias, float(gate_init_bias))

            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))

            self.proj = None
            self.gate = None

        else:
            raise NotImplementedError(f"Fusion2Branch does not support method='{self.method}'")

    def forward(self, h_spa: torch.Tensor, h_spe: torch.Tensor) -> torch.Tensor:
        if h_spa.shape != h_spe.shape:
            raise ValueError(f"Shape mismatch: h_spa{tuple(h_spa.shape)} vs h_spe{tuple(h_spe.shape)}")
        if h_spa.dim() != 4:
            raise ValueError(f"Expect [B,H,W,C], got {tuple(h_spa.shape)}")

        if self.method == "add":
            return h_spa + h_spe

        cat = torch.cat([h_spa, h_spe], dim=-1)  # [B,H,W,2C]

        if self.method == "concat":
            return self.proj(cat)

        if self.method == "gate":
            g = torch.sigmoid(self.gate(cat))  # [B,H,W,C]
            return h_spa + g * h_spe

        g_spa = torch.sigmoid(self.gate_spa(cat))  # [B,H,W,C]
        g_spe = torch.sigmoid(self.gate_spe(torch.cat([h_spe, h_spa], dim=-1)))  # [B,H,W,C]

        a = torch.sigmoid(torch.clamp(self.alpha, -self.clamp_scale, self.clamp_scale))
        b = torch.sigmoid(torch.clamp(self.beta, -self.clamp_scale, self.clamp_scale))

        return (1.0 + b * g_spe) * h_spa + (a * g_spa) * h_spe


# ======================================================================================
# Tree scanning functions
# ======================================================================================

def _center_index_hw(H: int, W: int) -> int:
    # patch center (integer center). For odd H/W this is the true center.
    return (H // 2) * W + (W // 2)


def tree_scanning_core(
    xs: torch.Tensor,         # [B, D, L]
    dts: torch.Tensor,        # [B, D, L]
    As: torch.Tensor,
    Bs: torch.Tensor,
    Cs: torch.Tensor,
    Ds: torch.Tensor,
    delta_bias: torch.Tensor,
    origin_shape: Tuple[int, int, int, int],
    h_norm: Optional[nn.Module],
    spectral_fusion: Optional[nn.Module],
    *,
    # spatial "causal" switch: True => keep only center-pixel as root repr (broadcast back to L)
    spa_center_root_only: bool = False,
) -> torch.Tensor:
    n_state = 128
    _, _, H, W = origin_shape
    B, D, L = xs.shape

    assert D % n_state == 0, f"D ({D}) must be divisible by n_state ({n_state})"
    K = D // n_state

    dts = F.softplus(dts + delta_bias.unsqueeze(0).unsqueeze(-1))

    deltaA = (dts * As.unsqueeze(0)).exp_()  # [B, D, L]
    deltaB = rearrange(dts, "b (k d) l -> b k d l", k=K, d=D // K) * Bs
    BX = deltaB * rearrange(xs, "b (k d) l -> b k d l", k=K, d=D // K)

    feat_in = BX.view(B, -1, L)   # [B, D, L]
    edge_weight = deltaA          # [B, D, L]

    fea4tree_hw = rearrange(xs, "b d (h w) -> b d h w", h=H, w=W)
    mst_layer = MinimumSpanningTree("Cosine", torch.exp)
    tree = mst_layer(fea4tree_hw)

    bfs = _BFS.apply
    refine = _Refine.apply
    sorted_index, sorted_parent, sorted_child = bfs(tree, 4)

    edge_weight = batch_index_opr(edge_weight, sorted_index)
    edge_weight_coef = torch.ones_like(sorted_index, dtype=edge_weight.dtype)

    feature_out = refine(feat_in, edge_weight, sorted_index, sorted_parent, sorted_child, edge_weight_coef)

    if h_norm is not None:
        out = h_norm(feature_out.transpose(-1, -2).contiguous())   # [B, L, D]
    else:
        out = feature_out.transpose(-1, -2).contiguous()           # [B, L, D]

    if spectral_fusion is not None:
        out = spectral_fusion(out)

    # spatial causal: keep center pixel as root, broadcast back
    if spa_center_root_only:
        cidx = _center_index_hw(H, W)
        out_center = out[:, cidx:cidx + 1, :]              # [B,1,D]
        out = out_center.expand(-1, L, -1).contiguous()    # [B,L,D]

    y = (
        rearrange(out, "b l (k d) -> b l k d", k=K, d=D // K).unsqueeze(-1)
        @ rearrange(Cs, "b k n l -> b l k n").unsqueeze(-1)
    ).squeeze(-1)

    y = rearrange(y, "b l k d -> b (k d) l")
    y = y + Ds.reshape(1, -1, 1) * xs
    return y


def tree_scanning(
    x: torch.Tensor,
    x_proj_weight: torch.Tensor,
    x_proj_bias: Optional[torch.Tensor],
    dt_projs_weight: torch.Tensor,
    dt_projs_bias: torch.Tensor,
    A_logs: torch.Tensor,
    Ds: torch.Tensor,
    out_norm: nn.Module,
    *,
    to_dtype: bool = True,
    force_fp32: bool = False,
    h_norm: Optional[nn.Module] = None,
    spectral_fusion: Optional[nn.Module] = None,
    spa_center_root_only: bool = False,
) -> torch.Tensor:
    B, D_in, H, W = x.shape
    origin_shape = x.shape
    D, N = A_logs.shape
    K, D2, R = dt_projs_weight.shape
    assert D2 == D, "dt_projs_weight shape mismatch with A_logs"
    L = H * W

    xs = rearrange(x.unsqueeze(1), "b k d h w -> b k d (h w)")
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)

    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)

    As = -torch.exp(A_logs.to(torch.float))
    Ds_f = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys = tree_scanning_core(
        xs, dts, As, Bs.contiguous(), Cs.contiguous(), Ds_f,
        delta_bias, origin_shape, h_norm, spectral_fusion,
        spa_center_root_only=spa_center_root_only,
    ).view(B, K, -1, H, W)

    y = rearrange(ys, "b k d h w -> b (k d) (h w)")
    y = y.transpose(1, 2).contiguous()
    y = out_norm(y).view(B, H, W, -1)
    return y.to(x.dtype) if to_dtype else y


# ======================================================================================
# Network blocks
# ======================================================================================

class DownsampleLayer(nn.Module):
    def __init__(self, channels: int, norm_layer: str = "LN"):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer, "channels_first", "channels_last")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.permute(0, 3, 1, 2))
        return self.norm(x)


class StemLayer(nn.Module):
    def __init__(self, in_chans: int = 30, out_chans: int = 64, act_layer: str = "GELU", norm_layer: str = "BN"):
        super().__init__()
        mid = out_chans // 2
        self.conv1 = nn.Conv2d(in_chans, mid, kernel_size=3, stride=1, padding=1)
        self.norm1 = build_norm_layer(mid, norm_layer, "channels_first", "channels_first")
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(mid, out_chans, kernel_size=3, stride=1, padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, "channels_first", "channels_last")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x


class MLPLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: str = "GELU",
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = in_features if out_features is None else out_features
        hidden_features = in_features if hidden_features is None else hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ======================================================================================
# Scan-strategy switch
# ======================================================================================

def _normalize_scan_strategy(strategy: str) -> Tuple[str, bool, bool]:
    """
    Unify 8 strategies into one switch.

    Returns:
      (branch_mode, spa_causal, spe_causal)
      - branch_mode in {"both","spa","spe"}
      - spa_causal: spatial tree "causal" (center root only)
      - spe_causal: spectral tree "causal" (last-channel root only)

    8 strategies:
      1) both_causal                        : both, spa_causal=True , spe_causal=True
      2) both_noncausal                     : both, spa_causal=False, spe_causal=False
      3) spa_causal                         : spa , spa_causal=True , spe_causal=False
      4) spa_noncausal                      : spa , spa_causal=False, spe_causal=False
      5) spe_causal                         : spe , spa_causal=False, spe_causal=True
      6) spe_noncausal                      : spe , spa_causal=False, spe_causal=False
      7) both_spacausal_spenoncausal        : both, spa_causal=True , spe_causal=False
      8) both_spanoncausal_specausal        : both, spa_causal=False, spe_causal=True

    Aliases are accepted (dual/two + acausal/non_causal etc.), and numeric "1".."8".
    """
    s = str(strategy).strip().lower()
    s = s.replace(" ", "").replace("-", "_").replace("+", "_")

    # numeric shortcuts
    if s in ("1", "01"):
        return "both", True, True
    if s in ("2", "02"):
        return "both", False, False
    if s in ("3", "03"):
        return "spa", True, False
    if s in ("4", "04"):
        return "spa", False, False
    if s in ("5", "05"):
        return "spe", False, True
    if s in ("6", "06"):
        return "spe", False, False
    if s in ("7", "07"):
        return "both", True, False
    if s in ("8", "08"):
        return "both", False, True

    # helpers
    def _is_noncausal(tok: str) -> bool:
        return tok in ("noncausal", "non_causal", "acausal", "noncause", "non_cause")

    def _is_causal(tok: str) -> bool:
        return tok in ("causal", "cause")

    # canonical names + robust aliases

    # 1) both_causal
    if s in ("both_causal", "dual_causal", "two_causal", "bothcausal", "dualcausal", "twocausal"):
        return "both", True, True

    # 2) both_noncausal
    if s in ("both_noncausal", "dual_noncausal", "two_noncausal", "bothnoncausal", "dualnoncausal", "twononcausal",
             "both_non_causal", "dual_non_causal", "two_non_causal", "both_acausal", "dual_acausal", "two_acausal"):
        return "both", False, False

    # 3) spa_causal
    if s in ("spa_causal", "spatial_causal", "space_causal", "spa_ca", "spatialcausal", "spacecausal"):
        return "spa", True, False

    # 4) spa_noncausal
    if s in ("spa_noncausal", "spatial_noncausal", "space_noncausal", "spanoncausal", "spatialnoncausal", "spacenoncausal",
             "spa_non_causal", "spatial_non_causal", "space_non_causal", "spa_acausal", "spatial_acausal", "space_acausal"):
        return "spa", False, False

    # 5) spe_causal
    if s in ("spe_causal", "spectral_causal", "spec_causal", "spe_ca", "spectralcausal", "speccausal"):
        return "spe", False, True

    # 6) spe_noncausal
    if s in ("spe_noncausal", "spectral_noncausal", "spec_noncausal", "spenoncausal", "spectralnoncausal", "specnoncausal",
             "spe_non_causal", "spectral_non_causal", "spec_non_causal", "spe_acausal", "spectral_acausal", "spec_acausal"):
        return "spe", False, False

    # 7) both_spacausal_spenoncausal  (your desired best-candidate)
    if s in (
        "both_spacausal_spenoncausal",
        "dual_spacausal_spenoncausal",
        "two_spacausal_spenoncausal",
        "both_spa_causal_spe_noncausal",
        "dual_spa_causal_spe_noncausal",
        "both_spa_causal_spe_non_causal",
        "both_spa_causal_spec_noncausal",
        "both_spa_causal_spectral_noncausal",
    ):
        return "both", True, False

    # 8) both_spanoncausal_specausal  (reverse)
    if s in (
        "both_spanoncausal_specausal",
        "dual_spanoncausal_specausal",
        "two_spanoncausal_specausal",
        "both_spa_noncausal_spe_causal",
        "dual_spa_noncausal_spe_causal",
        "both_spa_non_causal_spe_causal",
        "both_spa_noncausal_spec_causal",
        "both_spa_noncausal_spectral_causal",
    ):
        return "both", False, True

    # extra flexible parsing: "both_spaX_speY"
    # e.g. both_spa_causal_spe_noncausal
    if s.startswith("both_") or s.startswith("dual_") or s.startswith("two_"):
        toks = s.split("_")
        # try find spa status
        spa_c = None
        spe_c = None
        for i in range(len(toks) - 1):
            if toks[i] in ("spa", "spatial", "space"):
                if _is_causal(toks[i + 1]):
                    spa_c = True
                elif _is_noncausal(toks[i + 1]):
                    spa_c = False
            if toks[i] in ("spe", "spec", "spectral"):
                if _is_causal(toks[i + 1]):
                    spe_c = True
                elif _is_noncausal(toks[i + 1]):
                    spe_c = False
        if spa_c is not None and spe_c is not None:
            return "both", bool(spa_c), bool(spe_c)

    raise ValueError(
        f"Unsupported scan_strategy='{strategy}'.\n"
        f"Use one of:\n"
        f"  1) both_causal\n"
        f"  2) both_noncausal\n"
        f"  3) spa_causal\n"
        f"  4) spa_noncausal\n"
        f"  5) spe_causal\n"
        f"  6) spe_noncausal\n"
        f"  7) both_spacausal_spenoncausal\n"
        f"  8) both_spanoncausal_specausal\n"
        f"(or numeric '1'..'8')."
    )


# ======================================================================================
# Main SSM block
# ======================================================================================

class Tree_SSM(nn.Module):
    def __init__(
        self,
        d_model: int = 96,
        d_state: int | str = 16,
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        act_layer: type[nn.Module] = nn.SiLU,
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        # --- spectral tree config ---
        spe_tree_r: int = 3,
        # --- fusion between spa & spe ---
        spe_fusion: str = "crossgate",  # "add" / "concat" / "gate" / "crossgate"
        # --- SSF switch ---
        use_ssf: bool = True,
        ssf_n_state: int = 128,
        # SSF dilated conv settings (default 3 layers)
        ssf_conv_layers: int = 3,
        ssf_kernel_size: int = 3,
        ssf_conv_dilations: Optional[Sequence[int]] = None,
        ssf_conv_dilation_base: int = 2,
        ssf_conv_act: str = "SiLU",
        ssf_conv_dropout: float = 0.0,
        ssf_zero_init_last: bool = False,
        # scan strategy
        scan_strategy: str = "spa_causal",
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": None, "dtype": None}

        branch_mode, spa_causal, spe_causal = _normalize_scan_strategy(scan_strategy)
        self.branch_mode = branch_mode
        self.spa_causal = bool(spa_causal)
        self.spe_causal = bool(spe_causal)

        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else int(d_state)
        self.d_conv = int(d_conv)

        self.out_norm = nn.LayerNorm(d_inner)
        self.h_norm = nn.LayerNorm(d_inner)

        self.spectral_tree = SpectralTreeBranch(
            r=spe_tree_r,
            weight_mapping="exp_neg_cos",
            refine_weight_mode="sigmoid_cos",
            refine_temp=2.0,
        )

        self.spe_fusion = str(spe_fusion).lower()
        self.fuse_2branch = Fusion2Branch(dim=d_inner, method=self.spe_fusion)

        self.use_ssf = bool(use_ssf)
        self.ssf_n_state = int(ssf_n_state)
        self.ssf = SpectralStateFusion(
            n_state=self.ssf_n_state,
            kernel_size=ssf_kernel_size,
            conv_layers=ssf_conv_layers,
            conv_dilations=ssf_conv_dilations,
            conv_dilation_base=ssf_conv_dilation_base,
            conv_act=ssf_conv_act,
            conv_dropout=ssf_conv_dropout,
            zero_init_last=ssf_zero_init_last,
        ) if self.use_ssf else None

        self.K = 1
        self.K2 = self.K

        d_proj = d_expand * 2
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=self.d_conv,
                padding=(self.d_conv - 1) // 2,
                **factory_kwargs,
            )
        else:
            self.conv2d = None

        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        **factory_kwargs,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state: int, d_inner: int, copies: int = -1, device=None, merge: bool = True) -> nn.Parameter:
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner: int, copies: int = -1, device=None, merge: bool = True) -> nn.Parameter:
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(
        self,
        x: torch.Tensor,
        *,
        channel_first: bool = False,
        force_fp32: Optional[bool] = None
    ) -> torch.Tensor:
        force_fp32 = self.training if force_fp32 is None else force_fp32

        if channel_first:
            x_cf = x
            x_hwc = x.permute(0, 2, 3, 1).contiguous()
        else:
            x_hwc = x
            x_cf = x.permute(0, 3, 1, 2).contiguous()

        mode = self.branch_mode

        if mode == "spe":
            y_hwc = self.spectral_tree(x_hwc, spe_last_root_only=self.spe_causal)

        elif mode == "spa":
            y_hwc = tree_scanning(
                x_cf,
                self.x_proj_weight,
                None,
                self.dt_projs_weight,
                self.dt_projs_bias,
                self.A_logs,
                self.Ds,
                out_norm=self.out_norm,
                force_fp32=force_fp32,
                h_norm=self.h_norm,
                spectral_fusion=None,
                spa_center_root_only=self.spa_causal,
            )

        else:
            h_spe = self.spectral_tree(x_hwc, spe_last_root_only=self.spe_causal)
            h_spa = tree_scanning(
                x_cf,
                self.x_proj_weight,
                None,
                self.dt_projs_weight,
                self.dt_projs_bias,
                self.A_logs,
                self.Ds,
                out_norm=self.out_norm,
                force_fp32=force_fp32,
                h_norm=self.h_norm,
                spectral_fusion=None,
                spa_center_root_only=self.spa_causal,
            )
            y_hwc = self.fuse_2branch(h_spa, h_spe)

        if self.ssf is not None:
            B, H, W, C = y_hwc.shape
            y_blc = y_hwc.view(B, H * W, C).contiguous()
            y_blc = self.ssf(y_blc)
            y_hwc = y_blc.view(B, H, W, C).contiguous()

        return y_hwc

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)

        if self.conv2d is not None:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x)
            x = x.permute(0, 2, 3, 1).contiguous()

        x = self.act(x)
        y = self.forward_core(x, channel_first=False)
        y = y * z
        return self.dropout(self.out_proj(y))


class GrootVLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        norm_layer: str = "LN",
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        post_norm: bool = False,
        layer_scale: Optional[float] = None,
        with_cp: bool = False,
        spe_fusion: str = "crossgate",
        use_ssf: bool = True,
        scan_strategy: str = "spa_causal",
    ):
        super().__init__()
        self.with_cp = with_cp
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None

        self.norm1 = build_norm_layer(channels, "LN")
        self.TreeSSM = Tree_SSM(
            d_model=channels,
            d_state=1,
            ssm_ratio=2,
            ssm_rank_ratio=2,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=False,
            dropout=0.0,
            spe_tree_r=5,
            spe_fusion=spe_fusion,
            use_ssf=use_ssf,
            # SSF dilated conv settings (3 layers)
            ssf_conv_layers=2,
            ssf_kernel_size=3,
            ssf_conv_dilations=None,         # e.g. (1,2,4) to explicitly set
            ssf_conv_dilation_base=2,        # auto => [1,2,4]
            ssf_conv_act="SiLU",
            ssf_conv_dropout=0.0,
            ssf_zero_init_last=False,
            scan_strategy=scan_strategy,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(channels, "LN")
        self.mlp = MLPLayer(
            in_features=channels,
            hidden_features=int(channels * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _inner(x_: torch.Tensor) -> torch.Tensor:
            if not self.layer_scale:
                if self.post_norm:
                    x_ = x_ + self.drop_path(self.norm1(self.TreeSSM(x_)))
                    x_ = x_ + self.drop_path(self.norm2(self.mlp(x_)))
                else:
                    x_ = x_ + self.drop_path(self.TreeSSM(self.norm1(x_)))
                    x_ = x_ + self.drop_path(self.mlp(self.norm2(x_)))
                return x_

            if self.post_norm:
                x_ = x_ + self.drop_path(self.gamma1 * self.norm1(self.TreeSSM(x_)))
                x_ = x_ + self.drop_path(self.gamma2 * self.norm2(self.mlp(x_)))
            else:
                x_ = x_ + self.drop_path(self.gamma1 * self.TreeSSM(self.norm1(x_)))
                x_ = x_ + self.drop_path(self.gamma2 * self.mlp(self.norm2(x_)))
            return x_

        if self.with_cp and x.requires_grad:
            return checkpoint.checkpoint(_inner, x)
        return _inner(x)


class GrootVBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        downsample: bool = True,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path=0.0,
        act_layer: str = "GELU",
        norm_layer: str = "LN",
        post_norm: bool = False,
        layer_scale: Optional[float] = None,
        with_cp: bool = False,
        spe_fusion: str = "crossgate",
        use_ssf: bool = True,
        scan_strategy: str = "spa_causal",
    ):
        super().__init__()
        self.post_norm = post_norm

        self.blocks = nn.ModuleList([
            GrootVLayer(
                channels=channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                with_cp=with_cp,
                spe_fusion=spe_fusion,
                use_ssf=use_ssf,
                scan_strategy=scan_strategy,
            )
            for i in range(depth)
        ])

        self.norm = build_norm_layer(channels, "LN")
        self.downsample = DownsampleLayer(channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x: torch.Tensor, return_wo_downsample: bool = False):
        for blk in self.blocks:
            x = blk(x)

        if not self.post_norm:
            x = self.norm(x)

        if return_wo_downsample:
            x_ = x

        if self.downsample is not None:
            x = self.downsample(x)

        return (x, x_) if return_wo_downsample else x


class GrootV(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        depths: list[int] = [1],
        num_classes: int = 20,
        in_chans: int = 30,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.05,
        drop_path_rate: float = 0.2,
        drop_path_type: str = "linear",
        act_layer: str = "GELU",
        norm_layer: str = "BN",
        layer_scale: Optional[float] = None,
        post_norm: bool = False,
        with_cp: bool = False,
        cls_scale: int = 4,
        spe_fusion: str = "concat",
        use_ssf: bool = True,
        # scan strategy
        scan_strategy: str = "both_spacausal_spenoncausal",
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_levels = len(depths)
        self.num_features = int(channels * 2 ** (self.num_levels - 1))

        branch_mode, spa_causal, spe_causal = _normalize_scan_strategy(scan_strategy)

        self.patch_embed = StemLayer(
            in_chans=in_chans,
            out_chans=channels,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if drop_path_type == "uniform":
            dpr = [drop_path_rate for _ in dpr]

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = GrootVBlock(
                channels=int(channels * 2 ** i),
                depth=depths[i],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]): sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                with_cp=with_cp,
                spe_fusion=spe_fusion,
                use_ssf=use_ssf,
                scan_strategy=scan_strategy,
            )
            self.levels.append(level)

        self.conv_head = nn.Sequential(
            nn.Conv2d(self.num_features, int(self.num_features * cls_scale), kernel_size=1, bias=False),
            build_norm_layer(int(self.num_features * cls_scale), "BN", "channels_first", "channels_first"),
            build_act_layer(act_layer),
        )
        self.head = nn.Linear(int(self.num_features * cls_scale), num_classes) if num_classes > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def lr_decay_keywards(self, decay_ratio: float = 0.87):
        lr_ratios = {}
        idx = 0

        for layer_num in reversed(range(self.num_levels)):
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = f"levels.{layer_num}.blocks.{block_num}."
                lr_ratios[tag] = 1.0 * (decay_ratio ** idx)
                idx += 1

        if self.num_levels > 0:
            lr_ratios["patch_embed"] = lr_ratios.get("levels.0.blocks.0.", 1.0)
        for i in range(self.num_levels - 1):
            nxt = i + 1
            ref = lr_ratios.get(f"levels.{nxt}.blocks.0.", lr_ratios.get(f"levels.{i}.blocks.0.", 1.0))
            lr_ratios[f"levels.{i}.downsample"] = ref
            lr_ratios[f"levels.{i}.norm"] = ref

        return lr_ratios

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for level in self.levels:
            x = level(x)

        x = self.conv_head(x.permute(0, 3, 1, 2))
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward_features_seq_out(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        seq_out = []
        for level in self.levels:
            x, x_ = level(x, return_wo_downsample=True)
            seq_out.append(x_)
        return seq_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x.squeeze(1)
        x = self.forward_features(x)
        return self.head(x)