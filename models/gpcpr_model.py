""" Prototypical Network
1. generate 3D & text prototypes: point_prototypes, text_prototypes
2. Aveage fusion 3D & text prototypes: fusion_prototypes
3. QGPA query-guided prorotype adaption: fusion_prototype_post
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import DGCNN
from models.dgcnn_new import DGCNN_semseg
from models.attention import *
from models.gmmn import GMMNnetwork,ProjectorNetwork
from einops import rearrange, repeat
# from torch_cluster import fps
from models.similarity_head import ShallowSimilarityHead, DeepSimilarityHead, LogitsFusion, PointWiseDynamicFusion
from models.utils import PrototypeGuidedGating
from models.backbone_adapters import get_backbone


class BoundaryAwareShallowBranch(nn.Module):
    """Boundary-aware shallow branch for enhanced local feature extraction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Edge-aware convolution
        self.edge_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Boundary confidence head
        self.boundary_head = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )
        
        # Feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv1d(64 + 1, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, shallow_feat, xyz):
        """
        Args:
            shallow_feat: [B, C, N] or [C, N]
            xyz: [B, N, 3] or [N, 3]
        Returns:
            refined_shallow_feat: [B, C_out, N] or [C_out, N]
            boundary_confidence: [B, 1, N] or [1, N]
        """
        squeeze_batch = False
        if shallow_feat.dim() == 2:
            shallow_feat = shallow_feat.unsqueeze(0)
            squeeze_batch = True
        if xyz.dim() == 2:
            xyz = xyz.unsqueeze(0)

        # xyz is currently kept for interface compatibility and future geometry-aware refinement.
        _ = xyz

        edge_feat = self.edge_conv(shallow_feat)
        boundary_conf = torch.sigmoid(self.boundary_head(edge_feat))
        combined = torch.cat([edge_feat, boundary_conf], dim=1)
        refined = self.refine_conv(combined)

        if squeeze_batch:
            refined = refined.squeeze(0)
            boundary_conf = boundary_conf.squeeze(0)

        return refined, boundary_conf


class MutualAggregationModule(nn.Module):
    """Mutual Aggregation Module (MAM) for bidirectional support-query interaction"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Keep parameters for compatibility but operate in identity mode
        # Different support shots don't have point-index correspondence,
        # so we skip the aggregation that averages across shots
        self.num_heads = num_heads
        self.dim = dim

    def forward(self, support_feat, query_feat):
        """
        Args:
            support_feat: (n_way, k_shot, C, N)
            query_feat: (B, C, N)
        Returns:
            enhanced_support_feat: (n_way, k_shot, C, N) - unchanged
            enhanced_query_feat: (B, C, N) - unchanged
        
        Note: Operates in safety mode (identity) because different support shots
        don't have point-index correspondence. Averaging across shots would be invalid.
        """
        # Debug logs - file only when enabled
        if self.training and getattr(self, 'enable_debug_logs', False):
            debug_count = getattr(self, '_mam_debug_logged', 0)
            if debug_count < getattr(self, 'debug_log_first_n', 3):
                self.debug_log('\n=== MAM DEBUG (SAFETY MODE) ===')
                self.debug_log(f'MAM support_feat input shape: {support_feat.shape}')
                self.debug_log(f'MAM query_feat input shape: {query_feat.shape}')
                self.debug_log(f'MAM support_feat output shape: {support_feat.shape}')
                self.debug_log(f'MAM query_feat output shape: {query_feat.shape}')
                self.debug_log('MAM operating in identity mode (no changes)')
                self.debug_log('=== END MAM DEBUG ===')
                setattr(self, '_mam_debug_logged', debug_count + 1)
        
        # Identity mode - return inputs unchanged
        return support_feat, query_feat


class CommonalityBasedPrototypeSelection(nn.Module):
    """Commonality-based Prototype Selection (CPS) for semantic purification
    
    Support-only purification: computes similarity between support points and base prototype,
    then performs conservative residual blending.
    """
    def __init__(self, dim, cps_alpha=0.3):
        super().__init__()
        self.dim = dim
        self.cps_alpha = cps_alpha  # Blending weight for purified prototype

    def forward(self, support_feat, query_feat, fg_mask, bg_mask, n_way, k_shot):
        """
        Args:
            support_feat: (n_way, k_shot, C, N)
            query_feat: (B, C, N) - not used in support-only purification
            fg_mask: (n_way, k_shot, N)
            bg_mask: (n_way, k_shot, N)
            n_way: number of ways
            k_shot: number of shots
        Returns:
            purified_prototypes: list of [bg_prototype] + fg_prototypes
        """
        # Generate base prototypes (support-only, no query involved)
        fg_feat = support_feat * fg_mask.unsqueeze(2)
        bg_feat = support_feat * bg_mask.unsqueeze(2)

        fg_prototypes_base = []
        for way in range(n_way):
            sum_val = fg_mask[way].sum().clamp_min(1.0)
            fg_proto = fg_feat[way].sum(dim=(0, 2)) / sum_val
            fg_prototypes_base.append(fg_proto)

        bg_sum = bg_mask.sum().clamp_min(1.0)
        bg_prototype_base = bg_feat.sum(dim=(0, 1, 3)) / bg_sum

        # Support-only purification: compute similarity between support points and base prototypes
        # This avoids invalid query-support point correspondence
        fg_prototypes_purified = []
        for way in range(n_way):
            # Get foreground support features for this way
            fg_support = support_feat[way] * fg_mask[way].unsqueeze(1)  # [k_shot, C, N]
            
            # Compute similarity between support foreground points and base prototype
            # fg_support: [k_shot, C, N], fg_prototypes_base[way]: [C]
            similarity = F.cosine_similarity(
                fg_support, 
                fg_prototypes_base[way].unsqueeze(0).unsqueeze(-1), 
                dim=1
            )  # [k_shot, N]
            
            # Use similarity to weight support points (higher similarity = higher weight)
            # Clamp to avoid extreme weights
            weights = torch.clamp(similarity, min=0.0)  # [k_shot, N]
            
            # Apply weights and mask
            weighted_fg_support = fg_support * weights.unsqueeze(1)  # [k_shot, C, N]
            
            # Compute purified prototype
            sum_val = (weights * fg_mask[way]).sum().clamp_min(1.0)
            fg_proto_purified = weighted_fg_support.sum(dim=(0, 2)) / sum_val
            
            # Residual blending: final = (1-alpha)*base + alpha*purified
            fg_proto_final = (1 - self.cps_alpha) * fg_prototypes_base[way] + self.cps_alpha * fg_proto_purified
            fg_prototypes_purified.append(fg_proto_final)

        # Purify background prototype (support-only)
        bg_support = support_feat * bg_mask.unsqueeze(2)  # [n_way, k_shot, C, N]
        
        # Compute similarity between support background points and base prototype
        similarity_bg = F.cosine_similarity(
            bg_support, 
            bg_prototype_base.unsqueeze(0).unsqueeze(0).unsqueeze(-1), 
            dim=2
        )  # [n_way, k_shot, N]
        
        weights_bg = torch.clamp(similarity_bg, min=0.0)  # [n_way, k_shot, N]
        weighted_bg_support = bg_support * weights_bg.unsqueeze(2)  # [n_way, k_shot, C, N]
        
        sum_val_bg = (weights_bg * bg_mask).sum().clamp_min(1.0)
        bg_proto_purified = weighted_bg_support.sum(dim=(0, 1, 3)) / sum_val_bg
        
        # Residual blending for background
        bg_proto_final = (1 - self.cps_alpha) * bg_prototype_base + self.cps_alpha * bg_proto_purified

        # Debug logs - file only when enabled
        if self.training and getattr(self, 'enable_debug_logs', False):
            debug_count = getattr(self, '_cps_debug_logged', 0)
            if debug_count < getattr(self, 'debug_log_first_n', 3):
                self.debug_log('\n=== CPS DEBUG ===')
                self.debug_log(f'CPS alpha: {self.cps_alpha}')
                self.debug_log(f'Base fg prototype shapes: {[p.shape for p in fg_prototypes_base]}')
                self.debug_log(f'Base bg prototype shape: {bg_prototype_base.shape}')
                
                # Compute cosine similarities between base and purified
                self.debug_log('Cosine similarity between base and purified prototypes:')
                for way in range(n_way):
                    sim = F.cosine_similarity(fg_prototypes_base[way], fg_prototypes_purified[way], dim=0)
                    self.debug_log(f'  Foreground way {way}: {float(sim.item()):.4f}')
                bg_sim = F.cosine_similarity(bg_prototype_base, bg_proto_final, dim=0)
                self.debug_log(f'  Background: {float(bg_sim.item()):.4f}')
                
                # L2 norms
                self.debug_log('L2 norms:')
                for way in range(n_way):
                    self.debug_log(f'  FG way {way} - base: {float(fg_prototypes_base[way].norm().item()):.4f}, purified: {float(fg_prototypes_purified[way].norm().item()):.4f}')
                self.debug_log(f'  BG - base: {float(bg_prototype_base.norm().item()):.4f}, purified: {float(bg_proto_final.norm().item()):.4f}')
                
                # Check for NaN/Inf
                all_protos = [bg_proto_final] + fg_prototypes_purified
                has_nan = any(torch.any(torch.isnan(p)).item() for p in all_protos)
                has_inf = any(torch.any(torch.isinf(p)).item() for p in all_protos)
                self.debug_log(f'Prototypes contain NaN: {has_nan}')
                self.debug_log(f'Prototypes contain Inf: {has_inf}')
                self.debug_log('=== END CPS DEBUG ===')
                setattr(self, '_cps_debug_logged', debug_count + 1)

        # Combine prototypes
        purified_prototypes = [bg_proto_final] + fg_prototypes_purified
        return purified_prototypes


class BaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i - 1]
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_dim, params[i], 1),
                nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs - 1:
                x = F.relu(x)
        return x



class GPCPR(nn.Module):
    def __init__(self, args):
        super(GPCPR, self).__init__()
        # self.args = args
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = 'cosine'
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.use_linear_proj = args.use_linear_proj
        self.use_supervise_prototype = args.use_supervise_prototype # SR loss
        self.use_align = args.use_align # align loss
        self.sr_weight = getattr(args, 'sr_weight', 1.0) # Semantic Regularization loss weight
        # Get backbone using adapter
        self.encoder = get_backbone(args)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        if self.use_linear_proj:
            self.conv_1 = nn.Sequential(nn.Conv1d(args.train_dim, args.train_dim, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(args.train_dim),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.use_transformer = args.use_transformer
        if self.use_transformer:
            self.transformer = QGPA()

        # GPCPR add
        self.use_text = args.use_text
        self.use_text_diff = args.use_text_diff
        if args.use_text or args.use_text_diff:
            self.text_projector = ProjectorNetwork(args.noise_dim, args.train_dim, args.train_dim, args.gmm_dropout)
        if args.use_text:
            self.text_compressor = nn.MultiheadAttention(embed_dim=args.train_dim, num_heads=4, dropout=0.5)
        if args.use_text_diff:
            self.text_compressor_diff = nn.MultiheadAttention(embed_dim=args.train_dim, num_heads=4, dropout=0.5)

        self.use_pcpr=args.use_pcpr
        if args.use_pcpr:
            self.proto_compressor = MultiHeadAttention(in_channel=args.train_dim, out_channel=args.train_dim, n_heads=4,att_dropout=0.5, use_proj=False)

        self.use_dd_loss = args.use_dd_loss   #dd-loss
        self.dd_ratio1 = args.dd_ratio1
        self.dd_ratio2 = args.dd_ratio2

        sim_dim = args.train_dim
        if sim_dim is None:
            # 常见命名兜底：emb_dims / feat_dim 等（按你仓库实际字段调整）
            sim_dim = getattr(args, "emb_dims", None) or getattr(args, "feat_dim", None)
        if sim_dim is None:
            raise ValueError("Cannot infer sim_dim. Please set args.sim_dim to your feature dim C.")

        # Fusion configuration
        self.fusion_mode = getattr(args, "fusion_mode", "scalar")
        self.sim_head = nn.ModuleDict({
            "ssm": ShallowSimilarityHead(
                dim=sim_dim,
                num_heads=getattr(args, "ssm_heads", 4),
                init_scale=getattr(args, "ssm_init_scale", 10.0),
                attn_dropout=getattr(args, "ssm_attn_dropout", 0.0),
                proj_dropout=getattr(args, "ssm_proj_dropout", 0.0),
            ),
            "dsm": DeepSimilarityHead(
                dim=sim_dim,
                depth=getattr(args, "dsm_depth", 2),
                num_heads=getattr(args, "dsm_heads", 4),
                ffn_ratio=getattr(args, "dsm_ffn_ratio", 4.0),
                dropout=getattr(args, "dsm_dropout", 0.0),
                init_scale=getattr(args, "dsm_init_scale", 10.0),
            ),
            "fusion": LogitsFusion(init_alpha=getattr(args, "fusion_alpha", 0.5)),
        })

        # Add dynamic fusion if needed
        if self.fusion_mode == "dynamic":
            self.sim_head["dynamic_fusion"] = PointWiseDynamicFusion(
                dim=sim_dim,
                num_classes=self.n_way + 1
            )

        # Boundary-aware shallow branch
        self.use_boundary_shallow = getattr(args, "use_boundary_shallow", False)
        if self.use_boundary_shallow:
            self.boundary_branch = BoundaryAwareShallowBranch(
                in_channels=64,
                out_channels=sim_dim
            )
            self.lambda_boundary = getattr(args, "lambda_boundary", 0.1)
            self.boundary_knn_k = getattr(args, "boundary_knn_k", 5)

        # 添加任务感知的原型引导交叉门控模块
        # 计算正确的deep_dim值，使其与task_proto_expanded的实际维度匹配
        # 根据getFeatures方法的实现，task_proto_expanded的维度是320
        self.fusion = PrototypeGuidedGating(deep_dim=320, shallow_dim=64)

        # MAM and CPS modules
        self.use_mam = args.use_mam
        self.use_cps = args.use_cps
        if self.use_mam:
            self.mam = MutualAggregationModule(dim=args.train_dim, num_heads=4, dropout=0.1)
        if self.use_cps:
            self.cps = CommonalityBasedPrototypeSelection(
                dim=args.train_dim, 
                cps_alpha=getattr(args, 'cps_alpha', 0.3)
            )

        # Debug logging configuration
        self.logger = None
        self.enable_debug_logs = getattr(args, 'enable_debug_logs', False)
        self.debug_log_interval = getattr(args, 'debug_log_interval', 500)
        self.debug_log_first_n = getattr(args, 'debug_log_first_n', 3)
        self.debug_forward_counter = 0

    def set_logger(self, logger):
        self.logger = logger

    def debug_log(self, msg):
        if getattr(self, 'logger', None) is not None and hasattr(self.logger, 'debug'):
            self.logger.debug(str(msg))
        elif getattr(self, 'logger', None) is not None and hasattr(self.logger, 'fprint'):
            self.logger.fprint(str(msg))
        else:
            pass

    def forward(self, support_x, support_y, query_x, query_y, text_emb=None, text_emb_diff=None):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 1, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        # Debug counter and gating
        self.debug_forward_counter += 1
        do_debug = self.enable_debug_logs and (
            self.debug_forward_counter <= self.debug_log_first_n or
            self.debug_forward_counter % self.debug_log_interval == 0
        )

        # get features
        support_x = support_x.view(self.n_way * self.k_shot, self.in_channels, self.n_points)
        if self.use_attention:
            support_feat, support_xyz, support_shallow = self.getFeatures(support_x)
        else:
            support_feat, support_shallow = self.getFeatures(support_x)
            support_xyz = support_x[:, :3, :].transpose(1, 2)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)

        if self.use_attention:
            query_feat, query_xyz, query_shallow = self.getFeatures(query_x)
        else:
            query_feat, query_shallow = self.getFeatures(query_x)
            query_xyz = query_x[:, :3, :].transpose(1, 2)

        # Boundary-aware shallow branch
        boundary_loss = 0
        if self.use_boundary_shallow:
            support_shallow, _ = self.boundary_branch(support_shallow, support_xyz)
            query_shallow, query_boundary = self.boundary_branch(query_shallow, query_xyz)

            if self.training:
                query_boundary_labels = self.generate_boundary_labels(query_y, query_xyz)
                boundary_loss = F.binary_cross_entropy(query_boundary.squeeze(1), query_boundary_labels)

        # Reshape support shallow features
        support_shallow = support_shallow.view(self.n_way, self.k_shot, -1, self.n_points)

        # Mutual Aggregation Module (MAM)
        if self.use_mam:
            support_feat, query_feat = self.mam(support_feat, query_feat)

        # get bg/fg features: Fs'=Fs*Ms
        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        # Debug logs
        if do_debug:
            self.debug_log('\n=== PROTOTYPE DEBUG LOG ===')
            self.debug_log(f'support_feat shape: {support_feat.shape}')
            self.debug_log(f'query_feat shape: {query_feat.shape}')
            self.debug_log(f'fg_mask shape: {fg_mask.shape}')
            self.debug_log(f'bg_mask shape: {bg_mask.shape}')
            
            # Foreground point counts per way/shot
            fg_counts = fg_mask.sum(dim=-1)  # [n_way, k_shot]
            self.debug_log(f'Foreground point counts (way x shot):')
            for way in range(self.n_way):
                shot_counts = [int(fg_counts[way, shot].item()) for shot in range(self.k_shot)]
                self.debug_log(f'  Way {way}: {shot_counts} (total: {int(fg_counts[way].sum().item())})')
            
            # Background point counts per way/shot
            bg_counts = bg_mask.sum(dim=-1)  # [n_way, k_shot]
            self.debug_log(f'Background point counts (way x shot):')
            for way in range(self.n_way):
                shot_counts = [int(bg_counts[way, shot].item()) for shot in range(self.k_shot)]
                self.debug_log(f'  Way {way}: {shot_counts} (total: {int(bg_counts[way].sum().item())})')

        # Commonality-based Prototype Selection (CPS)
        if self.use_cps:
            # Apply CPS to purify prototypes
            purified_prototypes = self.cps(support_feat, query_feat, fg_mask, bg_mask, self.n_way, self.k_shot)
            prototypes = torch.stack(purified_prototypes, dim=0)
        else:
            # Use robust shot-level weighted prototype aggregation
            fg_prototypes, bg_prototype = self.getRobustWeightedPrototype(support_feat, fg_mask, bg_mask)
            prototypes = [bg_prototype] + fg_prototypes
            prototypes = torch.stack(prototypes, dim=0)
        
        # Additional debug logs for prototype quality
        if do_debug:
            self.debug_log(f'prototypes shape: {prototypes.shape}')
            # Background is prototypes[0], foreground are prototypes[1:]
            bg_norm = float(prototypes[0].norm().item())
            fg_norms = [float(prototypes[i].norm().item()) for i in range(1, prototypes.shape[0])]
            self.debug_log(f'Background prototype L2 norm: {bg_norm:.4f}')
            self.debug_log('Foreground prototype L2 norms:', [f'{n:.4f}' for n in fg_norms])
            self.debug_log(f'prototypes contains NaN: {torch.any(torch.isnan(prototypes)).item()}')
            self.debug_log(f'prototypes contains Inf: {torch.any(torch.isinf(prototypes)).item()}')
            self.debug_log('=== END PROTOTYPE DEBUG LOG ===')
        # 任务感知的原型引导交叉门控
        # 使用前景原型的平均值作为任务原型（排除背景）
        # 如果存在前景原型则使用前景，否则回退到全部原型
        if prototypes.shape[0] > 1:
            task_proto = prototypes[1:].mean(dim=0)  # [C_d] - 前景原型（排除背景）
        else:
            task_proto = prototypes.mean(dim=0)  # [C_d] - 安全回退
        # 扩展维度以匹配查询批次
        B = query_feat.shape[0]
        task_proto_expanded = task_proto.unsqueeze(0).repeat(B, 1)  # [B, C_d]
        # 应用融合模块
        query_refined = self.fusion(query_feat, query_shallow, task_proto_expanded)

        # save multi-stage results
        tep_proto = {}
        tep_pred = {}
        if self.use_dd_loss:
            tep_proto['orig'] = (prototypes.unsqueeze(0).repeat(query_refined.shape[0], 1, 1))
            tep_pred['orig'] = (torch.stack(
                [self.calculateSimilarity(query_refined, prototype, self.dist_method) for prototype in prototypes],
                dim=1))

        # GCPR - diverse text
        if self.use_text and text_emb is not None:
            text_emb = self.text_projector(text_emb)   # [3, num, dim]
            prototypes = prototypes.unsqueeze(1)+self.text_compressor(prototypes.unsqueeze(1), text_emb, text_emb,need_weights=False)[0] # (out,attn)
            prototypes = prototypes.squeeze(1)  # [3,320]
            # prototypes = prototypes.unsqueeze(1)+self.text_compressor(prototypes.unsqueeze(1).transpose(0,1), text_emb.transpose(0,1), text_emb.transpose(0,1),need_weights=False)[0].transpose(0,1) # (out,attn)
            # prototypes = prototypes.squeeze(1)  # [3,320]
            if self.use_dd_loss:
                tep_proto['text']=(prototypes.unsqueeze(0).repeat(query_refined.shape[0], 1, 1))
                tep_pred['text']=(torch.stack([self.calculateSimilarity(query_refined, prototype, self.dist_method) for prototype in prototypes],dim=1))
        # GCPR - differentiated text
        if self.use_text_diff and text_emb_diff is not None:
            text_emb_diff = self.text_projector(text_emb_diff)   # [3, num, dim]
            prototypes = prototypes.unsqueeze(1)+self.text_compressor_diff(prototypes.unsqueeze(1), text_emb_diff, text_emb_diff,need_weights=False)[0] # (out,attn)
            prototypes = prototypes.squeeze(1)  # [3,320]
            # prototypes = prototypes.unsqueeze(1)+self.text_compressor_diff(prototypes.unsqueeze(1).transpose(0,1), text_emb_diff.transpose(0,1), text_emb_diff.transpose(0,1),need_weights=False)[0].transpose(0,1) # (out,attn)
            # prototypes = prototypes.squeeze(1)  # [3,320]
            if self.use_dd_loss:
                tep_proto['text_diff']=(prototypes.unsqueeze(0).repeat(query_refined.shape[0], 1, 1))
                tep_pred['text_diff']=(torch.stack([self.calculateSimilarity(query_refined, prototype, self.dist_method) for prototype in prototypes],dim=1))

        # Semantic Regularization (SR) loss: Support Self-Alignment
        sr_loss = 0
        if self.use_supervise_prototype:
            # Use the same prototypes for SR loss as used for query prediction
            if self.use_transformer and 'prototypes_all_post' in locals():
                # Use adapted prototypes from QGPA for SR loss
                sr_loss = self.semantic_regularization_loss(prototypes_all_post, support_feat, fg_mask, bg_mask, True)
            else:
                # Use purified prototypes from CPS for SR loss
                sr_loss = self.semantic_regularization_loss(prototypes, support_feat, fg_mask, bg_mask, False)


        if self.use_transformer:   # QGPA & loss Lseg
            prototypes_all = prototypes.unsqueeze(0).repeat(query_refined.shape[0], 1, 1)  # [2,3,320]
            # Select best support shot per way instead of averaging across shots
            # (point indices don't correspond across different support point clouds)
            support_feat_ = self.selectSupportMemory(support_feat, fg_mask)  # [n_way, C, N]
            # Debug printing
            if do_debug:
                self.debug_log(f'DEBUG: support_memory shape: {support_feat_.shape}')
            prototypes_all_post = self.transformer(query_refined, support_feat_, prototypes_all)

            # 注释掉了
            # prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
            # similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for
            #               prototype in prototypes_new]
            # query_pred = torch.stack(similarity, dim=1)
            # ================== SSM + DSM similarity head (transformer branch) ==================

            # 添加
            #q_feat = query_feat.transpose(1, 2).contiguous()
            # query_refined: [B, C, Nq] -> [B, Nq, C]
            q_feat = query_refined.transpose(1, 2).contiguous()

            # prototypes_all_post: [B, K, C]
            proto = prototypes_all_post

            # Shallow Similarity Module (SSM)
            logits_s = self.sim_head["ssm"](q_feat, proto)  # [B, Nq, K]

            # Deep Similarity Module (DSM)
            logits_d = self.sim_head["dsm"](q_feat, proto)  # [B, Nq, K]

            # Fusion
            if self.fusion_mode == "dynamic":
                # Get shallow and deep features for dynamic fusion
                # For shallow features, use the refined shallow features
                shallow_feat_for_fusion = query_shallow.transpose(1, 2).contiguous()  # [B, Nq, C]
                # For deep features, use the query features
                deep_feat_for_fusion = q_feat  # [B, Nq, C]
                
                logits_final, alpha = self.sim_head["dynamic_fusion"](
                    logits_s, logits_d, shallow_feat_for_fusion, deep_feat_for_fusion
                )
            else:
                logits_final = self.sim_head["fusion"](logits_s, logits_d)  # [B, Nq, K]

            # keep original interface: [B, K, Nq]
            query_pred = logits_final.permute(0, 2, 1).contiguous()

            if self.use_dd_loss:
                tep_proto['qgpa']=(prototypes_all_post)
                tep_pred['qgpa']=(query_pred)

            if self.use_pcpr:
                query_bg_fg_features = self.extract_query_features(query_refined, query_pred)  # (n_way+1, kp, d) - 使用refined特征
                spt_prototypes = prototypes_all_post.transpose(0, 1)
                qry_bg_prototypes = self.proto_compressor([spt_prototypes[:1], query_bg_fg_features[:1], query_bg_fg_features[:1]])  # (n_way, n_proto, d)
                qry_fg_prototypes = self.proto_compressor([spt_prototypes[1:], query_bg_fg_features[1:],query_bg_fg_features[1:]])  # (n_way, n_proto, d)
                prototypes_all_post = torch.cat([qry_bg_prototypes, qry_fg_prototypes], dim=0).transpose(0, 1)
                prototypes_new = torch.chunk(prototypes_all_post, prototypes_all_post.shape[1], dim=1)
                similarity = [self.calculateSimilarity_trans(query_refined, prototype.squeeze(1), self.dist_method) for
                              prototype in prototypes_new]  # 使用refined特征
                query_pred = torch.stack(similarity, dim=1)
                if self.use_dd_loss:
                    tep_proto['pqmqm']=(prototypes_all_post)
                    tep_pred['pqmqm']=(query_pred)
            loss = self.computeCrossEntropyLoss(query_pred, query_y)
        else:
            # ================== SSM + DSM similarity head (no transformer branch) ==================

            # query_refined: [B, C, Nq] -> [B, Nq, C] - 使用refined特征
            q_feat = query_refined.transpose(1, 2).contiguous()

            # prototypes: [K, C] -> [B, K, C]
            proto = prototypes.unsqueeze(0).expand(q_feat.shape[0], -1, -1).contiguous()

            # SSM
            logits_s = self.sim_head["ssm"](q_feat, proto)  # [B, Nq, K]

            # DSM
            logits_d = self.sim_head["dsm"](q_feat, proto)  # [B, Nq, K]

            # Fusion
            if self.fusion_mode == "dynamic":
                # Get shallow and deep features for dynamic fusion
                shallow_feat_for_fusion = query_shallow.transpose(1, 2).contiguous()  # [B, Nq, C]
                deep_feat_for_fusion = q_feat  # [B, Nq, C]
                
                logits_final, alpha = self.sim_head["dynamic_fusion"](
                    logits_s, logits_d, shallow_feat_for_fusion, deep_feat_for_fusion
                )
            else:
                logits_final = self.sim_head["fusion"](logits_s, logits_d)

            # [B, K, Nq] for loss & evaluation
            query_pred = logits_final.permute(0, 2, 1).contiguous()

            loss = self.computeCrossEntropyLoss(query_pred, query_y)   # segmentation loss

        align_loss = 0
        if self.use_align:
            align_loss = align_loss + self.alignLoss_trans(query_refined, query_pred, support_feat, fg_mask, bg_mask)  # 使用refined特征

        # Initialize mam_loss as zero tensor (MAM operates in identity mode)
        mam_loss = query_refined.new_tensor(0.0)

        dd_loss = 0
        if self.use_dd_loss and self.use_pcpr and self.use_transformer:
            kl = torch.nn.KLDivLoss()
            T = 2
            keys = list(tep_proto.keys())
            if 'qgpa' in keys and 'pqmqm' in keys:
                dd_loss = dd_loss + self.dd_ratio1 * kl(F.log_softmax(tep_proto['qgpa'] / T, dim=-1),
                                                             F.softmax(tep_proto['pqmqm'].detach() / T, dim=-1)) * T * T  # [2, 3, 320]
                dd_loss = dd_loss + self.dd_ratio2 * kl(F.log_softmax(tep_pred['qgpa'] / T, dim=-2),
                                                          F.softmax(tep_pred['pqmqm'].detach() / T, dim=-2)) * T * T  # [2, 3, 2048]
            if 'text' in keys and 'text_diff' in keys:
                dd_loss = dd_loss + self.dd_ratio1 * kl(F.log_softmax(tep_proto['text'] / T, dim=-1),
                                                         F.softmax(tep_proto['text_diff'].detach() / T, dim=-1)) * T * T  # [2, 3, 320]

        # Add boundary loss if enabled
        total_loss = loss + align_loss + sr_loss * self.sr_weight + dd_loss + mam_loss
        if self.use_boundary_shallow:
            total_loss += boundary_loss * self.lambda_boundary

        # Logits diagnostics
        if do_debug:
            self.debug_log('\n=== LOGITS DIAGNOSTICS ===')
            self.debug_log(f'query_pred shape: {query_pred.shape}')
            self.debug_log(f'Logits stats - min: {query_pred.min().item():.4f}, max: {query_pred.max().item():.4f}, mean: {query_pred.mean().item():.4f}')
            
            # Per-class logits mean
            if query_pred.dim() >= 2:
                for cls in range(query_pred.shape[1]):
                    cls_mean = query_pred[:, cls].mean().item()
                    self.debug_log(f'  Class {cls} logits mean: {cls_mean:.4f}')
            
            # Target and prediction analysis
            if query_y is not None:
                target_unique = torch.unique(query_y)
                self.debug_log(f'Target unique values: {sorted([int(v.item()) for v in target_unique])}')
                
                query_pred_argmax = query_pred.argmax(dim=1)
                pred_unique = torch.unique(query_pred_argmax)
                self.debug_log(f'Prediction unique values: {sorted([int(v.item()) for v in pred_unique])}')
                
                # Check for collapse
                if len(pred_unique) == 1:
                    self.debug_log(f'WARNING: Predictions collapsed to single class: {int(pred_unique[0].item())}')
                if len(pred_unique) == 2 and 0 in [int(v.item()) for v in pred_unique]:
                    bg_ratio = (query_pred_argmax == 0).float().mean().item()
                    if bg_ratio > 0.95:
                        self.debug_log(f'WARNING: {bg_ratio:.1%} of predictions are background')
            
            self.debug_log('=== END LOGITS DIAGNOSTICS ===')

        return query_pred, total_loss


    def semantic_regularization_loss(self, prototypes, support_feat, fg_mask, bg_mask, use_transformer):
        """
        Semantic Regularization (SR) loss: Support Self-Alignment
        Ensures that prototypes can correctly segment the support set itself

        Args:
            prototypes: prototypes used for prediction (either from CPS or QGPA)
                shape: (K, C) if not using transformer, (B, K, C) if using transformer
            support_feat: support features
                shape: (n_way, k_shot, C, N)
            fg_mask: foreground masks for support images
                shape: (n_way, k_shot, N)
            bg_mask: background masks for support images
                shape: (n_way, k_shot, N)
            use_transformer: whether prototypes are from transformer (QGPA)
        """
        n_ways, n_shots = self.n_way, self.k_shot
        loss = 0

        for way in range(n_ways):
            for shot in range(n_shots):
                # Get support features for this way and shot
                img_fts = support_feat[way, shot].unsqueeze(0)  # (1, C, N)
                
                # Get prototypes: background + current way foreground
                if use_transformer:
                    # For transformer, prototypes have shape (B, K, C), take first batch
                    bg_proto = prototypes[0, 0]
                    fg_proto = prototypes[0, way + 1]
                else:
                    # For non-transformer, prototypes have shape (K, C)
                    bg_proto = prototypes[0]
                    fg_proto = prototypes[way + 1]
                
                # Calculate similarity using the same method as query set
                # Use the same similarity head approach
                img_fts_transposed = img_fts.transpose(1, 2).contiguous()  # (1, N, C)
                proto = torch.stack([bg_proto, fg_proto], dim=0).unsqueeze(0)  # (1, 2, C)
                
                # Shallow Similarity Module (SSM)
                logits_s = self.sim_head["ssm"](img_fts_transposed, proto)  # (1, N, 2)
                
                # Deep Similarity Module (DSM)
                logits_d = self.sim_head["dsm"](img_fts_transposed, proto)  # (1, N, 2)
                
                # Fusion
                logits_final = self.sim_head["fusion"](logits_s, logits_d)  # (1, N, 2)
                
                # Reshape for loss: (1, 2, N)
                supp_pred = logits_final.permute(0, 2, 1).contiguous()
                
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fg_mask[way, shot], 255, device=img_fts.device).long()
                supp_label[fg_mask[way, shot] == 1] = 1  # foreground
                supp_label[bg_mask[way, shot] == 1] = 0  # background
                
                # Compute Cross-Entropy loss
                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        
        return loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features.

        Args:
            x: input data with shape (B, C_in, L)
        Returns:
            if use_attention:
                feat: (B, C_out, L), xyz: (B, L, 3), shallow_feat: (B, C_s, L)
            else:
                feat: (B, C_out, L), shallow_feat: (B, C_s, L)
        """
        enc_out = self.encoder(x)
        if not isinstance(enc_out, dict):
            raise TypeError(f"encoder output must be a dict, got {type(enc_out)}")

        feat_level2 = enc_out['final_feat']
        x_shallow = enc_out['shallow_feat']
        xyz = enc_out['xyz']
        multi_scale_feats = enc_out.get('multi_scale_feats', [])
        if isinstance(multi_scale_feats, torch.Tensor):
            multi_scale_feats = [multi_scale_feats]
        elif multi_scale_feats is None:
            multi_scale_feats = []
        else:
            multi_scale_feats = list(multi_scale_feats)

        if not isinstance(feat_level2, torch.Tensor):
            raise TypeError(f"final_feat must be a tensor, got {type(feat_level2)}")
        if not isinstance(x_shallow, torch.Tensor):
            raise TypeError(f"shallow_feat must be a tensor, got {type(x_shallow)}")
        if not isinstance(xyz, torch.Tensor):
            raise TypeError(f"xyz must be a tensor, got {type(xyz)}")

        if xyz.dim() != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must have shape [B, N, 3], got {tuple(xyz.shape)}")
        if feat_level2.dim() != 3:
            raise ValueError(f"final_feat must have shape [B, C, N], got {tuple(feat_level2.shape)}")
        if x_shallow.dim() != 3:
            raise ValueError(f"shallow_feat must have shape [B, C, N], got {tuple(x_shallow.shape)}")

        if len(multi_scale_feats) == 0:
            multi_scale_feats = [x_shallow, feat_level2]

        feat_level3 = self.base_learner(feat_level2)

        if self.use_attention:
            local_feats = list(multi_scale_feats[:3])
            while len(local_feats) < 3:
                local_feats.append(local_feats[-1] if len(local_feats) > 0 else feat_level2)

            att_feat = self.att_learner(feat_level2)
            fused = torch.cat((local_feats[0], local_feats[1], local_feats[2], att_feat, feat_level3), dim=1)
            if self.use_linear_proj:
                fused = self.conv_1(fused)
            return fused, xyz, x_shallow

        local_feat = multi_scale_feats[0]
        map_feat = self.linear_mapper(feat_level2)
        return torch.cat((local_feat, map_feat, feat_level3), dim=1), x_shallow

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype (original equal-shot averaging)

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0, 1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def selectSupportMemory(self, support_feat, fg_mask):
        """
        Select best support shot per way based on foreground point count.
        Different support shots don't have point-index correspondence,
        so we select the shot with most foreground points instead of averaging.

        Args:
            support_feat: support features, shape: (n_way, k_shot, C, N)
            fg_mask: foreground masks, shape: (n_way, k_shot, N)
        Returns:
            support_memory: selected support features, shape: (n_way, C, N)
        """
        n_way, k_shot, C, N = support_feat.shape
        
        # Count foreground points per way/shot
        fg_counts = fg_mask.float().sum(dim=-1)  # [n_way, k_shot]
        
        # Select best shot index per way
        best_ids = fg_counts.argmax(dim=1)  # [n_way]
        
        # Debug logs
        if self.training and getattr(self, '_proto_debug_logged', 0) <= 3:
            print('\n=== SUPPORT MEMORY SELECTION DEBUG ===')
            print(f'support_feat shape: {support_feat.shape}')
            print(f'fg_mask shape: {fg_mask.shape}')
            print(f'Foreground counts (way x shot):')
            for way in range(n_way):
                print(f'  Way {way}: {[int(fg_counts[way, s].item()) for s in range(k_shot)]}')
            print(f'Selected best shot IDs: {[int(best_ids[way].item()) for way in range(n_way)]}')
            print(f'Selected foreground counts: {[int(fg_counts[way, best_ids[way]].item()) for way in range(n_way)]}')
            print('=== END SUPPORT MEMORY SELECTION DEBUG ===')
        
        # Select best shot for each way
        support_memory = []
        for way in range(n_way):
            best_shot = best_ids[way]
            support_memory.append(support_feat[way, best_shot])
        
        support_memory = torch.stack(support_memory, dim=0)  # [n_way, C, N]
        
        return support_memory

    def getWeightedPrototype(self, support_feat, fg_mask, bg_mask):
        """
        Compute prototypes using point-count weighted aggregation.
        Each support shot contributes proportionally to the number of valid points it contains.

        Args:
            support_feat: support features, shape: (n_way, k_shot, C, N)
            fg_mask: foreground masks, shape: (n_way, k_shot, N)
            bg_mask: background masks, shape: (n_way, k_shot, N)
        Returns:
            fg_prototypes: list of n_way foreground prototypes, each [C]
            bg_prototype: background prototype [C]
        """
        n_way, k_shot, C, N = support_feat.shape
        
        # Compute foreground prototypes (one per way)
        fg_prototypes = []
        for way in range(n_way):
            # Get all foreground points for this way across all shots
            # support_feat[way]: [k_shot, C, N]
            # fg_mask[way]: [k_shot, N]
            fg_mask_expanded = fg_mask[way].unsqueeze(1)  # [k_shot, 1, N]
            weighted_feat = support_feat[way] * fg_mask_expanded  # [k_shot, C, N]
            
            # Sum across shots and points
            fg_sum = weighted_feat.sum(dim=(0, 2))  # [C]
            fg_count = fg_mask[way].sum()  # scalar
            fg_count = fg_count.clamp_min(1.0)  # Avoid division by zero
            
            fg_proto = fg_sum / fg_count
            fg_prototypes.append(fg_proto)
        
        # Compute background prototype (all background points across all ways and shots)
        bg_mask_expanded = bg_mask.unsqueeze(2)  # [n_way, k_shot, 1, N]
        weighted_bg_feat = support_feat * bg_mask_expanded  # [n_way, k_shot, C, N]
        
        bg_sum = weighted_bg_feat.sum(dim=(0, 1, 3))  # [C]
        bg_count = bg_mask.sum()  # scalar
        bg_count = bg_count.clamp_min(1.0)  # Avoid division by zero
        
        bg_prototype = bg_sum / bg_count
        
        return fg_prototypes, bg_prototype

    def getRobustWeightedPrototype(self, support_feat, fg_mask, bg_mask, tau=0.2):
        """
        Compute robust weighted prototypes using shot-level reliability weighting.
        Combines point-count weighting with cosine similarity-based reliability weighting.
        
        Args:
            support_feat: support features, shape: (n_way, k_shot, C, N)
            fg_mask: foreground masks, shape: (n_way, k_shot, N)
            bg_mask: background masks, shape: (n_way, k_shot, N)
            tau: temperature for softmax on cosine similarities
        Returns:
            fg_prototypes: list of n_way foreground prototypes, each [C]
            bg_prototype: background prototype [C]
        """
        n_way, k_shot, C, N = support_feat.shape
        
        # Debug logging
        do_debug = self.should_debug_log()
        
        # Compute foreground prototypes (one per way) with shot-level reliability weighting
        fg_prototypes = []
        for way in range(n_way):
            # Compute shot prototypes and counts
            shot_protos = []
            shot_counts = []
            
            for shot in range(k_shot):
                # Get foreground mask for this shot
                shot_mask = fg_mask[way, shot].float()  # [N]
                mask_sum = shot_mask.sum()
                
                if mask_sum > 0:
                    # Compute shot prototype
                    masked_feat = support_feat[way, shot] * shot_mask.unsqueeze(0)  # [C, N]
                    shot_proto = masked_feat.sum(dim=-1) / mask_sum  # [C]
                else:
                    # If no foreground points, use zero vector
                    shot_proto = support_feat.new_zeros(C)
                
                shot_protos.append(shot_proto)
                shot_counts.append(mask_sum)
            
            # Convert to tensors
            shot_protos = torch.stack(shot_protos, dim=0)  # [k_shot, C]
            shot_counts = torch.tensor(shot_counts, device=support_feat.device)  # [k_shot]
            
            # Compute preliminary mean prototype
            valid_mask = shot_counts > 0
            if valid_mask.any():
                # Weight by point count for preliminary mean
                count_weights = shot_counts.clamp_min(0) / shot_counts.clamp_min(0).sum().clamp_min(1.0)
                mean_proto = (shot_protos * count_weights.unsqueeze(-1)).sum(dim=0)  # [C]
            else:
                # Fallback: use average of all shot protos
                mean_proto = shot_protos.mean(dim=0)  # [C]
            
            # Compute cosine similarity between each shot proto and mean proto
            shot_protos_norm = shot_protos.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            mean_proto_norm = mean_proto.norm().clamp_min(1e-6)
            similarities = (shot_protos * mean_proto.unsqueeze(0)).sum(dim=-1) / (shot_protos_norm.squeeze() * mean_proto_norm)
            similarities = similarities.clamp(-1, 1)  # Ensure valid cosine values
            
            # Compute count weights (normalized point counts)
            count_w = shot_counts.clamp_min(0) / shot_counts.clamp_min(0).sum().clamp_min(1.0)
            
            # Compute similarity-based reliability weights
            sim_w = F.softmax(similarities / tau, dim=0)
            
            # Combine weights: 70% count-based, 30% reliability-based
            final_w = 0.7 * count_w + 0.3 * sim_w
            final_w = final_w / final_w.sum().clamp_min(1e-6)  # Normalize
            
            # Compute final foreground prototype
            fg_proto = (shot_protos * final_w.unsqueeze(-1)).sum(dim=0)  # [C]
            fg_prototypes.append(fg_proto)
            
            # Debug logging
            if do_debug:
                self.log(f'\n[Way {way} Shot Weights]')
                self.log(f'  Point counts: {[int(c.item()) for c in shot_counts]}')
                self.log(f'  Cosine similarities: {[f"{s:.4f}" for s in similarities]}')
                self.log(f'  Count weights: {[f"{w:.4f}" for w in count_w]}')
                self.log(f'  Sim weights: {[f"{w:.4f}" for w in sim_w]}')
                self.log(f'  Final weights: {[f"{w:.4f}" for w in final_w]}')
        
        # Compute background prototype (point-count weighted, unchanged)
        bg_mask_expanded = bg_mask.unsqueeze(2).float()  # [n_way, k_shot, 1, N]
        weighted_bg_feat = support_feat * bg_mask_expanded  # [n_way, k_shot, C, N]
        
        bg_sum = weighted_bg_feat.sum(dim=(0, 1, 3))  # [C]
        bg_count = bg_mask.float().sum()  # scalar
        bg_count = bg_count.clamp_min(1.0)  # Avoid division by zero
        
        bg_prototype = bg_sum / bg_count
        
        # Debug logging
        if do_debug:
            fg_norms = [float(p.norm().item()) for p in fg_prototypes]
            self.log(f'\n[Robust Prototype Stats]')
            self.log(f'  Foreground prototype L2 norms: {[f"{n:.4f}" for n in fg_norms]}')
            self.log(f'  Background prototype L2 norm: {float(bg_prototype.norm().item()):.4f}')
            all_protos = torch.stack([bg_prototype] + fg_prototypes)
            self.log(f'  Prototypes contain NaN: {torch.any(torch.isnan(all_protos)).item()}')
            self.log(f'  Prototypes contain Inf: {torch.any(torch.isinf(all_protos)).item()}')
        
        return fg_prototypes, bg_prototype


    def calculateSimilarity(self, feat, prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':  # prototype[None, ..., None] [1, 320, 1]
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2) ** 2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        return similarity



    def calculateSimilarity_trans(self, feat, prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[..., None], p=2) ** 2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        return similarity

    def calculateSimilarity_trans(self, feat, prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[..., None], p=2) ** 2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels, keep_ratio=0.3):
        """ Calculate the OHEM Loss for query set
        """
        from models.utils import calc_ohem_loss
        return calc_ohem_loss(query_logits, query_labels, keep_ratio=keep_ratio)

    def extract_query_features(self, qry_fts, pred):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x num_points
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_feature = (qry_fts.unsqueeze(1) * pred_mask)

        return rearrange(qry_fts.unsqueeze(1) * pred_mask,'k n d p -> n (k p) d')

    def generate_boundary_labels(self, labels, xyz):
        """
        Generate boundary labels based on kNN neighbors.
        A point is a boundary point if it has at least one neighbor with a different label.
        
        Args:
            labels: [B, N]
            xyz: [B, N, 3]
        Returns:
            boundary_labels: [B, N]
        """
        B, N = labels.shape
        boundary_labels = torch.zeros_like(labels, dtype=torch.float32, device=labels.device)
        
        for b in range(B):
            # Compute pairwise distances
            xyz_b = xyz[b]
            dist = torch.cdist(xyz_b, xyz_b)
            
            # Get k nearest neighbors
            _, idx = dist.topk(self.boundary_knn_k + 1, dim=1, largest=False)
            idx = idx[:, 1:]  # Exclude self
            
            # Get neighbor labels
            neighbor_labels = labels[b][idx]
            
            # Check if any neighbor has different label
            boundary = torch.any(neighbor_labels != labels[b].unsqueeze(1), dim=1)
            boundary_labels[b] = boundary.float()
        
        return boundary_labels

    def alignLoss_trans(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x num_points
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'

        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
        # print('qry_prototypes shape',qry_prototypes.shape)   # [3,320]
        # print('text_prototypes shape',text_prototypes.shape)   #[2,3,320]
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)
                prototypes_all = torch.stack(prototypes, dim=0).unsqueeze(0)
                prototypes_all_post = self.transformer(img_fts, qry_fts.mean(0).unsqueeze(0), prototypes_all)
                prototypes_new = [prototypes_all_post[0, 0], prototypes_all_post[0, 1]]

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in
                             prototypes_new]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss