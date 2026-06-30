"""
model.py
智能音乐推荐系统 - 核心模型
包含: LightGCN召回、SASRec精排、动态权重系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


# ==================== 配置类 ====================

@dataclass
class ModelConfig:
    """模型配置参数"""
    num_users: int = 50           # 用户数量
    num_items: int = 2000         # 歌曲数量
    embedding_dim: int = 64       # 嵌入维度
    lightgcn_layers: int = 3      # LightGCN层数
    sasrec_maxlen: int = 50       # SASRec最大序列长度
    sasrec_num_blocks: int = 2    # SASRec Transformer块数
    sasrec_num_heads: int = 2     # SASRec注意力头数
    sasrec_dropout: float = 0.2   # SASRec dropout率
    device: str = 'cpu'


# ==================== 音乐特征定义 ====================

class MusicFeatureDim:
    """音乐多维度特征定义"""
    # 流派 (10种)
    GENRES = ['流行', '摇滚', '电子', '民谣', '说唱', 'R&B', '爵士', '古典', '金属', '蓝调']
    # 情绪 (8种)
    EMOTIONS = ['快乐', '悲伤', '愤怒', '平静', '兴奋', '忧郁', '浪漫', '怀旧']
    # 创作目的/主题 (10种)
    PURPOSES = ['励志', '青春', '爱情', '分手', '梦想', '友情', '孤独', '治愈', '派对', '旅行']
    # 场景 (6种)
    SCENES = ['运动', '学习', '通勤', '休息', '聚会', '睡前']
    # 时间段 (4种)
    TIME_SLOTS = ['早晨(6-9)', '上午(9-12)', '下午(12-18)', '晚上(18-24)']


# ==================== LightGCN 召回模型 ====================

class LightGCN(nn.Module):
    """
    LightGCN: 简化图卷积网络用于协同过滤召回
    论文: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
    """
    def __init__(self, config: ModelConfig):
        super(LightGCN, self).__init__()
        self.config = config
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.embedding_dim = config.embedding_dim
        self.num_layers = config.lightgcn_layers

        # 用户和物品嵌入 (唯一可学习参数)
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        # Xavier初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # 存储归一化邻接矩阵
        self.norm_adj = None

    def set_norm_adj_matrix(self, norm_adj: torch.Tensor):
        """设置归一化邻接矩阵 (稀疏格式)"""
        self.norm_adj = norm_adj

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: 执行图卷积并返回最终用户/物品嵌入
        Returns: (user_embeddings, item_embeddings)
        """
        # 初始嵌入
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)  # shape: (num_users + num_items, embedding_dim)

        embeddings_list = [all_embeddings]

        # 执行K层图卷积 (线性传播，无激活函数，无特征变换)
        for layer in range(self.num_layers):
            if self.norm_adj is not None:
                all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            else:
                all_embeddings = all_embeddings
            embeddings_list.append(all_embeddings)

        # 层聚合: 取所有层的平均
        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)

        # 分离用户和物品嵌入
        user_embeddings = final_embeddings[:self.num_users]
        item_embeddings = final_embeddings[self.num_users:]

        return user_embeddings, item_embeddings

    def get_rec_scores(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        计算用户对所有物品的评分 (用于召回)
        Returns: (num_users, num_items) 分数矩阵
        """
        user_emb, item_emb = self.forward()
        user_emb_selected = user_emb[user_ids]  # (batch, embedding_dim)
        scores = torch.matmul(user_emb_selected, item_emb.t())  # (batch, num_items)
        return scores


# ==================== SASRec 精排模型 ====================

class PointWiseFeedForward(nn.Module):
    """SASRec中的前馈网络"""
    def __init__(self, hidden_units: int, dropout_rate: float):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dropout2(self.conv2(self.relu(
            self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        )))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs  # 残差连接
        return outputs


class SASRec(nn.Module):
    """
    SASRec: 基于自注意力的序列推荐模型 (精排)
    论文: "Self-Attentive Sequential Recommendation"
    """
    def __init__(self, config: ModelConfig):
        super(SASRec, self).__init__()
        self.config = config
        self.num_items = config.num_items
        self.embedding_dim = config.embedding_dim
        self.maxlen = config.sasrec_maxlen
        self.num_blocks = config.sasrec_num_blocks
        self.num_heads = config.sasrec_num_heads

        # 物品嵌入 (同时作为位置嵌入的查询)
        self.item_embedding = nn.Embedding(self.num_items + 1, self.embedding_dim, padding_idx=0)

        # 位置嵌入
        self.pos_embedding = nn.Embedding(self.maxlen, self.embedding_dim)

        # 嵌入dropout
        self.emb_dropout = nn.Dropout(p=config.sasrec_dropout)

        # Transformer块 (自注意力 + 前馈)
        self.attention_layers = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.layer_norms1 = nn.ModuleList()
        self.layer_norms2 = nn.ModuleList()

        for _ in range(self.num_blocks):
            self.attention_layers.append(
                nn.MultiheadAttention(self.embedding_dim, self.num_heads, dropout=config.sasrec_dropout, batch_first=True)
            )
            self.forward_layers.append(PointWiseFeedForward(self.embedding_dim, config.sasrec_dropout))
            self.layer_norms1.append(nn.LayerNorm(self.embedding_dim, eps=1e-8))
            self.layer_norms2.append(nn.LayerNorm(self.embedding_dim, eps=1e-8))

        # 输出层: 预测下一个物品
        self.output_layer = nn.Linear(self.embedding_dim, self.num_items)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, seq: torch.Tensor, seq_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            seq: (batch, seq_len) 物品ID序列
            seq_len: (batch,) 每个序列的实际长度
        Returns:
            logits: (batch, num_items) 对每个物品的预测分数
        """
        batch_size, seq_len_val = seq.size()

        # 物品嵌入 + 位置嵌入
        item_emb = self.item_embedding(seq)  # (batch, seq_len, embedding_dim)
        positions = torch.arange(seq_len_val, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)

        seq_emb = item_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)

        # 生成因果掩码 (防止看到未来信息)
        mask = torch.triu(torch.ones(seq_len_val, seq_len_val, device=seq.device), diagonal=1).bool()
        padding_mask = (seq == 0)  # (batch, seq_len)

        # Transformer块
        for i in range(self.num_blocks):
            # 自注意力
            seq_emb_norm = self.layer_norms1[i](seq_emb)
            attn_out, _ = self.attention_layers[i](
                seq_emb_norm, seq_emb_norm, seq_emb_norm,
                attn_mask=mask,
                key_padding_mask=padding_mask
            )
            seq_emb = seq_emb + attn_out  # 残差连接

            # 前馈网络
            seq_emb_norm = self.layer_norms2[i](seq_emb)
            seq_emb = seq_emb + self.forward_layers[i](seq_emb_norm)

        # 取序列最后一个有效位置作为用户表示
        if seq_len is not None:
            user_repr = seq_emb[torch.arange(batch_size), seq_len - 1]
        else:
            last_positions = (seq != 0).sum(dim=1) - 1
            user_repr = seq_emb[torch.arange(batch_size), last_positions]

        # 预测下一个物品
        logits = self.output_layer(user_repr)  # (batch, num_items)
        return logits

    def predict(self, seq: torch.Tensor, candidate_items: torch.Tensor) -> torch.Tensor:
        """
        对候选物品进行精排打分
        Args:
            seq: (batch, seq_len) 历史序列
            candidate_items: (batch, num_candidates) 候选物品ID
        Returns:
            scores: (batch, num_candidates) 精排分数
        """
        logits = self.forward(seq)  # (batch, num_items)
        scores = logits.gather(1, candidate_items)
        return scores


# ==================== 动态权重系统 ====================

class DynamicWeightSystem:
    """
    动态权重调整系统
    为每个推荐逻辑标签维护权重，根据用户行为动态调整
    """
    def __init__(self):
        # 所有推荐逻辑标签及其基础权重
        self.weights = {
            'is_favorited': 1.0,      # 用户是否收藏
            'play_count': 1.0,        # 用户听歌次数
            'long_time_no_play': 1.0, # 用户是否长时间未收听
            'time_slot': 1.0,         # 用户听歌时间段
            'scene': 1.0,             # 用户听歌场景
            'dwell_time': 1.0,        # 用户听一首新歌停留时间
            'artist': 1.0,            # 该歌曲的歌手
            'genre': 1.0,             # 流派
            'emotion': 1.0,           # 表达的情绪
            'purpose': 1.0,           # 创作目的
        }

        # 权重调整历史记录 (用于追踪变化)
        self.weight_history = defaultdict(list)

        # 权重上下限
        self.min_weight = 0.5
        self.max_weight = 3.0

        # 调整步长
        self.adjust_step = 0.1

    def get_weight(self, tag: str) -> float:
        """获取某个标签的当前权重"""
        return self.weights.get(tag, 1.0)

    def get_all_weights(self) -> Dict[str, float]:
        """获取所有权重"""
        return dict(self.weights)

    def adjust_weight(self, tag: str, reason: str, increase: bool = True):
        """
        调整权重
        Args:
            tag: 标签名称
            reason: 调整原因 (用于日志)
            increase: True=增加, False=减少
        """
        old_weight = self.weights[tag]

        if increase:
            new_weight = min(old_weight + self.adjust_step, self.max_weight)
        else:
            new_weight = max(old_weight - self.adjust_step, self.min_weight)

        self.weights[tag] = new_weight

        # 记录历史
        self.weight_history[tag].append({
            'old': old_weight,
            'new': new_weight,
            'reason': reason,
            'change': 'increase' if increase else 'decrease'
        })

        return new_weight

    def update_weights_by_behavior(self, behavior_change: Dict):
        """
        根据用户行为变化批量更新权重
        behavior_change: {
            'tag_name': {
                'type': 'sharp_increase' | 'state_change' | 'pattern_match',
                'description': '描述'
            }
        }
        """
        updates = []

        for tag, info in behavior_change.items():
            if tag not in self.weights:
                continue

            if info['type'] == 'sharp_increase':
                new_w = self.adjust_weight(tag, info['description'], increase=True)
                updates.append(f"{tag}: {self.weights[tag]-0.1:.1f} -> {new_w:.1f} (急剧上升)")

            elif info['type'] == 'state_change':
                new_w = self.adjust_weight(tag, info['description'], increase=True)
                updates.append(f"{tag}: {self.weights[tag]-0.1:.1f} -> {new_w:.1f} (状态变化)")

            elif info['type'] == 'pattern_match':
                new_w = self.adjust_weight(tag, info['description'], increase=True)
                updates.append(f"{tag}: {self.weights[tag]-0.1:.1f} -> {new_w:.1f} (模式匹配)")

        return updates

    def compute_final_score(self, tag_scores: Dict[str, float]) -> float:

        total_weight = 0.0
        weighted_sum = 0.0

        for tag, score in tag_scores.items():
            weight = self.weights.get(tag, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight


# ==================== 完整推荐引擎 ====================

class MusicRecommendationEngine:
    """
    完整的音乐推荐引擎
    整合 LightGCN召回 + SASRec精排 + 动态权重 + 多维度特征
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device)

        # 初始化模型
        self.lightgcn = LightGCN(config).to(self.device)
        self.sasrec = SASRec(config).to(self.device)

        # 动态权重系统
        self.weight_system = DynamicWeightSystem()

        # 用户-物品交互图 (用于LightGCN)
        self.interaction_matrix = None

        # 用户行为序列 (用于SASRec)
        self.user_sequences = {}  # {user_id: [item_id, ...]}

        # 歌曲元数据
        self.song_metadata = {}  # {item_id: {genre, emotion, purpose, artist, ...}}

        # 用户画像
        self.user_profiles = {}  # {user_id: {favorite_artists, time_slots, scenes, ...}}

    def build_interaction_graph(self, interactions: List[Tuple[int, int, float]]):
        """
        构建用户-物品交互图 (用于LightGCN)
        interactions: [(user_id, item_id, rating), ...]
        """
        num_users = self.config.num_users
        num_items = self.config.num_items

        # 构建邻接矩阵
        rows = []
        cols = []
        vals = []

        for u, i, r in interactions:
            # 用户->物品
            rows.append(u)
            cols.append(num_users + i)
            vals.append(r)
            # 物品->用户 (对称)
            rows.append(num_users + i)
            cols.append(u)
            vals.append(r)

        # 归一化
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)

        # 计算度矩阵
        deg = torch.zeros(num_users + num_items)
        for r, c in zip(rows, cols):
            deg[r] += 1

        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

        # 归一化值
        norm_vals = []
        for r, c, v in zip(rows, cols, vals):
            norm_val = v * deg_inv_sqrt[r].item() * deg_inv_sqrt[c].item()
            norm_vals.append(norm_val)

        norm_values = torch.tensor(norm_vals, dtype=torch.float32)

        # 创建稀疏矩阵
        adj = torch.sparse_coo_tensor(
            indices, norm_values,
            (num_users + num_items, num_users + num_items)
        ).to(self.device)

        self.lightgcn.set_norm_adj_matrix(adj)
        self.interaction_matrix = adj

    def update_user_sequence(self, user_id: int, item_id: int):
        """更新用户行为序列"""
        if user_id not in self.user_sequences:
            self.user_sequences[user_id] = []
        self.user_sequences[user_id].append(item_id)
        if len(self.user_sequences[user_id]) > self.config.sasrec_maxlen:
            self.user_sequences[user_id] = self.user_sequences[user_id][-self.config.sasrec_maxlen:]

    def set_song_metadata(self, metadata: Dict[int, Dict]):
        """设置歌曲元数据"""
        self.song_metadata = metadata

    def set_user_profiles(self, profiles: Dict[int, Dict]):
        """设置用户画像"""
        self.user_profiles = profiles

    def recall(self, user_id: int, top_k: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN 召回阶段
        Returns: (candidate_item_ids, candidate_scores)
        """
        self.lightgcn.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], device=self.device)
            scores = self.lightgcn.get_rec_scores(user_tensor)  # (1, num_items)
            scores = scores.squeeze(0)  # (num_items,)

            # 排除已听过的歌曲
            listened = set(self.user_sequences.get(user_id, []))
            mask = torch.ones(self.config.num_items, dtype=torch.bool, device=self.device)
            for item in listened:
                if 0 <= item < self.config.num_items:
                    mask[item] = False

            masked_scores = scores.clone()
            masked_scores[~mask] = float('-inf')

            # 取top-k
            top_scores, top_indices = torch.topk(masked_scores, min(top_k, mask.sum().item()))

        return top_indices, top_scores

    def rank(self, user_id: int, candidate_items: torch.Tensor) -> torch.Tensor:
        """
        SASRec 精排阶段
        Returns: 精排分数 (num_candidates,)
        """
        if user_id not in self.user_sequences or len(self.user_sequences[user_id]) == 0:
            # 新用户: 返回均匀分数，避免nan
            return torch.zeros(len(candidate_items), device=self.device)

        self.sasrec.eval()
        with torch.no_grad():
            # 构建序列
            seq = self.user_sequences[user_id][-self.config.sasrec_maxlen:]
            seq_tensor = torch.zeros(1, self.config.sasrec_maxlen, dtype=torch.long, device=self.device)
            seq_len = len(seq)
            start_idx = self.config.sasrec_maxlen - seq_len
            seq_tensor[0, start_idx:start_idx+seq_len] = torch.tensor(seq, device=self.device)

            # 候选物品
            candidates = candidate_items.unsqueeze(0)  # (1, num_candidates)

            scores = self.sasrec.predict(seq_tensor, candidates)
            scores = scores.squeeze(0)

            # 防止nan/inf
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)

        return scores

    def apply_multi_feature_scoring(
        self,
        user_id: int,
        candidate_items: torch.Tensor,
        base_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        应用多维度特征加权评分
        返回: (final_scores, explanation_list)
        """
        if user_id not in self.user_profiles:
            return base_scores, []

        user_profile = self.user_profiles[user_id]
        explanations = []
        final_scores = base_scores.clone()

        for idx, item_id in enumerate(candidate_items.tolist()):
            if item_id not in self.song_metadata:
                continue

            song = self.song_metadata[item_id]
            tag_scores = {}
            reasons = []

            # 1. 歌手匹配
            if song['artist'] in user_profile.get('favorite_artists', []):
                tag_scores['artist'] = 1.0
                reasons.append(f"歌手「{song['artist']}」是你常听的歌手")
            else:
                tag_scores['artist'] = 0.0

            # 2. 流派匹配
            if song['genre'] in user_profile.get('favorite_genres', []):
                tag_scores['genre'] = 1.0
                reasons.append(f"流派「{song['genre']}」符合你的喜好")
            else:
                tag_scores['genre'] = 0.3

            # 3. 情绪匹配
            if song['emotion'] in user_profile.get('preferred_emotions', []):
                tag_scores['emotion'] = 1.0
                reasons.append(f"情绪「{song['emotion']}」与你近期偏好一致")
            else:
                tag_scores['emotion'] = 0.2

            # 4. 创作目的匹配
            if song['purpose'] in user_profile.get('preferred_purposes', []):
                tag_scores['purpose'] = 1.0
                reasons.append(f"主题「{song['purpose']}」符合你的口味")
            else:
                tag_scores['purpose'] = 0.2

            # 5. 时间段匹配
            current_hour = user_profile.get('current_hour', 12)
            song_time_slot = song.get('preferred_time_slot', '')
            user_time_slot = self._hour_to_slot(current_hour)
            if song_time_slot == user_time_slot:
                tag_scores['time_slot'] = 1.0
                reasons.append(f"适合「{user_time_slot}」收听")
            else:
                tag_scores['time_slot'] = 0.3

            # 6. 场景匹配
            current_scene = user_profile.get('current_scene', '休息')
            if song.get('suitable_scene') == current_scene:
                tag_scores['scene'] = 1.0
                reasons.append(f"适合「{current_scene}」场景")
            else:
                tag_scores['scene'] = 0.2

            # 7. 收藏状态
            if item_id in user_profile.get('favorited_songs', []):
                tag_scores['is_favorited'] = 1.0
                reasons.append("你已收藏此歌")
            else:
                tag_scores['is_favorited'] = 0.0

            # 8. 听歌次数
            play_count = user_profile.get('play_counts', {}).get(item_id, 0)
            if play_count > 10:
                tag_scores['play_count'] = 1.0
                reasons.append(f"你已播放{play_count}次")
            elif play_count > 0:
                tag_scores['play_count'] = play_count / 10.0
            else:
                tag_scores['play_count'] = 0.0

            # 9. 长时间未听
            last_play = user_profile.get('last_play_time', {}).get(item_id, None)
            if last_play is not None:
                days_since = 35  # 模拟值
                if days_since > 30:
                    tag_scores['long_time_no_play'] = 0.8
                    reasons.append("很久没听了，推荐重温")
                else:
                    tag_scores['long_time_no_play'] = 0.0
            else:
                tag_scores['long_time_no_play'] = 0.5

            # 10. 停留时间
            dwell_time = user_profile.get('dwell_times', {}).get(item_id, 0)
            song_duration = song.get('duration', 180)
            if dwell_time > song_duration * 0.8:
                tag_scores['dwell_time'] = 1.0
                reasons.append("你通常会完整听完这首歌")
            elif dwell_time > 0:
                tag_scores['dwell_time'] = dwell_time / (song_duration * 0.8)
            else:
                tag_scores['dwell_time'] = 0.5

            # 计算加权分数
            feature_score = self.weight_system.compute_final_score(tag_scores)

            # 防止nan
            if math.isnan(feature_score) or math.isinf(feature_score):
                feature_score = 0.0

            # 融合精排分数和特征分数 (50% SASRec + 50% 特征)
            base_val = base_scores[idx].item()
            if math.isnan(base_val) or math.isinf(base_val):
                base_val = 0.0

            final_scores[idx] = 0.5 * base_val + 0.5 * feature_score

            explanations.append({
                'item_id': item_id,
                'song_name': song.get('name', f'歌曲{item_id}'),
                'artist': song.get('artist', '未知'),
                'sasrec_score': round(base_val, 4),
                'feature_score': round(feature_score, 4),
                'final_score': round(final_scores[idx].item(), 4),
                'reasons': reasons,
                'tag_scores': {k: round(v, 2) for k, v in tag_scores.items()}
            })

        return final_scores, explanations

    def _hour_to_slot(self, hour: int) -> str:
        """将小时转换为时间段"""
        if 6 <= hour < 9:
            return '早晨(6-9)'
        elif 9 <= hour < 12:
            return '上午(9-12)'
        elif 12 <= hour < 18:
            return '下午(12-18)'
        else:
            return '晚上(18-24)'

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        return_explanation: bool = True
    ) -> Dict:
        """
        完整推荐流程: 召回 -> 精排 -> 多维度加权 -> 去重 -> 返回
        """
        # Step 1: LightGCN 召回 (200首候选)
        candidate_items, recall_scores = self.recall(user_id, top_k=200)

        # Step 2: SASRec 精排
        rank_scores = self.rank(user_id, candidate_items)

        # Step 3: 多维度特征加权
        final_scores, explanations = self.apply_multi_feature_scoring(
            user_id, candidate_items, rank_scores
        )

        # Step 4: 排序并去重
        sorted_indices = torch.argsort(final_scores, descending=True)
        recommended_items = candidate_items[sorted_indices]
        final_scores_sorted = final_scores[sorted_indices]

        # 去重 (确保没有重复)
        seen = set()
        unique_items = []
        unique_scores = []
        unique_explanations = []

        for i, item_id in enumerate(recommended_items.tolist()):
            if item_id not in seen:
                seen.add(item_id)
                unique_items.append(item_id)

                # 安全获取分数，防止nan
                score_val = final_scores_sorted[i].item()
                if math.isnan(score_val) or math.isinf(score_val):
                    score_val = 0.0
                unique_scores.append(score_val)

                if explanations:
                    for exp in explanations:
                        if exp['item_id'] == item_id:
                            unique_explanations.append(exp)
                            break

            if len(unique_items) >= top_n:
                break

        # 构建返回结果
        result = {
            'user_id': user_id,
            'recommendations': []
        }

        for i, item_id in enumerate(unique_items):
            song = self.song_metadata.get(item_id, {})
            rec = {
                'rank': i + 1,
                'item_id': item_id,
                'song_name': song.get('name', f'歌曲{item_id}'),
                'artist': song.get('artist', '未知歌手'),
                'genre': song.get('genre', '未知'),
                'emotion': song.get('emotion', '未知'),
                'purpose': song.get('purpose', '未知'),
                'score': round(unique_scores[i], 4),
            }
            if return_explanation and unique_explanations:
                exp = unique_explanations[i]
                rec['explanation'] = {
                    'sasrec_score': exp['sasrec_score'],
                    'feature_score': exp['feature_score'],
                    'final_score': exp['final_score'],
                    'reasons': exp['reasons'],
                    'active_tags': {k: v for k, v in exp['tag_scores'].items() if v > 0}
                }
            result['recommendations'].append(rec)

        return result

    def update_weights_from_feedback(self, user_id: int, feedback: Dict):
        """
        根据用户反馈更新权重
        feedback: {
            'favorited': [item_id, ...],
            'played': [item_id, ...],
            'skipped': [item_id, ...]
        }
        """
        behavior_changes = {}

        if feedback.get('favorited'):
            behavior_changes['is_favorited'] = {
                'type': 'state_change',
                'description': f'用户收藏了{len(feedback["favorited"])}首歌'
            }

        if feedback.get('played'):
            behavior_changes['play_count'] = {
                'type': 'sharp_increase',
                'description': f'短时间内播放了{len(feedback["played"])}首歌'
            }

        updates = self.weight_system.update_weights_by_behavior(behavior_changes)
        return updates