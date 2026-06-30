"""
main.py
智能音乐推荐系统 - FastAPI后端
包含: 数据生成、模型训练、推荐API、权重管理
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from model import (
    ModelConfig, MusicRecommendationEngine, MusicFeatureDim,
    LightGCN, SASRec, DynamicWeightSystem
)


# ==================== 全局配置 ====================

app = FastAPI(title="智能音乐推荐系统", version="1.0.0")

# 模型配置
CONFIG = ModelConfig(
    num_users=50,
    num_items=2000,
    embedding_dim=64,
    lightgcn_layers=3,
    sasrec_maxlen=50,
    sasrec_num_blocks=2,
    sasrec_num_heads=2,
    sasrec_dropout=0.2,
    device='cpu'
)

# 全局引擎实例
engine: Optional[MusicRecommendationEngine] = None

# 数据存储
USERS = []
SONGS = []
INTERACTIONS = []  # [(user_id, item_id, rating), ...]


# ==================== 真实歌曲数据生成 ====================

def generate_realistic_songs(num_songs: int = 2000) -> List[Dict]:
    """
    生成2000首具有真实感的歌曲数据
    包含: 真实歌手名、歌曲名、多维度特征
    """

    # 真实歌手库 (华语、欧美、日韩等)
    artists_pool = {
        '华语': [
            '周杰伦', '林俊杰', '邓紫棋', '薛之谦', '陈奕迅', '李荣浩', '毛不易',
            '华晨宇', '周深', '张杰', '蔡依林', '王心凌', '五月天', '苏打绿',
            '告五人', '房东的猫', '陈粒', '赵雷', '许嵩', '徐佳莹', '张惠妹',
            '林宥嘉', '田馥甄', '杨丞琳', '萧敬腾', '王力宏', '陶喆', '方大同',
            '蔡健雅', '孙燕姿', '梁静茹', '刘若英', '朴树', '李健', '老狼',
            '新裤子', '痛仰乐队', '万能青年旅店', '草东没有派对', 'deca joins',
            '落日飞车', '椅子乐团', '八三夭', '麋先生', '理想混蛋', '康士坦的变化球'
        ],
        '欧美': [
            'Taylor Swift', 'Ed Sheeran', 'Adele', 'Bruno Mars', 'The Weeknd',
            'Billie Eilish', 'Dua Lipa', 'Harry Styles', 'Justin Bieber', 'Drake',
            'Kanye West', 'Kendrick Lamar', 'Eminem', 'Jay-Z', 'Rihanna', 'Beyoncé',
            'Lady Gaga', 'Katy Perry', 'Ariana Grande', 'Selena Gomez', 'Shawn Mendes',
            'Sam Smith', 'John Legend', 'Charlie Puth', 'Maroon 5', 'Imagine Dragons',
            'Coldplay', 'One Direction', 'Backstreet Boys', 'Linkin Park', 'Green Day',
            'Metallica', 'Queen', 'The Beatles', 'Pink Floyd', 'Nirvana', 'Oasis',
            'Radiohead', 'Arctic Monkeys', 'The 1975', 'Lana Del Rey', 'Sia',
            'Halsey', 'Doja Cat', 'Olivia Rodrigo', 'Sabrina Carpenter', 'Troye Sivan'
        ],
        '日韩': [
            '宇多田光', '椎名林檎', '米津玄師', 'YOASOBI', 'Official髭男dism',
            'King Gnu', '藤井風', 'Aimer', 'LiSA', '米蕾', 'RADWIMPS', 'ONE OK ROCK',
            'BTS', 'BLACKPINK', 'TWICE', 'IU', 'Zico', 'Crush', 'DEAN', 'Heize',
            'NewJeans', 'IVE', 'aespa', 'LE SSERAFIM', 'SEVENTEEN', 'Stray Kids',
            'NCT', 'EXO', 'Red Velvet', 'MAMAMOO', 'ITZY', 'STAYC', 'TREASURE'
        ]
    }

    # 歌曲名生成模板
    song_templates = [
        "{}的{}", "关于{}", "{}之后", "致{}", "像{}一样", "{}日记",
        "{}物语", "{}之歌", "那年{}", "{}回忆", "{}幻想", "{}漫步",
        "{}星空", "{}黎明", "{}黄昏", "{}雨季", "{}晴天", "{}微光",
        "{}远方", "{}归途", "{}彼岸", "{}序曲", "{}终章", "{}独白",
        "{}信笺", "{}画框", "{}镜中", "{}深海", "{}云端", "{}烟火",
        "{}轨迹", "{}频率", "{}波长", "{}磁场", "{}引力", "{}共振"
    ]

    # 主题词
    themes = [
        '夏天', '冬天', '雨天', '星空', '海洋', '城市', '故乡', '旅途',
        '梦想', '青春', '爱情', '离别', '重逢', '孤独', '自由', '勇气',
        '时光', '记忆', '未来', '过去', '现在', '永恒', '瞬间', '远方',
        '黎明', '黄昏', '午夜', '清晨', '花开', '落叶', '飘雪', '暖阳',
        '微风', '细雨', '彩虹', '流星', '烟火', '灯塔', '港湾', '彼岸',
        '旋律', '节奏', '和弦', '音符', '乐章', '序曲', '终章', '间奏'
    ]

    songs = []
    all_artists = []
    for region, artists in artists_pool.items():
        all_artists.extend([(a, region) for a in artists])

    for i in range(num_songs):
        # 随机选择歌手
        artist, region = random.choice(all_artists)

        # 生成歌曲名
        template = random.choice(song_templates)
        theme = random.choice(themes)
        if '{}' in template:
            if template.count('{}') == 2:
                theme2 = random.choice(themes)
                name = template.format(theme, theme2)
            else:
                name = template.format(theme)
        else:
            name = f"{theme}{template}"

        # 多维度特征
        genre = random.choice(MusicFeatureDim.GENRES)
        emotion = random.choice(MusicFeatureDim.EMOTIONS)
        purpose = random.choice(MusicFeatureDim.PURPOSES)
        scene = random.choice(MusicFeatureDim.SCENES)
        time_slot = random.choice(MusicFeatureDim.TIME_SLOTS)

        # 时长 (2-5分钟)
        duration = random.randint(120, 300)

        # 发行年份 (1990-2024)
        year = random.randint(1990, 2024)

        # 热度 (0-100)
        popularity = random.randint(20, 98)

        song = {
            'id': i,
            'name': name,
            'artist': artist,
            'region': region,
            'genre': genre,
            'emotion': emotion,
            'purpose': purpose,
            'suitable_scene': scene,
            'preferred_time_slot': time_slot,
            'duration': duration,
            'year': year,
            'popularity': popularity
        }
        songs.append(song)

    return songs


def generate_users(num_users: int = 50) -> List[Dict]:
    """生成50个用户及其画像"""

    user_names = [
        '小明', '小红', '阿杰', '小雨', '子涵', '浩然', '欣怡', '伟豪',
        '思琪', '俊凯', '梦瑶', '天宇', '佳慧', '志强', '晓雯', '鹏飞',
        '雪梅', '建国', '丽华', '勇军', '敏娜', '海涛', '晶晶', '磊磊',
        '婷婷', '超超', '莉莉', '军军', '燕燕', '波波', '玲玲', '强强',
        '芳芳', '亮亮', '媛媛', '涛涛', '秀秀', '彬彬', '娜娜', '阳阳',
        '静静', '龙龙', '敏敏', '飞飞', '艳艳', '鹏鹏', '琳琳', '东东',
        '慧慧', '川川'
    ]

    scenes = MusicFeatureDim.SCENES
    time_slots = MusicFeatureDim.TIME_SLOTS

    users = []
    for i in range(num_users):
        # 每个用户有偏好特征
        favorite_genres = random.sample(MusicFeatureDim.GENRES, k=random.randint(2, 4))
        preferred_emotions = random.sample(MusicFeatureDim.EMOTIONS, k=random.randint(2, 3))
        preferred_purposes = random.sample(MusicFeatureDim.PURPOSES, k=random.randint(2, 4))
        favorite_artists = random.sample([s['artist'] for s in SONGS], k=random.randint(3, 8))

        # 常听时间段 (1-2个)
        preferred_time_slots = random.sample(time_slots, k=random.randint(1, 2))

        # 常听场景
        preferred_scenes = random.sample(scenes, k=random.randint(2, 4))

        # 当前场景 (模拟)
        current_scene = random.choice(scenes)
        current_hour = random.randint(6, 23)

        user = {
            'id': i,
            'name': user_names[i],
            'favorite_genres': favorite_genres,
            'preferred_emotions': preferred_emotions,
            'preferred_purposes': preferred_purposes,
            'favorite_artists': favorite_artists,
            'preferred_time_slots': preferred_time_slots,
            'preferred_scenes': preferred_scenes,
            'current_scene': current_scene,
            'current_hour': current_hour,
            'play_counts': {},  # {song_id: count}
            'favorited_songs': set(),
            'last_play_time': {},  # {song_id: timestamp}
            'dwell_times': {},  # {song_id: seconds}
            'listening_history': []  # [(song_id, timestamp, duration), ...]
        }
        users.append(user)

    return users


def generate_interactions(users: List[Dict], songs: List[Dict]) -> List[Tuple[int, int, float]]:
    """
    生成用户-歌曲交互数据
    模拟真实听歌行为，考虑用户偏好
    """
    interactions = []

    for user in users:
        user_id = user['id']

        # 每个用户听过的歌曲数量 (50-400首)
        num_listened = random.randint(50, 400)

        # 根据用户偏好选择歌曲
        candidate_scores = []
        for song in songs:
            score = 0.0

            # 流派匹配
            if song['genre'] in user['favorite_genres']:
                score += 3.0

            # 情绪匹配
            if song['emotion'] in user['preferred_emotions']:
                score += 2.0

            # 目的匹配
            if song['purpose'] in user['preferred_purposes']:
                score += 2.0

            # 歌手匹配
            if song['artist'] in user['favorite_artists']:
                score += 4.0

            # 场景匹配
            if song['suitable_scene'] in user['preferred_scenes']:
                score += 1.5

            # 时间段匹配
            if song['preferred_time_slot'] in user['preferred_time_slots']:
                score += 1.0

            # 热度加成
            score += song['popularity'] / 100.0

            # 随机噪声
            score += random.gauss(0, 0.5)

            candidate_scores.append((song['id'], max(score, 0.1)))

        # 按分数加权采样
        total_score = sum(s for _, s in candidate_scores)
        probs = [s / total_score for _, s in candidate_scores]

        # 关键修复: 将 numpy.int64 转为 Python int
        listened_songs = np.random.choice(
            [sid for sid, _ in candidate_scores],
            size=min(num_listened, len(songs)),
            replace=False,
            p=probs
        ).tolist()

        # 为每首歌生成交互细节
        base_time = datetime(2024, 1, 1)

        for song_id in listened_songs:
            song = songs[song_id]

            # 播放次数 (1-50次)
            play_count = int(max(1, np.random.exponential(8)))
            play_count = min(play_count, 50)
            user['play_counts'][song_id] = play_count

            # 是否收藏 (概率与播放次数相关)
            is_favorited = random.random() < (play_count / 50.0 * 0.6)
            if is_favorited:
                user['favorited_songs'].add(song_id)

            # 停留时间 (秒)
            full_duration = song['duration']
            if random.random() < 0.7:  # 70%概率完整听完
                dwell_time = full_duration
            else:
                dwell_time = random.randint(30, full_duration)
            user['dwell_times'][song_id] = dwell_time

            # 上次播放时间
            days_ago = random.randint(0, 90)
            last_play = base_time + timedelta(days=days_ago)
            user['last_play_time'][song_id] = last_play.isoformat()

            # 生成多次交互记录
            for _ in range(play_count):
                # 交互强度 (0.5-5.0)
                rating = min(5.0, 1.0 + play_count * 0.1 + (2.0 if is_favorited else 0))
                interactions.append((user_id, song_id, rating))

            # 记录听歌历史
            user['listening_history'].append({
                'song_id': song_id,
                'play_count': play_count,
                'is_favorited': is_favorited,
                'dwell_time': dwell_time,
                'last_play': user['last_play_time'][song_id]
            })

    return interactions


# ==================== 模型训练 ====================

def train_models():
    """训练LightGCN和SASRec模型"""
    global engine

    print("正在初始化推荐引擎...")
    engine = MusicRecommendationEngine(CONFIG)

    # 设置歌曲元数据
    song_metadata = {s['id']: s for s in SONGS}
    engine.set_song_metadata(song_metadata)

    # 设置用户画像
    user_profiles = {}
    for u in USERS:
        user_profiles[u['id']] = {
            'favorite_genres': u['favorite_genres'],
            'preferred_emotions': u['preferred_emotions'],
            'preferred_purposes': u['preferred_purposes'],
            'favorite_artists': u['favorite_artists'],
            'preferred_time_slots': u['preferred_time_slots'],
            'preferred_scenes': u['preferred_scenes'],
            'current_scene': u['current_scene'],
            'current_hour': u['current_hour'],
            'play_counts': u['play_counts'],
            'favorited_songs': u['favorited_songs'],
            'last_play_time': u['last_play_time'],
            'dwell_times': u['dwell_times']
        }
    engine.set_user_profiles(user_profiles)

    # 构建交互图
    print("构建用户-物品交互图...")
    engine.build_interaction_graph(INTERACTIONS)

    # 更新用户序列
    print("更新用户行为序列...")
    for u in USERS:
        # 按时间排序的历史
        history = sorted(u['listening_history'], key=lambda x: x['last_play'])
        for h in history:
            for _ in range(h['play_count']):
                engine.update_user_sequence(u['id'], h['song_id'])

    # 训练LightGCN (简化版: 使用BPR Loss)
    print("训练 LightGCN 召回模型...")
    optimizer_gcn = torch.optim.Adam(engine.lightgcn.parameters(), lr=0.001)

    for epoch in range(100):
        engine.lightgcn.train()
        optimizer_gcn.zero_grad()

        user_emb, item_emb = engine.lightgcn.forward()

        # 采样BPR训练
        loss = 0.0
        num_samples = 1000

        for _ in range(num_samples):
            # 随机采样正样本
            u, i, _ = random.choice(INTERACTIONS)
            # 采样负样本
            j = random.randint(0, CONFIG.num_items - 1)
            while (u, j) in [(x[0], x[1]) for x in INTERACTIONS]:
                j = random.randint(0, CONFIG.num_items - 1)

            pos_score = torch.dot(user_emb[u], item_emb[i])
            neg_score = torch.dot(user_emb[u], item_emb[j])

            bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10)
            loss += bpr_loss

        loss = loss / num_samples

        # L2正则化
        reg_loss = 0.0001 * (user_emb.norm(2).pow(2) + item_emb.norm(2).pow(2))
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer_gcn.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/100, Loss: {total_loss.item():.4f}")

    # 训练SASRec (简化版)
    print("训练 SASRec 精排模型...")
    optimizer_sas = torch.optim.Adam(engine.sasrec.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        engine.sasrec.train()
        total_loss = 0.0
        num_batches = 200

        for _ in range(num_batches):
            # 采样用户序列
            user_id = random.randint(0, CONFIG.num_users - 1)
            seq = engine.user_sequences.get(user_id, [])

            if len(seq) < 2:
                continue

            # 构建训练样本: 预测下一个物品
            seq_input = seq[:-1]
            target = seq[-1]

            # 填充序列
            seq_tensor = torch.zeros(1, CONFIG.sasrec_maxlen, dtype=torch.long)
            seq_len = min(len(seq_input), CONFIG.sasrec_maxlen)
            start = CONFIG.sasrec_maxlen - seq_len
            seq_tensor[0, start:start+seq_len] = torch.tensor(seq_input[-seq_len:])

            target_tensor = torch.tensor([target], dtype=torch.long)

            optimizer_sas.zero_grad()
            logits = engine.sasrec(seq_tensor)
            loss = criterion(logits, target_tensor)

            loss.backward()
            optimizer_sas.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50, Loss: {total_loss/num_batches:.4f}")

    print("模型训练完成!")
    return engine


# ==================== API 模型 ====================

class RecommendRequest(BaseModel):
    user_id: int
    top_n: int = 10


class FeedbackRequest(BaseModel):
    user_id: int
    favorited: List[int] = []
    played: List[int] = []
    skipped: List[int] = []


class SceneRequest(BaseModel):
    user_id: int
    scene: str


# ==================== API 路由 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据和模型"""
    global SONGS, USERS, INTERACTIONS, engine

    print("=" * 50)
    print("启动智能音乐推荐系统...")
    print("=" * 50)

    print("生成歌曲数据...")
    SONGS = generate_realistic_songs(2000)
    print(f"   已生成 {len(SONGS)} 首歌曲")

    print("生成用户数据...")
    USERS = generate_users(50)
    print(f"   已生成 {len(USERS)} 个用户")

    print("生成交互数据...")
    INTERACTIONS = generate_interactions(USERS, SONGS)
    print(f"   已生成 {len(INTERACTIONS)} 条交互记录")

    # 训练模型
    engine = train_models()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回前端页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/songs")
async def get_songs(limit: int = Query(50, ge=1, le=200)):
    """获取歌曲列表"""
    return {
        "total": len(SONGS),
        "songs": SONGS[:limit]
    }


@app.get("/api/users")
async def get_users():
    """获取用户列表"""
    return {
        "total": len(USERS),
        "users": [{"id": u['id'], "name": u['name']} for u in USERS]
    }


@app.get("/api/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """获取用户画像"""
    if user_id < 0 or user_id >= len(USERS):
        raise HTTPException(status_code=404, detail="用户不存在")
    user = USERS[user_id]
    return {
        "id": user['id'],
        "name": user['name'],
        "favorite_genres": user['favorite_genres'],
        "preferred_emotions": user['preferred_emotions'],
        "preferred_purposes": user['preferred_purposes'],
        "favorite_artists": user['favorite_artists'][:5],
        "preferred_time_slots": user['preferred_time_slots'],
        "preferred_scenes": user['preferred_scenes'],
        "current_scene": user['current_scene'],
        "current_hour": user['current_hour'],
        "total_played": len(user['play_counts']),
        "total_favorited": len(user['favorited_songs'])
    }


@app.get("/api/users/{user_id}/history")
async def get_user_history(user_id: int, limit: int = Query(20, ge=1, le=100)):
    """获取用户听歌历史"""
    if user_id < 0 or user_id >= len(USERS):
        raise HTTPException(status_code=404, detail="用户不存在")

    user = USERS[user_id]
    history = []

    for song_id, count in sorted(user['play_counts'].items(), key=lambda x: -x[1])[:limit]:
        song = SONGS[song_id]
        history.append({
            "song_id": song_id,
            "song_name": song['name'],
            "artist": song['artist'],
            "play_count": count,
            "is_favorited": song_id in user['favorited_songs'],
            "dwell_time": user['dwell_times'].get(song_id, 0),
            "last_play": user['last_play_time'].get(song_id, '')
        })

    return {
        "user_id": user_id,
        "user_name": user['name'],
        "history": history
    }


@app.post("/api/recommend")
async def get_recommendations(request: RecommendRequest):
    """
    获取推荐结果
    完整流程: LightGCN召回 -> SASRec精排 -> 多维度加权 -> 去重
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")

    if request.user_id < 0 or request.user_id >= CONFIG.num_users:
        raise HTTPException(status_code=404, detail="用户不存在")

    result = engine.recommend(request.user_id, top_n=request.top_n)

    # 添加用户当前状态
    user = USERS[request.user_id]
    result['user_context'] = {
        'current_scene': user['current_scene'],
        'current_hour': user['current_hour'],
        'scene_description': _get_scene_description(user['current_scene'])
    }

    # 添加当前权重
    result['current_weights'] = engine.weight_system.get_all_weights()

    return result


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """提交用户反馈，更新权重"""
    if engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")

    updates = engine.update_weights_from_feedback(
        request.user_id,
        {
            'favorited': request.favorited,
            'played': request.played,
            'skipped': request.skipped
        }
    )

    return {
        "user_id": request.user_id,
        "weight_updates": updates,
        "current_weights": engine.weight_system.get_all_weights()
    }


@app.post("/api/users/{user_id}/scene")
async def update_scene(user_id: int, request: SceneRequest):
    """更新用户当前场景"""
    if user_id < 0 or user_id >= len(USERS):
        raise HTTPException(status_code=404, detail="用户不存在")

    USERS[user_id]['current_scene'] = request.scene
    USERS[user_id]['current_hour'] = random.randint(6, 23)  # 模拟时间变化

    # 更新引擎中的用户画像
    engine.user_profiles[user_id]['current_scene'] = request.scene
    engine.user_profiles[user_id]['current_hour'] = USERS[user_id]['current_hour']

    return {
        "user_id": user_id,
        "new_scene": request.scene,
        "current_hour": USERS[user_id]['current_hour'],
        "message": "场景已更新，推荐结果将相应调整"
    }


@app.get("/api/weights")
async def get_weights():
    """获取当前所有权重"""
    if engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")
    return engine.weight_system.get_all_weights()


@app.get("/api/weights/history")
async def get_weight_history():
    """获取权重调整历史"""
    if engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")
    return dict(engine.weight_system.weight_history)


def _get_scene_description(scene: str) -> str:
    """获取场景描述"""
    descriptions = {
        '运动': '适合节奏感强、鼓点明显的歌曲',
        '学习': '适合轻柔、不分散注意力的背景音乐',
        '通勤': '适合提神醒脑、节奏明快的歌曲',
        '休息': '适合舒缓放松、旋律优美的歌曲',
        '聚会': '适合热闹欢快、氛围感强的歌曲',
        '睡前': '适合安静温柔、助眠的歌曲'
    }
    return descriptions.get(scene, '通用场景')


# ==================== 主入口 ====================

if __name__ == "__main__":
    # 确保static目录存在
    os.makedirs("static", exist_ok=True)

    print("=" * 50)
    print("智能音乐推荐系统启动中...")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)