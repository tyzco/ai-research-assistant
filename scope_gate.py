"""学术范围白名单闸门 — 锁定用户只能问论文/学术相关问题。

目的: 防止用户把工具当免费通用 LLM 使用 (问编程/美食/股票/娱乐/违法犯罪等)。
白名单 = 只放行「论文/学术研究」意图, 其余拒绝。避免不法分子借用算力做无关/恶意的事。

三层漏斗 (复用项目意图识别哲学):
  L0 会话短句: 多轮续接的「继续/好的/补充」等短句放行 (仅拦硬非学术词)
  L1 规则:     强学术关键词 → 放行; 硬非学术关键词 → 拒绝 (免费, 无嵌入)
  L2 语义:     嵌入 margin 判定 (学术原型 vs 非学术原型), 兜底规则未覆盖的情况
  失败降级:    嵌入不可用 → 仅规则闸 (防服务不可用)

gate_query(query) -> (allowed: bool, reason: str, level: str)
level ∈ {empty, short_pass, rule_pass, rule_reject, semantic_pass, semantic_reject, fallback_pass}
"""

import logging
import threading

logger = logging.getLogger(__name__)

# ---- 阈值 ----
MARGIN = 0.06  # 非学术得分必须比学术高这么多才拒绝 (容忍嵌入噪声)
ACAD_FLOOR = 0.42  # 学术得分下限: 低于且不占优 → 视为弱学术, 拒绝

# ---- L1 规则: 强学术关键词 (命中 → 直接放行) ----
# 顺序重要: 学术先查, 硬非学术后查, 保证「论文里实验代码」这类合法请求先被学术词救回
ACADEMIC_KEYWORDS = (
    "论文",
    "文献",
    "综述",
    "调研",
    "研究",
    "学术",
    "期刊",
    "会议",
    "投稿",
    "审稿",
    "引用",
    "参考文献",
    "摘要",
    "选题",
    "课题",
    "大纲",
    "写作",
    "实验",
    "消融",
    "数据集",
    "评估",
    "指标",
    "假设",
    "结论",
    "相关工作",
    "研究空白",
    "知识库",
    "检索",
    "检索策略",
    "关键词",
    "查重",
    "降重",
    "引文",
    "被引",
    "引用链",
    "脉络",
    "排版",
    "图表",
    "方法",
    "方向",
    "题目",
    "paper",
    "research",
    "literature",
    "survey",
    "journal",
    "conference",
    "manuscript",
    "thesis",
    "dissertation",
    "abstract",
    "citation",
    "reference",
    "bibliography",
    "experiment",
    "methodology",
    "related work",
    "research gap",
    "dataset",
    "evaluation",
    "hypothesis",
    "peer review",
    "machine learning",
    "deep learning",
    "neural network",
    "artificial intelligence",
    "natural language processing",
    "computer vision",
    "algorithm",
    "state of the art",
    "arxiv",
    "doi",
    "scholar",
    # AI/ML 领域术语 (本工具的学术主战场, 命中即视为学术意图)
    "注意力",
    "神经网络",
    "深度学习",
    "机器学习",
    "强化学习",
    "大模型",
    "大语言模型",
    "预训练",
    "多模态",
    "知识图谱",
    "生成模型",
    "扩散模型",
    "图神经网络",
    "损失函数",
    "梯度",
    "特征",
    "embedding",
    "llm",
    "bert",
    "gpt",
    "transformer",
    "attention",
    "智能体",
    "推荐系统",
    "语音识别",
    "图像分类",
    "目标检测",
    "语义分割",
    "自然语言",
    "计算机视觉",
    "数据挖掘",
    "聚类",
    "统计",
    "概率",
    "泛化",
    "过拟合",
    "benchmark",
    "评测",
    "消融",
    "参数量",
    "自编码",
    "编码器",
    "解码器",
    "卷积",
    "池化",
    "激活函数",
    "反向传播",
    "归一化",
    "dropout",
    "学习率",
    "优化器",
    "微调",
    "prompt",
    "提示词",
    "生成对抗",
    "vae",
    "gan",
    "cnn",
    "rnn",
    "lstm",
    "gru",
    "贝叶斯",
    "马尔可夫",
    "蒙特卡洛",
    "策略梯度",
    "采样",
    "隐变量",
    "强化",
)

# ---- L1 规则: 硬非学术关键词 (命中 → 直接拒绝) ----
# 只收录「无歧义的非学术」词: 学术上下文几乎不会出现 (含这些词的合法学术请求已被学术词救回)
NON_ACADEMIC_KEYWORDS = (
    "菜谱",
    "做饭",
    "做菜",
    "写代码",
    "编程",
    "代码实现",
    "股票",
    "基金",
    "理财",
    "彩票",
    "赌博",
    "赌",
    "色情",
    "追剧",
    "综艺",
    "游戏攻略",
    "星座",
    "算命",
    "天气",
    "明星",
    "八卦",
    "美食推荐",
    "探店",
    "减肥",
    "穿搭",
    "化妆",
    "装修",
    "买车",
    "买房",
    "找工作",
    "面试",
    "约会",
    "相亲",
    "钓鱼",
    "健身",
    "运动打卡",
    "怎么说",
    "怎么翻译",
    "翻译一下",
    "赚钱",
    "兼职",
    "黑客",
    "入侵",
    "破解",
    "木马",
    "钓鱼网站",
    "诈骗",
    "洗钱",
    "毒品",
    "枪支",
    "爆炸",
    "简历",
    "写简历",
    "爬虫",
    "抓取数据",
    "刷流量",
    "薅羊毛",
    "外挂",
    "脚本抢购",
    "ddos",
    "翻墙",
)

# ---- L0 会话续接短语 (多轮对话的「继续/好的/补充」等短句, 精确匹配) ----
# 短查询只有恰好等于续接短语才放行; 否则走完整闸门 (规则+语义), 防「怎么做红烧肉」这类短句漏网
CONTINUATION_PHRASES = {
    "继续",
    "接着",
    "然后",
    "好的",
    "好",
    "嗯",
    "可以",
    "是的",
    "对",
    "谢谢",
    "补充",
    "详细",
    "展开",
    "具体",
    "具体点",
    "修改",
    "改一下",
    "调整",
    "换成",
    "为什么",
    "如何",
    "怎么",
    "这个",
    "那个",
    "上面的",
    "上一条",
    "重新",
    "用",
    "跑",
    "帮我写",
    "ok",
    "okay",
    "yes",
    "continue",
    "next",
    "more",
    "again",
}

# ---- L2 语义原型 (嵌入中心点) ----
_ACADEMIC_PROTOTYPES = [
    "文献综述与调研",
    "论文写作与学术写作",
    "研究方向与选题分析",
    "参考文献与引用格式",
    "实验设计与消融实验",
    "论文审稿与同行评审",
    "机器学习与深度学习算法原理",
    "科研方法与数据分析",
    "期刊投稿与会议论文",
    "学术概念解释与问答",
    "literature review and research survey",
    "academic paper writing",
    "research gap and topic selection",
    "citation and reference formatting",
    "experimental design and ablation study",
    "peer review of academic papers",
    "machine learning and deep learning concepts",
    "research methodology and data analysis",
    "journal submission and conference papers",
    "explain academic concepts",
    "自编码器与生成模型",
    "autoencoder and generative adversarial network",
    "神经网络结构与反向传播",
    "neural network architecture and backpropagation",
    "大语言模型与提示词工程",
    "large language model and prompt engineering",
    "强化学习与策略优化",
    "reinforcement learning and policy optimization",
    "transformer与注意力机制",
    "transformer and attention mechanism",
]

_NON_ACADEMIC_PROTOTYPES = [
    "写代码和编程",
    "做菜和食谱",
    "天气预报",
    "新闻与时事",
    "娱乐八卦与明星",
    "游戏攻略",
    "股票与理财",
    "情感与恋爱咨询",
    "旅游攻略与景点推荐",
    "健身与减肥",
    "网络购物推荐",
    "星座运势与算命",
    "python programming code",
    "cooking recipe",
    "stock trading and investment",
    "sports news",
    "movie and tv recommendation",
    "travel guide",
]

# 原型嵌入缓存 (只算一次, 线程安全由初始化加锁保证)
_proto_cache: dict[str, list[list[float]]] = {}
_proto_cache_lock = threading.Lock()


def _proto_emb(key: str) -> list[list[float]]:
    global _proto_cache
    if key not in _proto_cache:
        from embedder import embed_texts

        with _proto_cache_lock:
            if key not in _proto_cache:
                _proto_cache[key] = embed_texts(
                    _ACADEMIC_PROTOTYPES if key == "acad" else _NON_ACADEMIC_PROTOTYPES
                )
    return _proto_cache[key]


def _max_sim(query_emb: list[float], proto_embs: list[list[float]]) -> float:
    import numpy as np

    q = np.asarray(query_emb, dtype=np.float64)
    m = np.asarray(proto_embs, dtype=np.float64)
    return float((m @ q).max())


_REASON = "本工具仅支持论文/学术相关问题。如需查文献、写论文、做研究分析等请描述您的学术需求。"


def gate_query(query: str, topic_mode: bool = False) -> tuple[bool, str, str]:
    """学术范围白名单闸门。返回 (allowed, reason, level)。

    topic_mode=True (选题判断模式): 输入是「裸选题短语」(如「情感分析」「图神经网络」),
    本来就默认是学术选题, 只拦硬非学术词, 不做「必须含学术关键词」+ 语义层
    (否则「情感分析」会被语义层误判为情感咨询而误拒)。
    """
    q = (query or "").strip()
    if not q:
        return False, "请输入问题", "empty"
    ql = q.lower()

    # 选题判断模式: 只拦硬非学术词, 其余一律放行 (裸选题天然是学术意图)
    if topic_mode:
        if any(k in ql for k in NON_ACADEMIC_KEYWORDS):
            return False, _REASON, "rule_reject"
        return True, "", "topic_pass"

    # L0: 短查询仅放行「会话续接短语」的精确匹配 (多轮对话的 继续/好的/补充 等);
    # 其余短查询走完整闸门, 防「怎么做红烧肉」这类短句绕过
    if len(q) <= 8 and ql.strip() in CONTINUATION_PHRASES:
        return True, "", "short_pass"

    # L1 规则: 强学术词 → 放行 (先查, 防止合法学术请求被非学术词误杀)
    if any(k in ql for k in ACADEMIC_KEYWORDS):
        return True, "", "rule_pass"

    # L1 规则: 硬非学术词 → 拒绝
    if any(k in ql for k in NON_ACADEMIC_KEYWORDS):
        return False, _REASON, "rule_reject"

    # L2 语义: 嵌入 margin (规则未覆盖的模糊请求)。
    # fail-closed: 只有学术明显占优才放行; 否则拒绝。
    # 裸选题短语 (如「情感分析」) 的误杀由 topic_mode 精准解决 (选题判断端点传 topic_mode=True),
    # 不在这里全局放宽 —— 全局 fail-open 是 control regression。
    try:
        from embedder import embed_query

        qe = embed_query(q)
        acad = _max_sim(qe, _proto_emb("acad"))
        non = _max_sim(qe, _proto_emb("non"))
        if acad - non >= MARGIN:
            return True, "", "semantic_pass"
        if acad >= ACAD_FLOOR and acad >= non:
            return True, "", "semantic_pass"
        # 非学术占优 或 学术太弱 → 拒绝 (白名单: 不明确即不放行)
        return False, _REASON, "semantic_reject"
    except Exception as e:
        # 降级: 嵌入不可用 → 规则闸兜底 (放行, 防服务不可用)
        logger.warning(f"scope_gate semantic layer unavailable: {e}")
        return True, "", "fallback_pass"
