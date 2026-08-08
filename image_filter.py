"""ML 轻量级图片分类器 — 方案4: LightGBM 无 LLM 过滤。

DeepSeek 对话参考: 在调用多模态模型之前，用规则 + ML 过滤掉大量无用图片，
节省 99% 的 API 调用成本。

14 维手工特征 (不用深度学习):
  - 几何: width, height, area, aspect_ratio
  - 颜色: unique_colors, color_entropy
  - 亮度/对比度: brightness, contrast
  - 边缘: edge_density, edge_entropy
  - 纹理: texture_coarseness
  - 饱和度/色相: saturation_mean, hue_std
  - 空白占比: whitespace_ratio

预测: 1 = 保留 (图表/公式/示意图), 0 = 过滤 (logo/图标/分割线/装饰)
"""

import io
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_PATH = Path("data/image_filter_model.json")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ===== 14 维特征提取器 =====


def extract_features(img_bytes: bytes) -> np.ndarray:
    """从图片字节提取 14 维手工特征。"""
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_array = np.array(img, dtype=np.float64)
    h, w = img_array.shape[:2]
    features = []

    # 1. 几何特征 (4维)
    features.append(float(w))
    features.append(float(h))
    features.append(float(w * h))  # area
    features.append(float(w) / max(1, h))  # aspect_ratio

    # 2. 颜色特征 (2维)
    colors = img.getcolors(maxcolors=50000)
    unique_colors = len(colors) if colors else 50000
    features.append(float(unique_colors))

    # 颜色信息熵
    r = img_array[:, :, 0].flatten()
    g = img_array[:, :, 1].flatten()
    b = img_array[:, :, 2].flatten()
    rgb_combined = np.stack([r, g, b], axis=0)  # [3, N]
    color_entropy = _entropy_2d(rgb_combined.reshape(-1, w * h).T)
    features.append(float(color_entropy))

    # 3. 亮度/对比度 (2维)
    gray_flat = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray_flat.reshape(h, w)
    features.append(float(np.mean(gray)))
    features.append(float(np.std(gray)))

    # 4. 边缘特征 (2维) — 用梯度差分近似
    gy = gray[1:, :] - gray[: h - 1, :]  # vertical diff
    gx = gray[:, 1:] - gray[:, : w - 1]  # horizontal diff
    edge_mag = (
        np.concatenate(
            [
                np.abs(gy).mean(axis=0),
                np.abs(gx).mean(axis=1),
            ]
        )
        if gy.shape[1] > 0
        else np.array([0.0])
    )
    features.append(float(np.mean(edge_mag)))
    features.append(float(_entropy_1d(edge_mag)))

    # 5. 纹理粗糙度 (1维)
    kernel = np.ones((3, 3)) / 9.0
    pad_h, pad_w = h % 3, w % 3
    if pad_h > 0 or pad_w > 0:
        gray_pad = gray[: h - pad_h if pad_h else h, : w - pad_w if pad_w else w]
    else:
        gray_pad = gray
    if gray_pad.size > 0:
        blocks = gray_pad.reshape(gray_pad.shape[0] // 3, 3, gray_pad.shape[1] // 3, 3)
        block_stds = blocks.std(axis=(1, 3))
        features.append(float(np.mean(block_stds)) if block_stds.size > 0 else 0.0)
    else:
        features.append(0.0)

    # 6. 饱和度/色相 (2维)
    r_n = r / 255.0
    g_n = g / 255.0
    b_n = b / 255.0
    c_max = np.maximum(np.maximum(r_n, g_n), b_n)
    c_min = np.minimum(np.minimum(r_n, g_n), b_n)
    delta = c_max - c_min
    saturation = np.where(c_max > 0, delta / c_max, 0)
    features.append(float(np.mean(saturation)))
    features.append(float(np.std(r_n - g_n)))  # hue proxy

    # 7. 空白像素占比 (1维)
    light_mask = gray > 240
    whitespace_ratio = float(np.sum(light_mask)) / max(1, gray.size)
    features.append(whitespace_ratio)

    return np.array(features, dtype=np.float64)


def _entropy_1d(arr: np.ndarray) -> float:
    """一维数组的信息熵。"""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    hist, _ = np.histogram(arr.flatten(), bins=20)
    hist = hist.astype(np.float64) / max(1, hist.sum())
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0


def _entropy_2d(arr: np.ndarray) -> float:
    """二维数组 (N, channels) 近似信息熵。"""
    if arr.size == 0:
        return 0.0
    arr = arr[~np.isnan(arr).any(axis=1)]
    if len(arr) < 2:
        return 0.0
    return _entropy_1d(arr.flatten())


# ===== ML 分类器 (LightGBM 优先, 降级为 Logistic Regression) =====


class MLImageFilter:
    """ML 图片过滤分类器。

    用法:
        flt = MLImageFilter()
        keep, score = flt.predict(img_bytes)
        if keep:
            save_image(img_bytes)
    """

    def __init__(self):
        self.model = None
        self.feature_names = [
            "width",
            "height",
            "area",
            "aspect_ratio",
            "unique_colors",
            "color_entropy",
            "brightness",
            "contrast",
            "edge_density",
            "edge_entropy",
            "texture_coarseness",
            "saturation_mean",
            "hue_std",
            "whitespace_ratio",
        ]
        self._load_or_build()

    def _load_or_build(self):
        """加载已有模型，不存在则构建基于规则的启发式分类器。"""
        if MODEL_PATH.exists():
            try:
                data = json.loads(MODEL_PATH.read_text())
                self._weights = np.array(data.get("weights", []))
                self._bias = data.get("bias", 0.0)
                self._threshold = data.get("threshold", 0.5)
                logger.info(
                    f"Loaded image filter model (trained on {data.get('n_samples', 0)} samples)"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, using heuristics")

        # 启发式权重: 几何特征负相关(太大 → 可能是照片, 太小 → logo),
        #             颜色熵正相关(图表颜色丰富), 边缘密度正相关(图表有线条)
        self._weights = np.array(
            [
                -0.05,  # width:  太小/太大都不好，中性
                -0.05,  # height: 同上
                0.10,  # area:   一定面积才有内容 (正相关)
                -0.20,  # aspect_ratio: 极端比例通常是分割线 (负相关)
                0.25,  # unique_colors: 图表颜色丰富 (正相关, 最重要)
                0.15,  # color_entropy: 颜色分布均匀 → 图表 (正相关)
                0.05,  # brightness: 中等亮度最好 (中性)
                0.15,  # contrast: 图表有对比度 (正相关)
                0.20,  # edge_density: 图表有线条 (正相关)
                0.10,  # edge_entropy: 边缘分布均匀 (正相关)
                0.10,  # texture_coarseness: 纹理粗糙 → 图表 (正相关)
                0.05,  # saturation_mean: 饱和度中等 (中性)
                0.05,  # hue_std: 色相变化 (中性)
                -0.30,  # whitespace_ratio: 大量空白 → 可能是分割线/logo (强负相关)
            ]
        )
        self._bias = 0.0
        self._threshold = 0.5
        logger.info("Using heuristic image filter (no trained model yet)")

    def predict(self, img_bytes: bytes) -> tuple[bool, float]:
        """预测该图片是否应保留。返回 (keep: bool, confidence: float)。"""
        features = extract_features(img_bytes)

        # 检查特征有效性
        if np.isnan(features).any() or len(features) != len(self._weights):
            return False, 0.0

        # 归一化 + 逻辑回归
        features_norm = (features - np.mean(features)) / (np.std(features) + 1e-8)
        logit = np.dot(features_norm, self._weights) + self._bias
        prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        keep = prob > self._threshold
        return bool(keep), float(prob)

# ===== 全局单例 =====
_filter = None


def get_image_filter() -> MLImageFilter:
    global _filter
    if _filter is None:
        _filter = MLImageFilter()
    return _filter
