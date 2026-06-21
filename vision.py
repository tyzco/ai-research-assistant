"""多模态图片理解：对 PDF 提取的图片生成文字描述。"""

import base64
import logging
from pathlib import Path

from openai import AsyncOpenAI

from config import VISION_API_KEY, VISION_BASE_URL, VISION_MODEL

logger = logging.getLogger(__name__)

IMAGE_DESCRIPTION_PROMPT = """请详细描述这张学术论文中的图片。包括：
1. 图表类型（如折线图、流程图、表格截图、模型架构图等）
2. 关键数据、数值、趋势
3. 图表要表达的核心结论
用中文回答，150-300 字。"""


async def describe_image(image_path: str) -> str | None:
    """调用视觉模型对一张图片生成描述。"""
    if not VISION_API_KEY:
        logger.warning("VISION_API_KEY not configured, skipping image description")
        return None

    path = Path(image_path)
    if not path.exists():
        return None

    ext = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(ext, "image/png")

    try:
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to read image {image_path}: {e}")
        return None

    client = AsyncOpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)

    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Vision API error for {image_path}: {e}")
        return None


async def describe_images(image_paths: list[str]) -> dict[str, str]:
    """批量处理图片描述。"""
    results = {}
    for img_path in image_paths:
        desc = await describe_image(img_path)
        if desc:
            results[img_path] = desc
    logger.info(f"Described {len(results)}/{len(image_paths)} images")
    return results
