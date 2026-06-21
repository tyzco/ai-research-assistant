"""Chainlit 前端（端口 8000）：完整的学术文献精读助手界面。"""

import logging
import traceback
import uuid
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("chainlit_app")

from downloader import chunk_text, parse_pdf
from knowledge_base import build_knowledge_base, store_image_descriptions
from models import PaperMeta, TopicStatus
from qa_engine import ask_question
from search import generate_search_strategy
from topic_manager import active_topics, create_topic
from vision import describe_images


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "# 📚 学术文献精读助手\n\n"
            "输入一个研究方向，我会帮你：\n"
            "1. **生成检索策略**（布尔检索式、推荐数据库、关键学者）\n"
            "2. **上传你的论文 PDF**，我帮你精读\n"
            "3. **深度问答**——带精确溯源和图表理解\n\n"
            "**直接输入研究方向开始！**"
        )
    ).send()
    cl.user_session.set("current_topic_id", None)
    cl.user_session.set("awaiting_upload", False)


@cl.on_message
async def on_message(msg: cl.Message):
    try:
        user_input = msg.content.strip()
        if not user_input:
            return

        logger.info(f"收到消息: {user_input[:80]}")

        topic_id = cl.user_session.get("current_topic_id")
        awaiting_upload = cl.user_session.get("awaiting_upload", False)

        # 检查附件中有无 PDF
        pdf_files = [
            el
            for el in msg.elements
            if isinstance(el, cl.File) and el.name.endswith(".pdf")
        ]
        if pdf_files and topic_id:
            await _handle_upload(topic_id, pdf_files)
            return

        # 有课题且在等待上传→可能提问
        if awaiting_upload and topic_id:
            if "?" in user_input or "？" in user_input:
                await _handle_question(topic_id, user_input)
            else:
                await cl.Message(content="请点击上方 📎 按钮上传论文 PDF。\n\n也可以直接输入带问号的问题。").send()
            return

        # 新建课题
        if not topic_id:
            await _handle_new_topic(user_input)
        else:
            await _handle_question(topic_id, user_input)

    except Exception as e:
        logger.error(f"消息处理异常: {e}\n{traceback.format_exc()}")
        await cl.Message(content=f"❌ 处理出错：{str(e)}").send()


async def _handle_new_topic(query: str):
    msg = cl.Message(content="🧠 正在生成检索策略...")
    await msg.send()

    try:
        state = create_topic(query)
        strategy = await generate_search_strategy(query)
        state.search_strategy = strategy

        cl.user_session.set("current_topic_id", state.topic_id)
        cl.user_session.set("awaiting_upload", True)

        kws_cn = "、".join(strategy.get("keywords_cn", []))
        kws_en = "、".join(strategy.get("keywords_en", []))

        lines = [
            f"## 🔍 检索策略：{query}",
            "",
            f"**中文关键词**：{kws_cn}",
            f"**英文关键词**：{kws_en}",
            "",
            "### 📋 推荐检索式",
            "",
        ]
        for bq in strategy.get("boolean_queries", []):
            lines.append(f"**{bq.get('database', '')}**")
            lines.append(f"```\n{bq.get('query', '')}\n```")
            lines.append(f"_{bq.get('note', '')}_")
            lines.append("")

        if strategy.get("top_authors"):
            lines.append("### 👤 领域关键学者")
            lines.append("")
            for a in strategy.get("top_authors", [])[:5]:
                lines.append(
                    f"- **{a.get('name', '')}** ({a.get('institution', '')}) — {a.get('reason', '')}"
                )

        if strategy.get("search_tips"):
            lines.append(f"\n### 💡 检索建议\n{strategy.get('search_tips', '')}")

        lines.append("\n---\n### 📤 下一步：上传论文 PDF\n点击上方 📎 按钮，选择你下载的论文 PDF。")

        msg.content = "\n".join(lines)
        await msg.update()
        logger.info(f"新课题: {state.topic_id}, keywords_cn={kws_cn}")

    except Exception as e:
        logger.error(f"生成检索策略失败: {e}")
        msg.content = f"❌ 生成检索策略失败：{str(e)}"
        await msg.update()


async def _handle_question(topic_id: str, question: str):
    state = active_topics.get(topic_id)
    if not state or not state.lancedb_table:
        await cl.Message(content="⚠️ 请先上传论文 PDF 构建知识库后再提问。").send()
        return

    msg = cl.Message(content="🔍 正在检索和分析...")
    await msg.send()

    try:
        result = await ask_question(state.lancedb_table, question)
        content = result.answer or "(暂无回答)"

        if result.references:
            content += "\n\n---\n**📖 引用文献**\n"
            for r in result.references[:10]:
                content += f"- {r['title']} ({r.get('year', 'N/A')})\n"

        if result.supplement:
            content += "\n**🔎 建议补充（库内无全文）**\n"
            for s in result.supplement[:5]:
                content += f"- {s['title']} ({s.get('year', 'N/A')}) | DOI: {s.get('doi', 'N/A')}\n"
            content += "\n_以上论文建议通过学校图书馆数据库自行检索全文。_"

        msg.content = content
        await msg.update()
        if result.images:
            elements = []
            for img_path in result.images[:3]:
                try:
                    elements.append(
                        cl.Image(
                            name=Path(img_path).name, path=img_path, display="inline"
                        )
                    )
                except Exception:
                    pass
            if elements:
                await cl.Message(content="**📊 相关图表**", elements=elements).send()

    except Exception as e:
        logger.error(f"问答失败: {e}")
        msg.content = f"❌ 问答失败：{str(e)}"
        await msg.update()


async def _handle_upload(topic_id: str, files: list[cl.File]):
    state = active_topics.get(topic_id)
    if not state:
        await cl.Message(content="⚠️ 课题不存在，请输入研究方向重新开始。").send()
        return

    msg = cl.Message(content="📥 正在处理论文...")
    await msg.send()

    try:
        papers: list[PaperMeta] = []
        fulltext_chunks: dict[str, list[str]] = {}
        all_images: list[dict] = []

        for file in files:
            pdf_name = file.name
            pdf_bytes = (
                Path(file.path).read_bytes()
                if file.path
                else getattr(file, "content", b"")
            )
            if not pdf_bytes:
                continue

            msg.content = f"📄 正在解析：{pdf_name}..."
            await msg.update()
            result = parse_pdf(pdf_bytes, pdf_name)
            paper_id = uuid.uuid4().hex[:16]
            papers.append(
                PaperMeta(
                    paper_id=paper_id, title=pdf_name.replace(".pdf", ""), is_oa=True
                )
            )

            chunks = chunk_text(result["full_text"])
            if chunks:
                fulltext_chunks[paper_id] = chunks
            for img in result["images"]:
                img["paper_id"] = paper_id
            all_images.extend(result["images"])

        msg.content = f"🧠 正在构建向量索引（{len(papers)} 篇论文）..."
        await msg.update()
        table_name = await build_knowledge_base(papers, fulltext_chunks, topic_id)

        described_count = 0
        if all_images:
            msg.content = f"🖼️ 正在理解图表（{len(all_images)} 张）..."
            await msg.update()
            descs = await describe_images([img["image_path"] for img in all_images])
            image_records = []
            for img in all_images:
                img_path = img["image_path"]
                if img_path in descs:
                    image_records.append(
                        {
                            "paper_id": img.get("paper_id", ""),
                            "title": Path(img_path).stem,
                            "text": descs[img_path],
                            "image_path": img_path,
                            "page_number": img.get("page_num", 0),
                        }
                    )
            store_image_descriptions(table_name, image_records)
            described_count = len(image_records)

        state.lancedb_table = table_name
        state.uploaded_papers = len(papers)
        state.total_images = described_count
        state.status = TopicStatus.READY

        cl.user_session.set("awaiting_upload", False)

        msg.content = (
            f"✅ **知识库就绪！**\n\n"
            f"- 📄 论文：{len(papers)} 篇\n"
            f"- 📝 文本块：{sum(len(v) for v in fulltext_chunks.values())} 个\n"
            f"- 🖼️ 图表：{len(all_images)} 张（{described_count} 张已理解）\n\n"
            f"现在可以提问了！"
        )
        await msg.update()

    except Exception as e:
        logger.error(f"上传处理失败: {e}\n{traceback.format_exc()}")
        msg.content = f"❌ 处理失败：{str(e)}"
        await msg.update()


@cl.action_callback("new_topic")
async def on_new_topic(_action):
    cl.user_session.set("current_topic_id", None)
    cl.user_session.set("awaiting_upload", False)
    await cl.Message(content="请输入新的研究方向。").send()
