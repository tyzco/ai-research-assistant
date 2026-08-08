# 模块 05 · 科研团队架构

## 上游调研层（选题 / 检索 / 筛选）

topic_evaluator → multi_source_searcher → quality_screener

## 中游产出层（综述 / 引言 / 方法 / 实验 / 图表 / 参考文献）

review_synthesizer → chapter_writer_intro → chapter_writer_method → chapter_writer_experiment → figure_generator → bib_manager

## 下游智控层（审稿 / 格式 / 伦理 / 定稿）

peer_reviewer → citation_verifier → format_checker → ethics_reviewer → final_drafter

## 人工审批闸门

选题方向 → 核心文献入选 → 最终结论定稿