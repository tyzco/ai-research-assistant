"""Agent 工作流可视化元数据 (文档型常量)。

这里**不承载业务逻辑** —— 只描述"代码里真实存在的执行顺序", 供
/flow/graph 端点与 static/flow.html 渲染。agents/tools/hooks 那三类
结构化数据在 api.py 里直接从代码导出; 本文件只补两样代码无法自描述的
信息: 每一轮主循环的阶段划分, 以及每段对应的代码锚点。

改 _loop 的顺序时, 记得同步这里 (flow.html 顶部会显示 anchor 供核对)。
"""

# 三层 → 中文名 / 一句话职责
LAYERS = [
    {"id": "upstream", "name": "调研层", "desc": "选题 → 检索 → 筛选, 产出一份可信的文献池"},
    {"id": "midstream", "name": "产出层", "desc": "综述 → 分章写作 → 图表 → 参考文献"},
    {"id": "downstream", "name": "智控层", "desc": "同行评审 → 引文核验 → 格式 → 伦理 → 终稿"},
]

GATES = [
    {
        "id": "topic_direction",
        "name": "选题方向闸门",
        "detail": "选题评估 Agent 给出可行性/创新性/研究空白后暂停, 由人确认方向再检索。",
    },
    {
        "id": "core_literature",
        "name": "核心文献闸门",
        "detail": "质量筛选 Agent 给出核心文献清单后暂停, 由人决定哪些真正进入后续写作。",
    },
    {
        "id": "final_conclusion",
        "name": "最终结论闸门",
        "detail": "终稿 Agent 完成后暂停, 由人确认结论与全文, 才视为交付。",
    },
]

# Pi-Agent 单任务主循环 (agent/pi_agent.py:_loop 的真实顺序)。
# kind: control=流程控制 / hook=干预点 / model=LLM / tool=工具 / guard=护栏 / output=输出
PI_LOOP = [
    {
        "id": "preflight",
        "title": "任务预处理",
        "kind": "control",
        "detail": "技能路由选中能力集 → 收窄可用工具 → 生成系统提示词 → 记录模型与提示词指纹",
        "code": "agent/pi_agent.py:run()",
        "steps": [
            "select_for_task(task) 选技能",
            "tools_for_task() 收窄工具集 (路由弱时不收窄)",
            "sha1(system_prompt)[:8] 写入 trace.prompt_version",
        ],
    },
    {
        "id": "plan",
        "title": "计划阶段 (可选)",
        "kind": "control",
        "detail": "长任务先让模型出计划; 开启 plan 模式时暂停等人工确认方向, 短任务直接跳过",
        "code": "agent/pi_agent.py:_plan()",
        "steps": [
            "len(task) >= AGENT_PLAN_MIN_CHARS 才触发",
            "计划以 assistant 消息回灌 messages",
            "plan 模式下写 _pending_confirm.type = 'plan'",
        ],
    },
    {
        "id": "trace_start",
        "title": "TRACE_START",
        "kind": "hook",
        "detail": "整个任务的第一个干预点: 可注入系统消息、设置限制; 返回 block 则直接结束",
        "code": "agent/hooks.py:HookPoint.TRACE_START",
        "hook": "trace_start",
    },
    {
        "id": "turn_start",
        "title": "TURN_START · 步数守卫",
        "kind": "guard",
        "detail": "每轮开始检查步数上限; 命中则强制结束 (硬护栏, 不可被模型绕过)",
        "code": "agent/hooks.py:_max_steps_guard",
        "hook": "turn_start",
        "steps": ["steps >= max_steps → block", "否则重置本轮 turn 状态"],
    },
    {
        "id": "compact",
        "title": "上下文压缩检查",
        "kind": "control",
        "detail": "按 token 水位压缩历史; 关键发现走独立通道, 不会被压缩摘要吞掉",
        "code": "agent/context_engine.py",
        "steps": [
            "保留最近若干条完整消息",
            "更早内容压成摘要",
            "trace.key_findings 单独保留, 摘要里附带【关键发现】",
        ],
    },
    {
        "id": "before_model",
        "title": "BEFORE_MODEL",
        "kind": "hook",
        "detail": "模型调用前可改写 messages、缩小工具集 (只允许缩小, 不允许新增)",
        "code": "agent/pi_agent.py:_fire_hook('before_model')",
        "hook": "before_model",
    },
    {
        "id": "llm_call",
        "title": "LLM 调用",
        "kind": "model",
        "detail": "流式调用 DeepSeek; 统一带 thinking=disabled (推理模型契约); 统计本轮 token",
        "code": "agent/pi_agent.py:_loop() → llm_client",
        "model": True,
    },
    {
        "id": "after_model",
        "title": "AFTER_MODEL",
        "kind": "hook",
        "detail": "模型返回后审计 tool_calls: 可过滤危险工具, 把调用结果回灌给模型",
        "code": "agent/pi_agent.py:_fire_hook('after_model')",
        "hook": "after_model",
    },
    {
        "id": "tool_limits",
        "title": "单轮工具硬上限",
        "kind": "guard",
        "detail": "对已知会自检过度的工具设上限: check_draft_quality ≤2 次, agent_ask ≤3 次, 超出直接回灌提示让它交付",
        "code": "agent/pi_agent.py:_loop() 内 tool_name 分支",
        "steps": ["防止模型用「再检查一次」烧光步数"],
    },
    {
        "id": "validate",
        "title": "参数校验 + 自动修正",
        "kind": "guard",
        "detail": "按 JSON Schema 校验必填/类型/enum/range; 可改的先修; 改不对计数, 超预算就禁用该工具",
        "code": "agent/tool_manager.py:validate_params",
        "steps": [
            "is_valid → 用 corrected 参数",
            "参数错误计数 > AGENT_PARAM_CORRECTION_BUDGET → 明确禁止再调",
        ],
    },
    {
        "id": "confirm",
        "title": "人工确认门",
        "kind": "guard",
        "detail": "need_confirm 工具 (写库/导出等) 执行前暂停, 返回 run_id; 人确认后 resume 续跑",
        "code": "agent/pi_agent.py:_pending_confirm",
        "steps": [
            "同一轮剩余待确认工具排队 (_pending_queue)",
            "暂停时把 messages/state/队列落 SQLite",
        ],
    },
    {
        "id": "before_tool",
        "title": "BEFORE_TOOL · 安全策略",
        "kind": "hook",
        "detail": "工具执行前的安全闸: hook 可 block; 另有 security_check 检查危险参数",
        "code": "agent/hooks.py + tool_manager.security_check",
        "hook": "before_tool",
    },
    {
        "id": "budget",
        "title": "调用预算",
        "kind": "guard",
        "detail": "工具黑名单 + 单任务频次上限; 被拦下的调用不计入频次",
        "code": "agent/tool_manager.py:check_call_budget",
    },
    {
        "id": "repeat",
        "title": "重复动作检测",
        "kind": "guard",
        "detail": "同工具 + 同参数反复调用 → 不再执行, 把观察结果换成警示, 让模型换路",
        "code": "agent/pi_agent.py:_loop() call_fingerprints",
        "steps": [
            "指纹 = 工具名 + 排序后的参数 JSON (前 200 字)",
            "最近 6 次里重复达到 AGENT_REPEAT_CALL_LIMIT → 拦截并计数",
        ],
    },
    {
        "id": "execute",
        "title": "工具执行 (按可重放性重试)",
        "kind": "tool",
        "detail": "retryable=True 的瞬时故障自动重试; 有副作用的工具绝不盲重试 (超时≠没生效)",
        "code": "agent/tool_manager.py:execute_with_retry",
        "steps": [
            "classify_tool_error 区分瞬时/永久错误",
            "永久错误不重试, 直接格式化回灌给模型",
        ],
    },
    {
        "id": "truncate",
        "title": "工具输出截断",
        "kind": "control",
        "detail": "过长结果截断后回灌, 防止单次工具输出把上下文挤爆",
        "code": "agent/context_engine.py 截断策略",
    },
    {
        "id": "failure_circuit",
        "title": "错误累积熔断",
        "kind": "guard",
        "detail": "连续工具失败达到阈值 → 整体熔断并如实报告, 不在错误路径上继续烧 token",
        "code": "agent/pi_agent.py: trace.consecutive_failures",
    },
    {
        "id": "after_tool",
        "title": "AFTER_TOOL",
        "kind": "hook",
        "detail": "记录工具调用 (名称/参数/结果摘要/成功与否), 供 trace 与统计使用",
        "code": "agent/pi_agent.py:_fire_hook('after_tool')",
        "hook": "after_tool",
    },
    {
        "id": "direct_answer",
        "title": "直答分支 (无工具调用)",
        "kind": "output",
        "detail": "模型不再调工具 → 流式生成答案; 过滤模型偶发输出的 <tool_calls> XML; 被截断则续写",
        "code": "agent/pi_agent.py:_loop() else 分支",
    },
    {
        "id": "reflexion",
        "title": "Reflexion 自批判",
        "kind": "control",
        "detail": "直答后轻量反思: 答案不完整则带批判意见重试一轮 (有界 1 次, 防无限自我修订)",
        "code": "agent/pi_agent.py:_reflexion_critique",
    },
    {
        "id": "before_emit",
        "title": "BEFORE_EMIT · 输出闸",
        "kind": "hook",
        "detail": "结果输出前做红线/敏感词检测; 命中则整段替换为拦截原因",
        "code": "agent/hooks.py:_sensitive_word_filter",
        "hook": "before_emit",
    },
    {
        "id": "completion",
        "title": "完成判定 + TRACE_END",
        "kind": "output",
        "detail": "assess_completion 5 条硬条件判定是否真的完成; 写终止原因; 做审计统计",
        "code": "agent/pi_agent.py:assess_completion",
        "steps": [
            "terminated_normally / has_output / not_fallback_text",
            "tool_evidence / grounded_or_declared",
        ],
    },
]

# 三级兜底 (与 pi_agent 的实际行为对应)
FALLBACKS = [
    {
        "level": "一级: 参数错",
        "detail": "给一次修正机会; 超预算停用该工具, 让它换路",
        "code": "AGENT_PARAM_CORRECTION_BUDGET",
    },
    {
        "level": "二级: 瞬时故障",
        "detail": "只重试 retryable=True 的工具; 有副作用的操作先查状态再决定",
        "code": "tool_manager.execute_with_retry",
    },
    {
        "level": "三级: 连续失败",
        "detail": "连续 N 次失败熔断, 如实报告最后一次错误与建议",
        "code": "AGENT_MAX_CONSECUTIVE_FAILURES",
    },
]
