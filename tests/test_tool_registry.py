"""tool_registry 单元测试: 工具注册 / 查询 / 幂等覆盖 / 插件加载。"""
from agent.tool_registry import TOOLS, Tool, get_tool, register


def _dummy_tool(name="dummy_test_tool"):
    return Tool(name, "测试工具", {"type": "object", "properties": {}}, lambda: "ok")


def test_all_tools_registered_no_dup_names():
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names)), f"工具名重复: {names}"
    # 核心工具必须存在
    for core in [
        "search_papers",
        "search_knowledge_bases",
        "agent_ask",
        "download_and_index",
    ]:
        assert core in names, f"核心工具缺失: {core}"


def test_get_tool_known_and_unknown():
    assert get_tool("search_papers") is not None
    assert get_tool("this_tool_does_not_exist") is None


def test_every_tool_has_schema_fields():
    for t in TOOLS:
        assert t.name, "工具名不能为空"
        assert t.description, f"{t.name} 缺描述"
        assert isinstance(t.parameters, dict), f"{t.name} 缺参数 schema"
        assert callable(t.func), f"{t.name} 缺执行函数"


def test_register_same_name_overwrites():
    name = "search_papers"
    original = get_tool(name)
    dummy = _dummy_tool(name)
    try:
        register(dummy)
        assert get_tool(name) is dummy
        assert sum(1 for t in TOOLS if t.name == name) == 1
    finally:
        register(original)  # 还原


def test_plugin_tools_loaded():
    # tools_extra 插件 (fetch_url/dataio/academic_extra) 应已自动注册
    for plugin_tool in [
        "fetch_url",
        "read_csv",
        "export_references_xlsx",
        "check_plagiarism",
        "lookup_venue_rank",
        "doi_lookup",
    ]:
        assert get_tool(plugin_tool) is not None, f"插件工具缺失: {plugin_tool}"
