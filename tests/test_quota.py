"""quota 配额系统单元测试 — 免费额度 / 用量追踪 / 超额阻断 / 充值。"""
import pytest


@pytest.fixture
def quota_env(tmp_path, monkeypatch):
    import config
    import quota

    # 隔离: 用临时文件 + 空状态, 不污染真实 data/quota_usage.json
    monkeypatch.setattr(quota, "_usage_file", tmp_path / "quota.json")
    monkeypatch.setattr(quota, "_usage", {})
    monkeypatch.setattr(config, "QUOTA_ENABLED", True)
    monkeypatch.setattr(config, "QUOTA_FREE_TOKENS", 1000)
    return quota


def test_user_key_resolution(quota_env):
    assert quota_env.get_user_key(user_id="u1") == "user:u1"
    assert quota_env.get_user_key(ip="1.2.3.4") == "anon:1.2.3.4"
    assert quota_env.get_user_key() == "anon:unknown"


def test_free_quota_initial(quota_env):
    s = quota_env.check_quota("user:test")
    assert s["enabled"] is True
    assert s["allowed"] == 1000
    assert s["remaining"] == 1000
    assert s["blocked"] is False


def test_usage_tracking_and_block(quota_env):
    key = "user:test"
    quota_env.record_usage(key, 600)
    assert quota_env.check_quota(key)["remaining"] == 400
    quota_env.record_usage(key, 500)
    s = quota_env.check_quota(key)
    assert s["remaining"] == 0
    assert s["blocked"] is True


def test_record_usage_ignores_nonpositive(quota_env):
    key = "user:test"
    quota_env.record_usage(key, 0)
    quota_env.record_usage(key, -5)
    assert quota_env.check_quota(key)["used"] == 0


def test_top_up_raises_allowance(quota_env):
    key = "user:test"
    quota_env.record_usage(key, 1000)  # 用完
    assert quota_env.check_quota(key)["blocked"] is True
    quota_env.top_up(key, 2000)
    s = quota_env.check_quota(key)
    assert s["allowed"] == 3000
    assert s["remaining"] == 2000
    assert s["blocked"] is False
