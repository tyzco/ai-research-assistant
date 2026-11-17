# Langfuse 自托管（本地观测）

端口：http://localhost:3001（3000 被 lobe-chat 占用）

## 启动 / 停止

```bash
cd deploy_langfuse
docker compose up -d
docker compose ps
docker compose down          # 停止
docker compose down -v       # 停止并删除数据（慎用）
```

首次启动需要拉取 Postgres / ClickHouse / Redis / MinIO 镜像，体积较大。

## 初始化

1. 打开 http://localhost:3001
2. 注册本地账号（自托管默认不发验证邮件）
3. 创建组织与项目，进入 Settings → API Keys，创建一对 public/secret key
4. 把 key 写入项目根目录 `.env`：

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3001
```

5. 重启 `python api.py`

未配置 key 时，`observability/tracer.py` 全部 no-op，不影响运行。
