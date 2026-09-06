# 手动启动本地模型网关

从仓库根目录执行：

```bash
bash scripts/local_gateway_startup/start.sh
```

脚本检查依赖和现有容器，按顺序启动 Colima、PostgreSQL 和 LiteLLM，
最后等待网关健康检查通过。可以重复运行；已运行的服务不会重建。
它固定使用本机 `colima` Docker context，不受当前 Docker context 影响。
脚本不需要 API Key，也不会发送模型推理请求。没有安装登录自启项。

配置路径、健康检查等待时限及可覆盖的环境变量见：

```bash
bash scripts/local_gateway_startup/start.sh --help
```

这是已有网关的启动入口。若 Compose 容器尚未创建，脚本会明确失败，
避免把重启操作变成重新部署。网关配置与数据库仍归原有网关目录管理。

## 文件说明

- `start.sh`：手动启动入口。
- `test_startup.py`：使用假命令检查启动顺序、重复运行和失败行为。
- `.gitignore`：排除本任务的测试缓存。
- `cache/`：本任务的本地验证日志和测试临时目录，不进入版本控制。

验证命令：

```bash
bash -n scripts/local_gateway_startup/start.sh
.venv/bin/python -m pytest scripts/local_gateway_startup/test_startup.py \
  --basetemp=scripts/local_gateway_startup/cache/pytest \
  -o cache_dir=scripts/local_gateway_startup/cache/pytest_cache
```
