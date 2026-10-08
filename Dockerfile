# AI Travel Companion — 单服务全栈镜像（FastAPI + 静态前端）
FROM python:3.11-slim

WORKDIR /app

# 先装依赖，利用层缓存
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# 复制代码与前端
COPY backend ./backend
COPY frontend ./frontend

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# 健康检查：/health 返回 200 视为存活
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
