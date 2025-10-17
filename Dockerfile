# -----------------------
# Stage 1: build (install dependencies & train model)
# -----------------------
FROM python:3.11-slim AS build

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# 创建 artifacts 目录并训练模型
RUN mkdir -p /app/artifacts \
    && python -m src.train --kind linear --out /app/artifacts/model.joblib

# -----------------------
# Stage 2: runtime
# -----------------------
FROM python:3.11-slim

WORKDIR /app

# 复制 build 阶段安装的依赖（直接 copy site-packages 也可以，但这里我们重新 pip 安装简单）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制 artifacts 和源码
COPY --from=build /app/artifacts /app/artifacts
COPY src/ src/

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
