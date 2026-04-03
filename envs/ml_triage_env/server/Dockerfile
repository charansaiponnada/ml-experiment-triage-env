FROM python:3.11-slim
WORKDIR /app
RUN pip install uv --no-cache-dir
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-cache
COPY . .
RUN useradd -m -u 1000 user
USER user
EXPOSE 7860
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
