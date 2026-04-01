FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv
RUN uv sync --frozen
EXPOSE 7860
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
