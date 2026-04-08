FROM python:3.11-slim

WORKDIR /app

# Set environment to disable web UI (only API endpoints needed)
ENV ENABLE_WEB_INTERFACE=false

# Install uv
RUN pip install uv --no-cache-dir

# Install dependencies directly with pip (avoids venv issues)
RUN uv pip install --system fastapi openai openenv-core pydantic pyyaml uvicorn jinja2 python-multipart requests

# Copy application code
COPY server/ ./server/
COPY models.py ./
COPY pyproject.toml ./
COPY openenv.yaml ./
COPY inference.py ./

# Create non-root user for security
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

CMD ["python", "-c", "from server.app import main; main()"]
