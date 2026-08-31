FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FACEKYC_CONFIG_PATH=/app/configs/base.yaml

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN python -m pip install --no-cache-dir ".[vision,api]"

COPY configs /app/configs
COPY backend /app/backend
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
