FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN useradd --create-home --uid 10001 appuser
COPY frontend/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
COPY frontend/app.py /app/app.py
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "/app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
