# Build context: project root.
#   CPU / remote backends (default): docker compose up -d --build
#   CUDA (in-process llama.cpp):     docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG WITH_LOCAL_LLM=0
ARG LLAMA_CMAKE_ARGS=""
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Compiler is only needed to build llama-cpp-python; python is installed if the base image lacks it (CUDA image).
RUN apt-get update \
 && (command -v python3 >/dev/null || apt-get install -y --no-install-recommends python3 python3-pip python3-venv) \
 && if [ "$WITH_LOCAL_LLM" = "1" ]; then apt-get install -y --no-install-recommends build-essential cmake; fi \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /code
COPY app/requirements.txt app/requirements.lock app/requirements-local.txt app/
RUN pip install --no-cache-dir --retries 10 --timeout 60 -r app/requirements.txt -c app/requirements.lock --index-url https://pypi.mydigipay.info/simple
RUN if [ "$WITH_LOCAL_LLM" = "1" ]; then \
      CMAKE_ARGS="$LLAMA_CMAKE_ARGS" pip install --no-cache-dir -r app/requirements-local.txt -c app/requirements.lock --index-url https://pypi.mydigipay.info/simple; \
    fi

COPY app app
# Single source of truth for the schema shown to the models (read by app/sql/prompt.py).
COPY ingestion/schema.sql ingestion/schema.sql
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
