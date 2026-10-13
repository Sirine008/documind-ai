# Backend image. Build context is the repository root (see docker-compose.yml) so the
# scripts/ and data/ folders are available inside the image with the same layout.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/srv/models

WORKDIR /srv/documind/backend

COPY backend/requirements.txt backend/requirements-ml.txt ./
RUN pip install -r requirements.txt

# Local embedding model support (default). Disable with --build-arg INSTALL_ML=false
# to build a smaller image that uses DOCUMIND_EMBEDDING_PROVIDER=hashing.
ARG INSTALL_ML=true
RUN if [ "$INSTALL_ML" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install -r requirements-ml.txt; \
    fi

COPY backend/ ./
COPY scripts/ /srv/documind/scripts/
COPY data/sample_documents/ /srv/documind/data/sample_documents/
COPY data/evaluation/default.json /srv/documind/data/evaluation/default.json

RUN useradd --create-home --uid 10001 documind \
 && mkdir -p /srv/documind/data/uploads /srv/documind/data/evaluation/results /srv/models \
 && chown -R documind:documind /srv/documind/data /srv/models \
 && chmod +x docker-entrypoint.sh
USER documind

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/ready', timeout=4).status == 200 else 1)"

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
