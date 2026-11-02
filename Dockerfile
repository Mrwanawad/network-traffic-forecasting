# ---- batch + API image ---------------------------------------------------------
# Multi-stage: build wheels once, ship a slim runtime without compilers.
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheel
RUN pip wheel --wheel-dir /wheels --extra-index-url https://download.pytorch.org/whl/cpu ".[prophet,dl,mlops]"

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NETTRAFFIC_ARTIFACTS=/app/artifacts
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=build /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels
WORKDIR /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status!=200)"
# default: serve the API; override with `nettraffic run ...` for the batch job
CMD ["uvicorn", "nettraffic.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
