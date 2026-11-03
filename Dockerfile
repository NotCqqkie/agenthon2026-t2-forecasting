FROM python:3.13-slim-bookworm@sha256:5024f48ba9441d4b13a95d3945abc6365538e3a31109833367a1923523c6efed
LABEL qfbench2.interface_version="2.0"
LABEL qfbench2.track="forecasting"
LABEL qfbench2.verb="forecast"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/tmp \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN pip install --no-cache-dir "numpy==2.1.3" "pandas==2.2.3" "pyarrow==18.1.0"
COPY t2agent /opt/t2agent
ENV PYTHONPATH=/opt
RUN printf '#!/bin/sh\nexec python3 -m t2agent "$@"\n' > /usr/local/bin/forecast && chmod +x /usr/local/bin/forecast
CMD ["forecast", "--help"]
