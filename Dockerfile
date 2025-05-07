FROM python:3.10-slim

ARG BUILD_DATE
ARG VERSION="v1"

LABEL maintainer="VaronaTech"
LABEL build_version="Nginx version:- ${VERSION} Build-date:- ${BUILD_DATE}"
LABEL org.opencontainers.image.authors="HL Varona <humberto.varona@gmail.com>"
LABEL org.opencontainers.image.description="AI-Lab: Artificial Intelligence Laboratory"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=America/Recife \
    ENABLE_GPU=no \
    GPU_TYPE=CUDA

WORKDIR /workspace

# Dependencias básicas + LaTeX
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    nodejs \
    npm \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-xetex \
    python3-dev \
    libpython3-dev \
    dvipng \
    unzip \
    p7zip \
    jq \
    netcdf-bin \
    nco \
    cdo \
    && printf "AI-Lab version: ${VERSION}\nBuild-date: ${BUILD_DATE}" > /build_version \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN node -v && npm -v

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN mkdir -p /workspace
WORKDIR /workspace

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8888/lab || exit 1

ENTRYPOINT ["/usr/local/bin/start.sh"]
