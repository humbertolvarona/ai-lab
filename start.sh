#!/bin/bash
set -e

LOG_FILE="/workspace/ai-lab.log"
touch "$LOG_FILE"

# Logging functions
log()     { echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $1"    | tee -a "$LOG_FILE"; }
error()   { echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1"   | tee -a "$LOG_FILE" >&2; }
success() { echo "$(date '+%Y-%m-%d %H:%M:%S') [SUCCESS] $1" | tee -a "$LOG_FILE"; }
warn()    { echo "$(date '+%Y-%m-%d %H:%M:%S') [WARNING] $1" | tee -a "$LOG_FILE"; }

log "🔧 Starting environment setup..."
log "🕒 Timezone: $TZ"

# Apply timezone at runtime
if ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime && echo "$TZ" > /etc/timezone; then
    success "Timezone set to $TZ"
else
    warn "Failed to set timezone. Default may be used."
fi

log "📆 Current system time: $(date)"
log "🎛️ ENABLE_GPU=$ENABLE_GPU"

if [[ "$ENABLE_GPU" == "yes" ]]; then
    if [[ "$GPU_TYPE" == "CUDA" ]]; then
        log "Activating GPU environment (CUDA)"
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export CUDA_VISIBLE_DEVICES=0
        success "CUDA environment variables set"
        pip install dgl-cu118
    elif [[ "$GPU_TYPE" == "ROCM" ]]; then
        log "Activating GPU environment (ROCm)"
        export HSA_OVERRIDE_GFX_VERSION=10.3.0
        success "ROCm environment variables set"
        pip install dgl
    else
        error "Invalid GPU_TYPE: '$GPU_TYPE'. Use 'CUDA' or 'ROCM'"
        exit 1
    fi
else
    log "🎛️ GPU_TYPE=No GPU"
    log "CPU mode activated (GPU disabled)"
    export CUDA_VISIBLE_DEVICES=""
    pip install dgl
fi

# Check for Jupyter availability
if ! command -v jupyter &> /dev/null; then
    error "'jupyter' command not found. Please ensure Jupyter is installed."
    exit 1
fi

log "🚪 Launching Jupyter Notebook on port 8888..."
exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password='' 2>&1 | tee -a "$LOG_FILE"
