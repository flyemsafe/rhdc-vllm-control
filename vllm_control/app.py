#!/usr/bin/env python3
"""
vLLM On-Demand Control API

Provides scale-to-zero behavior for vLLM containers on Akili.
Containers are created but not started - this API starts them on demand.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8100

Configuration:
    Set VLLM_CONTROL_CONFIG environment variable to path of models.yaml
    Default: /usr/local/etc/vllm/models.yaml
"""

import os
import subprocess
import time
import logging
import re
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_PATH = os.getenv('VLLM_CONTROL_CONFIG', '/usr/local/etc/vllm/models.yaml')

# Eviction policy configuration
class EvictionPolicy(str, Enum):
    MANUAL = "manual"  # User must manually stop models (default)
    LRU = "lru"        # Automatically stop least recently used model
    STOP_ALL = "stop_all"  # Stop all running models before starting new one

EVICTION_POLICY = EvictionPolicy(os.getenv('VLLM_EVICTION_POLICY', 'manual'))

# Fallback model definitions (used if config file not found)
FALLBACK_MODELS = {
    "qwen-coder-32b": {
        "container": "vllm-qwen25-coder-32b-awq",
        "port": 8010,
        "model_path": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
    },
    "deepseek-coder-33b": {
        "container": "vllm-deepseek-coder-v2-33b-awq",
        "port": 8011,
        "model_path": "deepseek-ai/DeepSeek-Coder-V2-Instruct-AWQ"
    },
    "codestral-22b": {
        "container": "vllm-codestral-22b-awq",
        "port": 8012,
        "model_path": "mistralai/Codestral-22B-v0.1-AWQ"
    },
    "mixtral-8x22b": {
        "container": "vllm-mixtral-8x22b-awq",
        "port": 8013,
        "model_path": "mistralai/Mixtral-8x22B-Instruct-v0.1-AWQ"
    },
    "mistral-7b": {
        "container": "vllm-mistral-7b-instruct-awq",
        "port": 8015,
        "model_path": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    },
}

def load_models_config() -> Dict:
    """
    Load model configuration from YAML file.
    Falls back to FALLBACK_MODELS if file not found.
    """
    config_file = Path(CONFIG_PATH)

    if not config_file.exists():
        logger.warning(f"Config file not found at {CONFIG_PATH}, using fallback models")
        return FALLBACK_MODELS

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            models = config.get('models', {})
            logger.info(f"Loaded {len(models)} models from {CONFIG_PATH}")
            return models
    except Exception as e:
        logger.error(f"Failed to load config from {CONFIG_PATH}: {e}")
        logger.warning("Using fallback models")
        return FALLBACK_MODELS

# Load models from config file
MODELS = load_models_config()

# Track last activity time per model
last_activity: Dict[str, datetime] = {}

app = FastAPI(
    title="vLLM On-Demand Control API",
    description="Start/stop vLLM containers on demand for GPU resource management",
    version="1.0.0"
)

class ModelInfo(BaseModel):
    """Model information response"""
    name: str
    container: str
    port: int
    model_path: str
    status: str
    endpoint: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_available: int
    models_running: int

def run_podman_command(args: list) -> subprocess.CompletedProcess:
    """Run a podman command with error handling"""
    try:
        result = subprocess.run(
            ["podman"] + args,
            capture_output=True,
            text=True,
            check=True
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Podman command failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Podman error: {e.stderr}")

def get_container_status(container_name: str) -> str:
    """Get the status of a container"""
    try:
        result = subprocess.run(
            ["podman", "inspect", "--format", "{{.State.Status}}", container_name],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "not_found"
    except Exception as e:
        logger.error(f"Failed to get container status: {e}")
        return "error"

def wait_for_model_ready(port: int, timeout: int = 120) -> bool:
    """
    Wait for vLLM model to be ready by checking the /v1/models endpoint

    Args:
        port: The port where vLLM is running
        timeout: Maximum time to wait in seconds

    Returns:
        True if model is ready, False if timeout
    """
    endpoint = f"http://localhost:{port}/v1/models"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(endpoint, timeout=2)
            if response.ok:
                logger.info(f"Model on port {port} is ready")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)

    logger.warning(f"Model on port {port} did not become ready within {timeout}s")
    return False

def get_gpu_memory() -> Tuple[float, float]:
    """
    Get GPU memory usage using nvidia-smi

    Returns:
        Tuple of (used_mb, free_mb)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output: "used_mb, free_mb"
        line = result.stdout.strip()
        used, free = map(float, line.split(','))
        return used, free
    except Exception as e:
        logger.error(f"Failed to query GPU memory: {e}")
        return 0.0, 0.0

def estimate_model_memory(model_name: str) -> float:
    """
    Estimate GPU memory required for a model in MB

    Based on model name patterns. This is a rough estimate.
    For AWQ models, we use approximately 90% of GPU for large models.

    Args:
        model_name: Name of the model

    Returns:
        Estimated memory in MB
    """
    # Extract model size from name (e.g., "7b", "22b", "32b")
    size_match = re.search(r'(\d+)b', model_name.lower())
    if not size_match:
        # Default conservative estimate: 40GB for unknown models
        return 40000.0

    size_b = int(size_match.group(1))

    # AWQ 4-bit quantization rough estimates (MB):
    # 7B model: ~4GB model + ~36GB KV cache = ~40GB total
    # 22B model: ~12GB model + ~30GB KV cache = ~42GB total
    # 32B model: ~17GB model + ~25GB KV cache = ~42GB total
    # For safety, we estimate 90% of 48GB GPU = 43GB for large models

    if size_b <= 7:
        return 40000.0  # 40GB
    elif size_b <= 22:
        return 42000.0  # 42GB
    else:
        return 43000.0  # 43GB

def estimate_model_startup_timeout(model_path: str) -> int:
    """
    Estimate startup timeout for a model based on its size

    Large models (22B+) need more time to initialize CUDA graphs and optimizations.

    Args:
        model_path: Path or name of the model (e.g., "mistralai/Codestral-22B-v0.1-AWQ")

    Returns:
        Timeout in seconds
    """
    # Extract model size from path (e.g., "7b", "22b", "32b")
    size_match = re.search(r'(\d+)b', model_path.lower())
    if not size_match:
        # Unknown model size - use conservative timeout
        logger.info(f"Could not determine model size from '{model_path}', using 180s timeout")
        return 180

    size_b = int(size_match.group(1))

    # Timeout estimates based on CUDA graph initialization times:
    # - 7B models: ~60-90s (base timeout 120s is sufficient)
    # - 22B models: ~150-180s (need 240s = 4 minutes)
    # - 32B+ models: ~180-240s (need 300s = 5 minutes)
    # - 70B+ models: ~300-360s (need 420s = 7 minutes)

    if size_b <= 7:
        timeout = 120  # 2 minutes
    elif size_b <= 13:
        timeout = 180  # 3 minutes
    elif size_b <= 22:
        timeout = 240  # 4 minutes
    elif size_b <= 34:
        timeout = 300  # 5 minutes
    elif size_b <= 70:
        timeout = 420  # 7 minutes
    else:
        timeout = 600  # 10 minutes for very large models

    logger.info(f"Model size {size_b}B detected, using {timeout}s startup timeout")
    return timeout

def get_running_models() -> List[Tuple[str, str]]:
    """
    Get list of currently running models

    Returns:
        List of tuples (model_name, container_name)
    """
    running = []
    for name, config in MODELS.items():
        status = get_container_status(config["container"])
        if status == "running":
            running.append((name, config["container"]))
    return running

def get_lru_model() -> Optional[str]:
    """
    Get the least recently used model name

    Returns:
        Model name of LRU model, or None if no models are running
    """
    running_models = get_running_models()
    if not running_models:
        return None

    # Find model with oldest last_activity time
    lru_model = None
    oldest_time = datetime.now()

    for model_name, _ in running_models:
        model_time = last_activity.get(model_name, datetime.min)
        if model_time < oldest_time:
            oldest_time = model_time
            lru_model = model_name

    # If no activity tracked, return first running model
    if lru_model is None and running_models:
        lru_model = running_models[0][0]

    return lru_model

async def handle_eviction(model_name: str, eviction_policy: EvictionPolicy) -> Dict[str, any]:
    """
    Handle model eviction based on policy

    Args:
        model_name: Name of model to start
        eviction_policy: Eviction policy to use

    Returns:
        Dict with eviction info
    """
    running_models = get_running_models()

    if not running_models:
        return {"evicted": []}

    evicted = []

    if eviction_policy == EvictionPolicy.STOP_ALL:
        # Stop all running models
        logger.info(f"Eviction policy 'stop_all': stopping all {len(running_models)} running models")
        for name, container in running_models:
            logger.info(f"Stopping model '{name}' (container: {container})")
            await stop_model(name)
            evicted.append(name)

    elif eviction_policy == EvictionPolicy.LRU:
        # Stop least recently used model
        lru_model = get_lru_model()
        if lru_model:
            logger.info(f"Eviction policy 'lru': stopping least recently used model '{lru_model}'")
            await stop_model(lru_model)
            evicted.append(lru_model)

    return {"evicted": evicted}

@app.get("/api/info", response_model=dict)
async def api_info():
    """API info - returns basic information about the API"""
    return {
        "service": "vLLM On-Demand Control API",
        "version": "1.0.0",
        "models_available": len(MODELS),
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "gpu": "/api/gpu",
            "start": "/models/{model_name}/start",
            "stop": "/models/{model_name}/stop",
            "status": "/models/{model_name}/status",
            "web_ui": "/"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    running_count = 0
    for model_config in MODELS.values():
        status = get_container_status(model_config["container"])
        if status == "running":
            running_count += 1

    return HealthResponse(
        status="healthy",
        models_available=len(MODELS),
        models_running=running_count
    )

@app.get("/models", response_model=Dict[str, ModelInfo])
async def list_models():
    """List all available models and their status"""
    models_info = {}
    for name, config in MODELS.items():
        status = get_container_status(config["container"])
        endpoint = f"http://akili.lab.rodhouse.net:{config['port']}" if status == "running" else None

        models_info[name] = ModelInfo(
            name=name,
            container=config["container"],
            port=config["port"],
            model_path=config["model_path"],
            status=status,
            endpoint=endpoint
        )

    return models_info

@app.get("/models/{model_name}/status", response_model=ModelInfo)
async def get_model_status(model_name: str):
    """Get the status of a specific model"""
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    config = MODELS[model_name]
    status = get_container_status(config["container"])
    endpoint = f"http://akili.lab.rodhouse.net:{config['port']}" if status == "running" else None

    return ModelInfo(
        name=model_name,
        container=config["container"],
        port=config["port"],
        model_path=config["model_path"],
        status=status,
        endpoint=endpoint
    )

@app.post("/models/{model_name}/start")
async def start_model(
    model_name: str,
    evict: Optional[str] = Query(None, description="Eviction policy: 'manual', 'lru', or 'stop_all'")
):
    """
    Start a vLLM container on demand with GPU memory awareness

    Query Parameters:
        evict: Override default eviction policy for this request
               - 'manual': Don't auto-evict (return error if insufficient memory)
               - 'lru': Evict least recently used model if needed
               - 'stop_all': Stop all running models before starting

    Returns the endpoint URL once the model is ready to serve requests
    """
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    config = MODELS[model_name]
    container = config["container"]
    port = config["port"]
    model_path = config["model_path"]

    # Check current status
    status = get_container_status(container)

    if status == "running":
        logger.info(f"Model '{model_name}' is already running")
        # Update last activity even if already running
        last_activity[model_name] = datetime.now()
        endpoint = f"http://akili.lab.rodhouse.net:{port}"
        return {
            "status": "already_running",
            "model": model_name,
            "container": container,
            "endpoint": endpoint
        }

    if status == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"Container '{container}' not found. Run the vLLM deployment playbook first."
        )

    # Determine eviction policy
    policy = EvictionPolicy(evict) if evict else EVICTION_POLICY

    # Pre-flight check: GPU memory availability
    used_mb, free_mb = get_gpu_memory()
    estimated_needed_mb = estimate_model_memory(model_path)

    logger.info(f"GPU Memory - Used: {used_mb:.0f}MB, Free: {free_mb:.0f}MB, Estimated needed: {estimated_needed_mb:.0f}MB")

    # Check if we have enough free memory
    if free_mb < estimated_needed_mb:
        running_models = get_running_models()
        running_model_names = [m[0] for m in running_models]

        # Not enough memory - handle based on eviction policy
        if policy == EvictionPolicy.MANUAL:
            # Manual policy: return helpful error
            raise HTTPException(
                status_code=507,  # Insufficient Storage
                detail={
                    "error": "insufficient_gpu_memory",
                    "message": f"Need ~{estimated_needed_mb/1024:.1f}GB but only {free_mb/1024:.1f}GB free",
                    "gpu_memory": {
                        "used_mb": used_mb,
                        "free_mb": free_mb,
                        "total_mb": used_mb + free_mb,
                        "needed_mb": estimated_needed_mb
                    },
                    "running_models": running_model_names,
                    "suggestion": f"Stop one or more running models first, or use ?evict=lru parameter",
                    "commands": {
                        "stop_model": f"curl -X POST http://akili.lab.rodhouse.net:8100/models/{{model_name}}/stop",
                        "auto_evict_lru": f"curl -X POST 'http://akili.lab.rodhouse.net:8100/models/{model_name}/start?evict=lru'",
                        "auto_evict_all": f"curl -X POST 'http://akili.lab.rodhouse.net:8100/models/{model_name}/start?evict=stop_all'"
                    }
                }
            )
        else:
            # LRU or STOP_ALL policy: automatically evict
            logger.info(f"Insufficient GPU memory. Applying eviction policy: {policy}")
            eviction_result = await handle_eviction(model_name, policy)
            # Wait a moment for containers to fully stop and free GPU memory
            time.sleep(3)

            # Re-check memory after eviction
            used_mb, free_mb = get_gpu_memory()
            logger.info(f"After eviction - Used: {used_mb:.0f}MB, Free: {free_mb:.0f}MB")

            if free_mb < estimated_needed_mb:
                raise HTTPException(
                    status_code=507,
                    detail={
                        "error": "insufficient_gpu_memory_after_eviction",
                        "message": f"Still need ~{estimated_needed_mb/1024:.1f}GB but only {free_mb/1024:.1f}GB free after evicting models",
                        "evicted_models": eviction_result["evicted"],
                        "gpu_memory": {
                            "used_mb": used_mb,
                            "free_mb": free_mb,
                            "needed_mb": estimated_needed_mb
                        }
                    }
                )

    # Start the container
    logger.info(f"Starting container '{container}' for model '{model_name}'")
    run_podman_command(["start", container])

    # Calculate model-size-aware timeout
    startup_timeout = estimate_model_startup_timeout(model_path)

    # Wait for model to be ready
    logger.info(f"Waiting for model '{model_name}' to be ready on port {port} (timeout: {startup_timeout}s)")
    if not wait_for_model_ready(port, timeout=startup_timeout):
        # Check container logs for better error message
        try:
            logs_result = subprocess.run(
                ["podman", "logs", "--tail", "50", container],
                capture_output=True,
                text=True,
                timeout=5
            )
            error_hints = []
            if "OOM" in logs_result.stderr or "out of memory" in logs_result.stderr.lower():
                error_hints.append("GPU OOM detected in logs")
            if "CUDA" in logs_result.stderr:
                error_hints.append("CUDA error detected")

            hint_msg = f" Hints: {', '.join(error_hints)}" if error_hints else ""
        except:
            hint_msg = ""

        raise HTTPException(
            status_code=504,
            detail=f"Model '{model_name}' start timed out after {startup_timeout}s.{hint_msg} Check container logs: podman logs {container}"
        )

    # Update last activity
    last_activity[model_name] = datetime.now()

    endpoint = f"http://akili.lab.rodhouse.net:{port}"
    logger.info(f"Model '{model_name}' is ready at {endpoint}")

    response = {
        "status": "started",
        "model": model_name,
        "container": container,
        "endpoint": endpoint,
        "health_check": f"{endpoint}/health",
        "models_endpoint": f"{endpoint}/v1/models",
        "gpu_memory": {
            "used_mb": used_mb,
            "free_mb": free_mb
        }
    }

    # Include eviction info if any models were evicted
    if policy != EvictionPolicy.MANUAL and 'eviction_result' in locals():
        response["evicted_models"] = eviction_result["evicted"]

    return response

@app.post("/models/{model_name}/stop")
async def stop_model(model_name: str):
    """
    Stop a vLLM container to free GPU memory
    """
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    config = MODELS[model_name]
    container = config["container"]

    # Check current status
    status = get_container_status(container)

    if status not in ["running", "paused"]:
        return {
            "status": "already_stopped",
            "model": model_name,
            "container": container
        }

    # Stop the container
    logger.info(f"Stopping container '{container}' for model '{model_name}'")
    run_podman_command(["stop", container])

    logger.info(f"Model '{model_name}' stopped successfully")

    return {
        "status": "stopped",
        "model": model_name,
        "container": container
    }

@app.post("/models/{model_name}/restart")
async def restart_model(model_name: str):
    """
    Restart a vLLM container (stop then start)
    """
    # Stop if running
    await stop_model(model_name)
    time.sleep(2)  # Brief pause
    # Start
    return await start_model(model_name)

@app.get("/api/gpu")
async def get_gpu_status():
    """
    Get GPU hardware status from nvidia-smi

    Returns VRAM usage, temperature, utilization, and power draw
    for all GPUs in the system.
    """
    try:
        # Query nvidia-smi for GPU stats
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 8:
                continue

            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "temperature": float(parts[2]) if parts[2] != '[N/A]' else None,
                "utilization": float(parts[3]) if parts[3] != '[N/A]' else None,
                "memory_used_mb": int(parts[4]) if parts[4] != '[N/A]' else None,
                "memory_total_mb": int(parts[5]) if parts[5] != '[N/A]' else None,
                "power_draw_w": float(parts[6]) if parts[6] != '[N/A]' else None,
                "power_limit_w": float(parts[7]) if parts[7] != '[N/A]' else None,
            })

        return {"gpus": gpus}

    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi command failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Failed to query GPU: {e.stderr}")
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for web UI (must be last, as it catches all paths)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    logger.info(f"Serving web UI from {static_dir}")
else:
    logger.warning(f"Static directory not found: {static_dir}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
