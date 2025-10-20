# RHDC vLLM Control

vLLM On-Demand Control API for managing vLLM container lifecycles with scale-to-zero behavior.

## Overview

Provides on-demand model loading by starting vLLM containers when needed and stopping them during idle periods to conserve GPU memory.

## Features

- Start/stop vLLM containers via REST API
- Health check and status monitoring
- Podman container integration
- Scale-to-zero behavior
- FastAPI with OpenAPI documentation

## Installation

### From GitHub

```bash
pip install git+https://github.com/flyemsafe/rhdc-vllm-control.git
```

### From Source

```bash
git clone https://github.com/flyemsafe/rhdc-vllm-control.git
cd rhdc-vllm-control
pip install -e .
```

## Usage

### Run as Service

```bash
# Run directly
vllm-control

# Or as Python module
python -m vllm_control
```

## API Endpoints

When running as a service (port 8100):

- **GET /models** - List available models
- **POST /models/{name}/start** - Start model container
- **POST /models/{name}/stop** - Stop model container
- **GET /models/{name}/status** - Get model status
- **GET /health** - Service health check

### Example API Calls

```bash
# List available models
curl http://localhost:8100/models | jq

# Start a model
curl -X POST http://localhost:8100/models/qwen25-coder-32b-awq/start

# Check model status
curl http://localhost:8100/models/qwen25-coder-32b-awq/status | jq

# Stop a model
curl -X POST http://localhost:8100/models/qwen25-coder-32b-awq/stop
```

## Configuration

The vLLM control reads configuration from:
- Environment variables
- YAML configuration file

### Environment Variables

- `VLLM_CONTROL_PORT` - API port (default: 8100)
- `VLLM_CONTROL_HOST` - Bind host (default: 0.0.0.0)
- `VLLM_CONTROL_CONFIG` - Path to models.yaml (default: /usr/local/etc/vllm/models.yaml)

### Models Configuration

Create `/usr/local/etc/vllm/models.yaml`:

```yaml
models:
  - name: qwen25-coder-32b-awq
    container: vllm-qwen25-coder-32b-awq
    port: 8010

  - name: deepseek-v2-33b-awq
    container: vllm-deepseek-v2-33b-awq
    port: 8011
```

## Architecture

```
┌──────────────────────────────────────────────┐
│ vLLM Control API (Port 8100)                 │
├──────────────────────────────────────────────┤
│ Features:                                    │
│   - Start/stop vLLM containers              │
│   - Health monitoring                        │
│   - Status tracking                          │
├──────────────────────────────────────────────┤
│ Integration:                                 │
│   - Podman CLI                               │
│   - vLLM containers                          │
└──────────────────────────────────────────────┘
```

## Requirements

- Python 3.9+
- Podman installed
- vLLM containers created
- YAML configuration file

## Development

```bash
# Clone repo
git clone https://github.com/flyemsafe/rhdc-vllm-control.git
cd rhdc-vllm-control

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Deployment

For production deployment with Ansible, see:
- [rhdc.ai.vllm_control](https://github.com/flyemsafe/rhdc-ansible) Ansible role

The Ansible role handles:
- Pip installation from GitHub
- Systemd user service configuration
- Firewall rules
- Service monitoring

## License

MIT

## Related Projects

- [rhdc-gpu-manager](https://github.com/flyemsafe/rhdc-gpu-manager) - GPU resource coordination
- [rhdc-ai-services](https://github.com/flyemsafe/rhdc-ai-services) - FastAPI microservices

## Issues

Report issues at: https://github.com/flyemsafe/rhdc-vllm-control/issues
