"""
Entry point for running vLLM control as a module:
    python -m vllm_control
"""

import sys
import uvicorn

def main():
    """Run the vLLM Control API FastAPI service"""
    from vllm_control.app import app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )

if __name__ == "__main__":
    sys.exit(main())
