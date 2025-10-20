"""
vLLM On-Demand Control API

Provides scale-to-zero behavior for vLLM containers by starting
models on-demand and stopping them during idle periods.
"""

__version__ = "1.0.0"

# Main app is in app.py and imported when running as service
__all__ = ["__version__"]
