"""
src/ffi/__init__.py

Python FFI bridge to the Go Cambia game engine (libcambia.so).
Exports GoEngine and GoAgentState as drop-in replacements for the
Python CambiaGameState and AgentState used in CFR training.
"""

from .bridge import GoAgentState, GoEngine

__all__ = ["GoEngine", "GoAgentState"]
