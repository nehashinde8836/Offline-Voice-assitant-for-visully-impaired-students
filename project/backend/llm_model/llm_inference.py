"""
llm_inference.py
----------------
Compatibility shim — re-exports everything from inference.py.
Existing code that imports from llm_inference continues to work unchanged.
"""
from backend.llm_model.inference import (   # noqa: F401
    generate_response,
    explain,
    story,
    pythagoras_explain,
    pythagoras_story,
    pythagoras_example,
    _get_tutor,
)
