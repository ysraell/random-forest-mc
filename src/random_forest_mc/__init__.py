__version__ = "1.5.0"

_PyRandomForestMC = None
_CPPRandomForestMC = None
_BaseRandomForestMC = None
_DecisionTreeMC = None
_CPP_AVAILABLE = None


def _load_backends():
    global _PyRandomForestMC, _CPPRandomForestMC, _BaseRandomForestMC, _DecisionTreeMC, _CPP_AVAILABLE
    if _PyRandomForestMC is None:
        from .model import RandomForestMC as _py
        _PyRandomForestMC = _py
    if _BaseRandomForestMC is None:
        from .forest import BaseRandomForestMC as _base
        _BaseRandomForestMC = _base
    if _DecisionTreeMC is None:
        from .tree import DecisionTreeMC as _dt
        _DecisionTreeMC = _dt
    if _CPP_AVAILABLE is None:
        try:
            from .cpp_model import RandomForestMC as _cpp
            _CPPRandomForestMC = _cpp
            _CPP_AVAILABLE = True
        except Exception:
            _CPPRandomForestMC = None
            _CPP_AVAILABLE = False


def RandomForestMC(*args, engine: str = "auto", **kwargs):
    """Factory creating a RandomForestMC instance with the requested backend.

    Args:
        engine (str, optional): Backend implementation to use:
            - 'auto': Use C++ backend if available, otherwise fallback to pure Python.
            - 'cpp' (or 'c', 'cpython'): Force the Modern C++ backend (raises ImportError if unavailable).
            - 'python' (or 'py'): Force the pure Python backend.
            Defaults to 'auto'.
    """
    _load_backends()
    engine_normalized = engine.lower() if isinstance(engine, str) else ""

    if engine_normalized == "auto":
        if _CPP_AVAILABLE and _CPPRandomForestMC is not None:
            return _CPPRandomForestMC(*args, **kwargs)
        return _PyRandomForestMC(*args, **kwargs)

    elif engine_normalized in ("cpp", "c", "cpython"):
        if not _CPP_AVAILABLE or _CPPRandomForestMC is None:
            raise ImportError(
                "Modern C++ backend (_cpp_forest) is not available. "
                "Ensure build dependencies are installed and run 'python build.py build_ext --inplace'."
            )
        return _CPPRandomForestMC(*args, **kwargs)

    elif engine_normalized in ("python", "py"):
        return _PyRandomForestMC(*args, **kwargs)

    else:
        raise ValueError(
            f"Invalid engine '{engine}'. Expected 'auto', 'cpp', or 'python'."
        )


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_backends()
    if name == "RandomForestMCPy":
        return _PyRandomForestMC
    if name in ("RandomForestMCCPP", "RandomForestMCC"):
        return _CPPRandomForestMC
    if name == "BaseRandomForestMC":
        return _BaseRandomForestMC
    if name == "DecisionTreeMC":
        return _DecisionTreeMC
    if name == "CPP_AVAILABLE":
        return _CPP_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "RandomForestMC",
    "RandomForestMCPy",
    "RandomForestMCCPP",
    "RandomForestMCC",
    "BaseRandomForestMC",
    "DecisionTreeMC",
    "CPP_AVAILABLE",
]

# EOF
