# Blueprint: Adding a Modern C++ (C++17 + nanobind) Backend to a Python Package

This document serves as a complete handoff guide and architectural reference for engineering teams looking to add a high-performance **Modern C++** backend to an existing Python package while preserving a **100% pure-Python fallback**, achieving zero-copy NumPy execution, and maintaining identical APIs and serialization parity.

---

## 1. Executive Summary & Why Modern C++ with `nanobind`

When accelerating Python packages (especially machine learning, scientific computing, or heavy data-processing libraries), developers often face the dilemma: **Pure C (CPython C-API)** vs. **Cython** vs. **pybind11** vs. **nanobind**.

### Why `nanobind` is the Modern Standard:
- **Binary Size**: Generates `.so` / `.pyd` files that are **~5x–10x smaller** than `pybind11` (often < 200KB vs ~1MB+).
- **Compilation Speed**: Compiles **3x–4x faster** than `pybind11` and avoids heavy template-instantiation overhead.
- **Memory & Overhead**: Minimal runtime overhead, clean zero-copy NumPy array integration (`nb::ndarray`), and direct buffer protocol support.
- **Future-Proof**: Native support for **Python 3.12, 3.13, and 3.14**, including experimental **free-threaded CPython (nogil / PEP 703)**.
- **Versus Cython**: Cython is a domain-specific dialect that compiles to generated C/C++. Writing clean C++17 gives you access to modern standard library algorithms (`std::nth_element`, `std::mt19937_64`, `std::vector`), first-class IDE support (`clangd`), native unit test frameworks (Catch2/GTest), and full separation of algorithmic concerns from Python binding boilerplate.

---

## 2. Dual-Engine Architecture Pattern

To give users full flexibility and ensure installation on any machine (even those without a C++ compiler), maintain dual engines behind a unified interface:

```
                        +---------------------------------------------+
                        |         from my_package import Model        |
                        |         (engine="auto"|"cpp"|"python")      |
                        +----------------------+----------------------+
                                               |
                     +-------------------------+-------------------------+
                     |                                                   |
                     v                                                   v
          engine="python"                                     engine="cpp"
    +------------------------------+                    +------------------------------+
    |    my_package.model.Model    |                    |  my_package.cpp_model.Model  |
    |      (Pure Python / NumPy)   |                    |    (Python wrapper layer)    |
    +------------------------------+                    +--------------+---------------+
                     |                                                 |
                     |                                                 v
                     |                                  +------------------------------+
                     |                                  |   _cpp_module (nanobind)     |
                     |                                  +--------------+---------------+
                     |                                                 |
                     |                                                 v
                     |                                  +------------------------------+
                     |                                  |   C++17 Algorithmic Core     |
                     |                                  | - Cache-friendly structs     |
                     |                                  | - std::thread worker pool    |
                     |                                  +--------------+---------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Exact Same Serialized      |
                              |    Dict / JSON Schema         |
                              |  (model2dict() / dict2model())|
                              +-------------------------------+
```

### Key API Principles:
1. **`engine="auto"` (default)**: Uses C++ if available; seamlessly falls back to pure Python if compiled extensions are absent.
2. **Explicit Options (`engine="cpp"` or `engine="python"`)**: Allows users and benchmarking pipelines to force an exact backend.
3. **Dedicated Submodules**: Always provide direct access to each engine:
   - `from my_package.model import Model as PyModel`
   - `from my_package.cpp_model import Model as CppModel`
4. **100% Bidirectional Serialization**: A model trained in C++ can be serialized (e.g. `model2dict()`) and loaded into Python (`dict2model()`), and vice versa, with identical inference outputs.

---

## 3. Directory Layout

For packages using the standard `src/` layout with Poetry:

```text
my-package/
├── pyproject.toml              # Build-system requirements & Poetry config
├── build.py                    # Setuptools build script invoked by Poetry
├── src/
│   └── my_package/
│       ├── __init__.py         # Unified factory & lazy backend loader
│       ├── model.py            # Pure Python implementation
│       ├── cpp_model.py        # Python wrapper around the compiled C++ extension
│       ├── utils.py
│       └── cpp/                # Pure C++ sources
│           ├── types.hpp       # Enums and core types
│           ├── core.hpp        # Algorithmic header
│           ├── core.cpp        # Algorithmic C++ implementation
│           └── bindings.cpp    # nanobind module definitions
└── tests/
    ├── test_python_model.py
    └── test_cpp_model.py
```

---

## 4. C++ Algorithmic Best Practices for 100x Speedup

### A. Flatten Data Structures for Cache Locality
- **Python**: Often uses recursive object trees or nested dictionaries (e.g., `{"split": {">=": ..., "<": ...}}`). Navigating Python dictionaries requires hash lookups and heap dereferences.
- **C++**: Store graphs/trees in flat `std::vector<Node>` arrays using integer indices (`int32_t left_child`, `int32_t right_child`). Traversal is a sequential CPU cache-line read without dynamic heap allocation.

### B. Use $O(N)$ Quickselect (`std::nth_element`) Instead of Sorting
- In decision trees, finding median splits for numeric features in Python typically invokes `np.quantile` or full array sorting ($O(N \log N)$).
- In C++, use `std::nth_element` which runs in **$O(N)$ linear time**, partitioning the array in-place:
  ```cpp
  std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
  double median = vals[mid];
  ```

### C. True Concurrency without GIL or OpenMP Dependencies
- **Avoid OpenMP if cross-platform friction is an issue**: OpenMP on macOS requires external `libomp` (via Homebrew), causing build failures for users without development tooling.
- **Use Standard C++17 Threads**: `std::thread` and an atomic task queue (`std::atomic<size_t>`) provide lightweight, zero-dependency parallel processing across Linux, Windows, and macOS:
  ```cpp
  std::atomic<size_t> next_task(0);
  std::vector<std::thread> workers;
  for (int t = 0; t < n_threads; ++t) {
      workers.emplace_back([&]() {
          while (true) {
              size_t task_id = next_task.fetch_add(1);
              if (task_id >= total_tasks) break;
              process_task(task_id);
          }
      });
  }
  for (auto& w : workers) w.join();
  ```

### D. Release the GIL During Computation
Always release the Python Global Interpreter Lock before entering compute-intensive loops in C++:
```cpp
// In bindings.cpp
.def("fit", [](MyModel& self, nb::ndarray<const double, nb::c_contig> X, nb::ndarray<const int32_t, nb::c_contig> y) {
    nb::gil_scoped_release release;  // GIL released! Other Python threads can run concurrently.
    self.fit(X.data(), y.data(), X.shape(0), X.shape(1));
})
```

---

## 5. Critical Findings & Lessons Learned

### Finding 1: Circular Imports with `__init__.py` Lazy Loading
- **The Issue**: When `__init__.py` imports `model.py`, and internal modules (like `tree.py`) import `from .__init__ import __version__`, Python's import machinery intercepts `from .__init__` as an attribute lookup on `sys.modules[pkg]`. This calls `__getattr__("__init__")`, which can trigger premature submodule loading and cause:
  `ImportError: cannot import name 'Model' from partially initialized module ...`
- **The Fix**: In `__init__.py`, always bypass dunder attributes (`name.startswith("__")`) before loading backends:
  ```python
  def __getattr__(name: str):
      if name.startswith("__"):
          raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
      _load_backends()
      ...
  ```

### Finding 2: Missing nanobind STL Headers
- **The Issue**: When casting Python strings or lists in C++ (e.g. `nb::cast<std::string>(val)` or `nb::cast<std::vector<T>>(val)`), nanobind will throw `std::bad_cast` at runtime if the respective STL header is not included in the compilation unit.
- **The Fix**: Always explicitly include the nanobind STL headers in your C++ header files whenever standard library containers are handled:
  ```cpp
  #include <nanobind/nanobind.h>
  #include <nanobind/stl/string.h>
  #include <nanobind/stl/vector.h>
  #include <nanobind/stl/unordered_map.h>
  #include <nanobind/stl/map.h>
  ```

### Finding 3: Nanobind Internal Sources & `robin_map`
- **The Issue**: `nanobind` is not header-only; it requires linking against its runtime library. Linking errors such as `undefined reference to nanobind::detail::...` occur if its internal runtime source is missing. Furthermore, compilation fails with `tsl/robin_map.h: No such file or directory` if nanobind's vendor include directory is omitted.
- **The Fix**: In `build.py`, include `nb_combined.cpp` in `sources` and add `robin_map` to `include_dirs`:
  ```python
  import nanobind

  nanobind_include = nanobind.include_dir()
  robin_map_include = os.path.join(os.path.dirname(nanobind_include), "ext", "robin_map", "include")

  ext = Extension(
      "my_package._cpp_module",
      sources=[
          os.path.join(nanobind.source_dir(), "nb_combined.cpp"),
          # ... your C++ files ...
      ],
      include_dirs=[nanobind_include, robin_map_include, cpp_dir],
      language="c++",
      extra_compile_args=["-std=c++17", "-O3"],
  )
  ```

### Finding 4: NumPy Scalar Types vs. Native Python Floats in C++
- **The Issue**: Python models often serialize dictionaries where numbers are NumPy scalars (`np.float64`, `np.int32`). When nanobind's `nb::cast<double>` receives a `numpy.float64` handle, it checks `PyFloat_Check()`, which is `false` because NumPy scalars are custom C extension types. This raises `std::bad_cast`.
- **The Fix**: Sanitize dictionaries before passing them to C++ deserializers:
  ```python
  def _clean_dict(obj):
      if isinstance(obj, dict):
          return {k: _clean_dict(v) for k, v in obj.items()}
      elif isinstance(obj, list):
          return [_clean_dict(v) for v in obj]
      elif hasattr(obj, "item"): # Converts np.float64, np.int64 to native float/int
          return obj.item()
      return obj
  ```

### Finding 5: In-Place Build Directory for `src/` Layout
- **The Issue**: When running `python build.py build_ext --inplace` in a project with a `src/my_package` layout, setuptools attempts to copy the `.so` to `my_package/` in the project root, failing with `No such file or directory`.
- **The Fix**: Add `"package_dir": {"": "src"}` to `setup_kwargs` in `build.py`:
  ```python
  setup_kwargs.update({
      "ext_modules": [ext],
      "package_dir": {"": "src"},
  })
  ```

### Finding 6: PyPI Binary Wheel Compatibility (`auditwheel` + `patchelf`)
- **The Issue**: Building wheels on Linux produces `my_package-...-linux_x86_64.whl`. PyPI **strictly rejects** generic `linux_x86_64` wheels, requiring a `manylinux` tag.
- **The Fix**: Use `auditwheel` (with `patchelf`) to automatically inspect libc version references and repair the wheel:
  ```bash
  pip install auditwheel patchelf
  auditwheel repair dist/my_package-*.whl -w dist/
  rm dist/*-linux_x86_64.whl
  ```

---

## 6. Complete Implementation Templates

### A. `build.py` (Poetry & Setuptools Integration with Fallback)

```python
import os
import sys
from setuptools import Extension
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Allows build to fail without aborting installation on pure Python systems."""

    def run(self):
        try:
            super().run()
        except Exception as e:
            print(f"\n[WARNING] Building C++ extension failed: {e}\nFalling back to pure Python backend.\n")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"\n[WARNING] Building extension {ext.name} failed: {e}\nFalling back to pure Python backend.\n")


def build(setup_kwargs):
    try:
        import nanobind
    except ImportError:
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(base_dir, "src", "my_package", "cpp")
    nanobind_include = nanobind.include_dir()
    robin_map_include = os.path.join(os.path.dirname(nanobind_include), "ext", "robin_map", "include")

    if sys.platform == "win32":
        extra_compile_args = ["/std:c++17", "/O2"]
        extra_link_args = []
    elif sys.platform == "darwin":
        extra_compile_args = ["-std=c++17", "-O3", "-mmacosx-version-min=10.14"]
        extra_link_args = []
    else:
        extra_compile_args = ["-std=c++17", "-O3", "-fvisibility=hidden"]
        extra_link_args = []

    ext = Extension(
        "my_package._cpp_module",
        sources=[
            os.path.join(nanobind.source_dir(), "nb_combined.cpp"),
            os.path.join(cpp_dir, "bindings.cpp"),
            os.path.join(cpp_dir, "core.cpp"),
        ],
        include_dirs=[nanobind_include, robin_map_include, cpp_dir],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    setup_kwargs.update({
        "ext_modules": [ext],
        "cmdclass": {"build_ext": OptionalBuildExt},
        "package_dir": {"": "src"},
    })


if __name__ == "__main__":
    from setuptools import setup
    kwargs = {}
    build(kwargs)
    setup(**kwargs)
```

### B. `pyproject.toml` Configuration (Poetry)

```toml
[tool.poetry.build]
script = "build.py"
generate-setup-file = true

include = [
    "LICENSE",
    "src/my_package/cpp/*",
]

[build-system]
requires = ["poetry-core>=1.0.0", "setuptools>=65.0", "nanobind>=2.0.0"]
build-backend = "poetry.core.masonry.api"
```

### C. `__init__.py` Lazy Dispatcher

```python
__version__ = "1.0.0"

_PyModel = None
_CppModel = None
_CPP_AVAILABLE = None


def _load_backends():
    global _PyModel, _CppModel, _CPP_AVAILABLE
    if _PyModel is None:
        from .model import Model as _py
        _PyModel = _py
    if _CPP_AVAILABLE is None:
        try:
            from .cpp_model import Model as _cpp
            _CppModel = _cpp
            _CPP_AVAILABLE = True
        except Exception:
            _CppModel = None
            _CPP_AVAILABLE = False


def Model(*args, engine: str = "auto", **kwargs):
    """Unified entry point dispatching to C++ or pure-Python backend."""
    _load_backends()
    engine_norm = engine.lower() if isinstance(engine, str) else ""

    if engine_norm == "auto":
        if _CPP_AVAILABLE and _CppModel is not None:
            return _CppModel(*args, **kwargs)
        return _PyModel(*args, **kwargs)

    elif engine_norm in ("cpp", "c"):
        if not _CPP_AVAILABLE or _CppModel is None:
            raise ImportError("C++ extension is not available.")
        return _CppModel(*args, **kwargs)

    elif engine_norm in ("python", "py"):
        return _PyModel(*args, **kwargs)

    raise ValueError(f"Invalid engine '{engine}'. Expected 'auto', 'cpp', or 'python'.")


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_backends()
    if name == "ModelPy":
        return _PyModel
    if name == "ModelCPP":
        return _CppModel
    if name == "CPP_AVAILABLE":
        return _CPP_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

---

## 7. Step-by-Step Workflow for New Packages

1. **Keep Existing Python Code Intact**: Use it as the gold-standard ground truth for equivalence testing.
2. **Implement C++ Core (`src/<pkg>/cpp/`)**:
   - Write standard C++ classes using standard containers (`std::vector`, `std::string`).
   - Keep C++ free of direct Python dependencies; only expose bindings in `bindings.cpp`.
3. **Write `build.py`**:
   - Add `nanobind` include dirs, runtime sources, and C++17 flags.
   - Implement `OptionalBuildExt` so pure-Python fallback remains functional.
4. **Compile In-Place**:
   ```bash
   python build.py build_ext --inplace
   ```
5. **Implement Python Wrapper (`src/<pkg>/cpp_model.py`)**:
   - Convert Pandas/NumPy structures into contiguous C-ordered arrays (`np.ascontiguousarray(..., dtype=np.float64)`).
   - Sanitize dictionary keys/values before deserialization.
6. **Implement Equivalence Tests**:
   - Create tests comparing predictions from `Model(engine="python")` and `Model(engine="cpp")`.
   - Verify bidirectional serialization (`model2dict` $\leftrightarrow$ `dict2model`).
7. **Build & Publish**:
   ```bash
   poetry build
   auditwheel repair dist/*.whl -w dist/
   rm dist/*-linux_x86_64.whl
   poetry publish -u __token__ -p <YOUR_PYPI_TOKEN>
   ```
