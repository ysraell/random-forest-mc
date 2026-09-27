import os
import sys
from setuptools import Extension
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Allows compilation to fail without failing package installation."""

    def run(self):
        try:
            super().run()
        except Exception as e:
            print(f"\n[WARNING] Building C++ extension failed: {e}\nFalling back to pure Python engine.\n")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"\n[WARNING] Building extension {ext.name} failed: {e}\nFalling back to pure Python engine.\n")


def build(setup_kwargs):
    try:
        import nanobind
    except ImportError:
        return

    nanobind_include = nanobind.include_dir()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(base_dir, "src", "random_forest_mc", "cpp")

    if sys.platform == "win32":
        extra_compile_args = ["/std:c++17", "/O2"]
        extra_link_args = []
    elif sys.platform == "darwin":
        extra_compile_args = ["-std=c++17", "-O3", "-mmacosx-version-min=10.14"]
        extra_link_args = []
    else:
        extra_compile_args = ["-std=c++17", "-O3", "-fvisibility=hidden"]
        extra_link_args = []

    robin_map_include = os.path.join(os.path.dirname(nanobind_include), "ext", "robin_map", "include")

    ext = Extension(
        "random_forest_mc._cpp_forest",
        sources=[
            os.path.join(nanobind.source_dir(), "nb_combined.cpp"),
            os.path.join(cpp_dir, "bindings.cpp"),
            os.path.join(cpp_dir, "forest.cpp"),
            os.path.join(cpp_dir, "tree.cpp"),
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
