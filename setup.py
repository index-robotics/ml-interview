"""Setup script for ml-interview package."""

from os import path
from pathlib import Path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
readme_path = path.join(this_directory, "README.md")
if path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "ML Interview environments - YAM tabletop manipulation environments"

# Path to i2rt submodule - use absolute path for file:// URL
i2rt_path = Path(this_directory) / "third_party" / "i2rt"
# Convert to file:// URL format (use 3 slashes for absolute paths on Windows, 2 on Unix)
# Using pathlib.as_uri() handles this correctly
i2rt_uri = i2rt_path.resolve().as_uri()

setup(
    name="ml-interview",
    version="0.1.0",
    description="ML Interview environments - YAM tabletop manipulation environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.12",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "mujoco>=3.3.7",
        "numpy>=1.24.0",
        "robot-descriptions>=1.21.0",
        "scipy>=1.16.3",
        "gymnasium>=0.29.0",
        "trimesh>=3.20.0",
        "robosuite>=1.3.0",
        # Install i2rt from local submodule
        f"i2rt @ {i2rt_uri}",
    ],
    extras_require={
        "dev": [
            "pytest>=9.0.1",
            "ruff>=0.14.5",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

