import os
import setuptools

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _parse_requirements(file):
    with open(os.path.join(_CURRENT_DIR, file)) as f:
        return [line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))]


setuptools.setup(
    name="stlcg",
    version="0.0.1",
    author="Karen Leung and Nikos Arechiga",
    author_email="karen.ym.leung@gmail.com and nikos.arechiga@tri.global",
    description="A toolbox to compute the robustness of STL formulas using computations graphs (PyTorch).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StanfordASL/stlcg",
    packages=setuptools.find_packages(include=['stlcg', 'stlcg.*']),
    install_requires=_parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
