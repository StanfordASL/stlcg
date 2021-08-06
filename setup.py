import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stlcg",
    version="0.0.1",
    author="Karen Leung and Nikos Arechiga",
    author_email="karen.ym.leung@gmail.com and nikos.arechiga@tri.global",
    description="A toolbox to compute the robustness of STL formulas using computations graphs (PyTorch).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StanfordASL/stlcg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

