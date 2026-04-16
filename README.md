# FruitNInja Refactoring - The Simpler FruitNinja!

Welcome! This repository is a lightweight, refactored version of the CVPR 2025 paper, [FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting](https://arxiv.org/abs/2411.12089).
While the original codebase is tightly integrated with PhysGaussian, this project aims to isolate FruitNinja's core features. By removing unused dependencies and refactoring the pipeline, this repo resolves potential "dependency hell" and offers a cleaner, ready-to-use environment.
Feel free to use this streamlined codebase for your own research. I am always open to feedback!

## How to install Dependency

If you use ```uv```, you can just simply use
```bash
uv sync
```
If you want to use ```pip```, no worries! It is way simple, too!
```bash
pip install .
```

## How to use FruitNinja