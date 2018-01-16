import os
from setuptools import setup, find_packages


ROOT_PATH = os.path.dirname(__file__)

# Determine version from top-level package __init__.py file
with open(os.path.join(ROOT_PATH, 'slicerecon', '__init__.py')) as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break

setup(
    name="slicerecon",
    package_dir={'slicerecon': 'slicerecon'},
    packages=find_packages(),

    install_requires=[
        "transforms3d >= 0.3",
        "numpy >= *",
        "tqdm >= *"],
    version=version,
)
