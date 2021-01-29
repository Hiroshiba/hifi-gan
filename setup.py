from setuptools import find_packages, setup

setup(
    name="hifi_gan",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "numba",
        "librosa",
        "scipy",
        "tensorboard",
        "soundfile",
        "matplotlib",
        "pillow",
    ],
)
