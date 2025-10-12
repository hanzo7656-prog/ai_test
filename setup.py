from setuptools import setup, find_packages

setup(
    name="vortex-ai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask==2.3.3",
        "psutil==5.9.5", 
        "requests==2.31.0",
        "numpy==1.24.3"
    ],
)
