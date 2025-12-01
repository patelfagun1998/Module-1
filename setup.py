from setuptools import setup, find_packages

setup(
    name="minitorch",
    version="0.4",
    packages=find_packages(include=["minitorch", "minitorch.*"]),
)
