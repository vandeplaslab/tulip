from setuptools import find_packages, setup

setup(
    name="tulip",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pyclustering",
        "tifffile",
        "joblib",
        "matplotlib",
    ],
)
