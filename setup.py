from setuptools import find_packages, setup


setup(
    name="wmdetection",
    version="0.1.0",
    description="Simple watermark detection.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["torch", "numpy", "timm", "huggingface_hub"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved ::Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
