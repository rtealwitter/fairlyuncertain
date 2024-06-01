import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fairlyuncertain",
    version="0.0.1",
    author="R. Teal Witter and Lucas Rosenblatt",
    author_email="rtealwitter@gmail.com, lr2872@nyu.edu",
    description="Heteroscedastic uncertainty estimates for fair algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rtealwitter/fairlyuncertain",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
