from setuptools import setup, find_packages

setup(
    name="dawn-spec",
    version="0.1.0",
    description="Implementation of the DAWN/AGNTCY specification for agent interoperability",
    author="DAWN Spec Contributors",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "google-generativeai>=0.3.0",
        "PyGithub>=1.59.0",
        "arxiv>=1.4.7",
        "duckduckgo-search>=3.9.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
)