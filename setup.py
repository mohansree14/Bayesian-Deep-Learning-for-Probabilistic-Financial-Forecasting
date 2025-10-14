"""Setup script for Financial Time Series Forecasting project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="financial-ts-forecasting",
    version="1.0.0",
    author="Mohansree Vijayakumar",
    author_email="mohansreesk14@gmail.com",
    description="Financial Time Series Forecasting with Uncertainty Quantification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ML-Intern",
    packages=find_packages(exclude=["tests", "tests.*", "experiments", "experiments.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
    },
    entry_points={
        "console_scripts": [
            "financial-ts-train=train:main",
            "financial-ts-evaluate=evaluate:main",
            "financial-ts-backtest=backtest:main",
            "financial-ts-hpo=hparam_search:main",
            "financial-ts-app=run_project:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "README.md", "LICENSE"],
    },
    keywords=[
        "finance",
        "time-series",
        "forecasting",
        "deep-learning",
        "lstm",
        "bayesian",
        "uncertainty-quantification",
        "stock-prediction",
        "pytorch",
    ],
    project_urls={
        "Documentation": "https://github.com/yourusername/ML-Intern#readme",
        "Source": "https://github.com/yourusername/ML-Intern",
        "Tracker": "https://github.com/yourusername/ML-Intern/issues",
    },
)
