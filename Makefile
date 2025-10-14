.PHONY: help install install-dev clean test lint format run train evaluate backtest hpo figures all

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo "  make test         - Run unit tests"
	@echo "  make lint         - Run code linting (flake8)"
	@echo "  make format       - Format code with black and isort"
	@echo "  make run          - Run Streamlit application"
	@echo "  make train        - Train models"
	@echo "  make evaluate     - Evaluate models"
	@echo "  make backtest     - Run backtesting"
	@echo "  make hpo          - Run hyperparameter optimization"
	@echo "  make figures      - Generate all documentation figures"
	@echo "  make all          - Run complete pipeline (train + evaluate + figures)"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy isort

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +

test:
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ utils/ --line-length=100
	isort src/ tests/ utils/ --profile black

run:
	python run_project.py

train:
	python train.py

train-all:
	python train_all_tickers.py

train-bnn:
	python train_bnn.py

evaluate:
	python evaluate.py

backtest:
	python backtest.py

hpo:
	python hparam_search.py

figures:
	python utils/generate_figures.py
	python utils/generate_hpo_table.py
	python utils/generate_indicators_table.py

all: train evaluate figures
	@echo "Complete pipeline executed successfully!"

docker-build:
	docker build -t financial-ts-forecasting .

docker-run:
	docker run -p 8501:8501 financial-ts-forecasting

setup:
	pip install -e .

package:
	python setup.py sdist bdist_wheel

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*
