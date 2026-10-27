.PHONY: help test check build bench clean

PYTHON ?= python3
CARGO ?= cargo

help:
	@echo "Available targets:"
	@echo "  make test   - Run shape parity unit tests (Python + Rust)"
	@echo "  make check  - Validate that README table matches results/latest.json"
	@echo "  make build  - Compile the Rust Candle release binary"
	@echo "  make bench  - Run the full 5-trial benchmark shootout"
	@echo "  make clean  - Clean build artifacts"

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"
	$(CARGO) test --manifest-path rust_candle/Cargo.toml

check:
	$(PYTHON) scripts/check_readme.py

build:
	$(CARGO) build --release --manifest-path rust_candle/Cargo.toml

bench: build
	$(PYTHON) benchmark_all.py --trials 5

clean:
	$(CARGO) clean --manifest-path rust_candle/Cargo.toml
	find . -type d -name "__pycache__" -exec rm -rf {} +
