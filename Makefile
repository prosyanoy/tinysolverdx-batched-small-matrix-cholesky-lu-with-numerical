# Convenience targets. Override PY / CXX / JOBS as needed:  make build CXX=clang++
PY ?= python3
JOBS ?= 4
VENV := .venv
BIN := $(VENV)/bin
export PYTHONPATH := $(CURDIR)/python

.PHONY: setup build test test-cpp test-py report bench clean

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -r requirements.txt cmake ninja

build:
	$(BIN)/cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(CURDIR)/$(BIN)/python
	$(BIN)/cmake --build build -j$(JOBS)

test: test-cpp test-py

test-cpp:
	$(BIN)/ctest --test-dir build --output-on-failure

test-py:
	$(BIN)/python -m pytest -q

report:   # error-vs-condition-number / size sweeps -> docs/report
	$(BIN)/python -m tinysolverdx.lab

bench:    # throughput + roofline charts -> docs/report
	$(BIN)/python -m tinysolverdx.bench
	$(BIN)/python -m tinysolverdx.bench --precision fp64

clean:
	rm -rf build python/tinysolverdx/_core*.so
