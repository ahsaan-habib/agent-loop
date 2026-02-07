.PHONY: install model
install:
	python -m venv .venv && .venv/bin/pip install -e '.[docs]'
model:
	ollama pull qwen3:4b
