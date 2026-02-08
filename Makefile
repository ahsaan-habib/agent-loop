.PHONY: install model
install:
	python -m venv .venv && .venv/bin/pip install -e '.[docs]'
model:
	ollama pull qwen3:4b-instruct   # not plain qwen3:4b: that tag is now a thinking-only build

test:
	pytest -q
