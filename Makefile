.PHONY: install test lint validate dry-run run summarize clean

install:
	uv sync --group dev

test:
	uv run pytest -q

lint:
	uv run ruff check .

# Validate the CEO/CIS datasets load and their distributions are consistent.
validate:
	uv run python -c "from political_alignment.dataset import load_datasets; \
	items = load_datasets(['data/ceo_items.csv','data/cis_items.csv']); \
	print('OK', len(items), 'items')"

# Network-free end-to-end run using the MockProvider, then summarise.
dry-run:
	uv run python scripts/model.py --model mock --mock
	uv run python scripts/summarize_results.py

# Real run over every model in models.yaml (needs API keys / Ollama).
run:
	uv run python scripts/run_evals.py

summarize:
	uv run python scripts/summarize_results.py

clean:
	rm -rf site evals/*.json
	rm -rf .pytest_cache .ruff_cache
