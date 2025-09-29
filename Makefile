.PHONY: data notebook

notebook:
	uv run jupyter notebook

data:
	uv run python -m zc_agent.prepare_data