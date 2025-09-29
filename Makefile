.PHONY: data notebook agent eval_questions evals

notebook:
	uv run jupyter notebook

data:
	uv run python -m zc_agent.prepare_data

agent:
	uv run python -m zc_agent.main

eval_generate_questions:
	uv run python -m zc_agent.eval.generate_questions --sample-size 50

eval_run_agent:
	uv run python -m zc_agent.eval.run_agent

eval_calculate_metrics:
	uv run python -m zc_agent.eval.calculate_metrics