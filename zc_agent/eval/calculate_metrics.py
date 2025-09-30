import json
import asyncio

import pandas as pd

from enum import Enum
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from zc_agent.llm import read_prompt
from zc_agent.logs import ConversationJsonLogger
from zc_agent.eval.async_paralell import map_progress


logger = ConversationJsonLogger("evals/ai_logs")


class CheckName(str, Enum):
    instructions_follow = "instructions_follow"
    instructions_avoid = "instructions_avoid" 
    answer_relevant = "answer_relevant"
    answer_clear = "answer_clear"
    answer_citations = "answer_citations"
    completeness = "completeness"
    tool_call_search = "tool_call_search"


CHECK_DESCRIPTIONS = {
    CheckName.instructions_follow: "The agent followed the user's instructions (in <INSTRUCTIONS>)",
    CheckName.instructions_avoid: "The agent avoided doing things it was told not to do",
    CheckName.answer_relevant: "The response directly addresses the user's question",
    CheckName.answer_clear: "The answer is clear and correct",
    CheckName.answer_citations: "The response includes proper citations or sources when required",
    CheckName.completeness: "The response is complete and covers all key aspects of the request",
    CheckName.tool_call_search: "Is the search tool invoked?"
}


class EvaluationCheck(BaseModel):
    check_name: CheckName = Field(description="The type of evaluation check")
    # reasoning: str = Field(description="The reasoning behind the check result")
    check_pass: bool = Field(description="Whether the check passed (True) or failed (False)")
    
class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck] = Field(description="List of all evaluation checks")


def generate_checklist_text():
    """Generate the checklist text from the enum and descriptions."""
    checklist_items = []
    for check_name in CheckName:
        description = CHECK_DESCRIPTIONS[check_name]
        checklist_items.append(f"- {check_name.value}: {description}")
    return "\n".join(checklist_items)


def load_log_file(log_file):
    with open(log_file, "r") as f_in:
        log_data = json.load(f_in)
        log_data["log_file"] = log_file
        return log_data


def simplify_log_messages(messages):
    log_simplified = []

    for m in messages:
        parts = []

        for original_part in m["parts"]:
            part = original_part.copy()
            kind = part["part_kind"]

            if kind == "user-prompt":
                del part["timestamp"]
            if kind == "tool-call":
                del part["tool_call_id"]
            if kind == "tool-return":
                del part["tool_call_id"]
                del part["metadata"]
                del part["timestamp"]
            if kind == "tool-return":
                part["content"] = "RETURN_RESULTS_REDACTED"
            if kind == "text":
                del part["id"]

            parts.append(part)

        message = {"kind": m["kind"], "parts": parts}

        log_simplified.append(message)
    return log_simplified


def load_evaluation_set():
    """Load and filter log files for evaluation."""
    eval_set = []

    for log_file in logger.list_logs():
        if "gh_agent" not in log_file.name:
            continue

        log_record = load_log_file(log_file)
        if log_record.get("source") != "ai-generated":
            continue

        eval_set.append(log_record)

    return eval_set


user_prompt_format = """
<INSTRUCTIONS>{instructions}</INSTRUCTIONS>
<QUESTION>{question}</QUESTION>
<ANSWER>{answer}</ANSWER>
<LOG>{log}</LOG>
""".strip()


async def evaluate_log_record(agent, log_record):
    """Evaluate a single log record using the evaluation agent."""
    messages = log_record["messages"]
    system_prompt = log_record['system_prompt']
    question = messages[0]["parts"][0]["content"]
    answer = messages[-1]["parts"][0]["content"]
    log_simplified = simplify_log_messages(messages)

    user_prompt = user_prompt_format.format(
        instructions=system_prompt,
        question=question,
        answer=answer,
        log=json.dumps(log_simplified, indent=2),
    )

    result = await agent.run(user_prompt)
    return result


def process_evaluation_results(eval_results):
    """Convert evaluation results into DataFrame rows."""
    rows = []

    for log_record, result in eval_results:
        eval_result = result.output
        usage = result.usage()

        # Calculate cost from usage
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0

        # gpt-5-mini pricing: input $0.250/1M tokens, output $2.000/1M tokens
        # gpt-5-nano pricing: input $0.050/1M tokens, output $0.400/1M tokens

        input_cost = (input_tokens / 1_000_000) * 0.050
        output_cost = (output_tokens / 1_000_000) * 0.400
        total_cost = input_cost + output_cost

        row = {
            "file": log_record["log_file"].name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

        checks = {c.check_name: c.check_pass for c in eval_result.checklist}
        row.update(checks)

        rows.append(row)

    return rows


async def run_evaluations():
    """Main function to run the evaluation process."""
    eval_set = load_evaluation_set()
    # eval_set = eval_set[:2]
    print(f"Loaded {len(eval_set)} evaluation records")

    evaluation_prompt_template = read_prompt("eval_checklist.md")
    evaluation_prompt = evaluation_prompt_template.format(
        checklist=generate_checklist_text()
    )

    eval_agent = Agent(
        name="eval_agent",
        model="gpt-5-nano",
        instructions=evaluation_prompt,
        output_type=EvaluationChecklist,
    )

    async def evaluate_with_record(record):
        result = await evaluate_log_record(eval_agent, record)
        return record, result

    eval_results = await map_progress(
        sequence=eval_set,
        function=evaluate_with_record,
        desc="Evaluating log records",
    )

    rows = process_evaluation_results(eval_results)
    df_evals = pd.DataFrame(rows)

    print(df_evals.head())

    # Print token usage and cost summary
    total_input_tokens = df_evals["input_tokens"].sum()
    total_output_tokens = df_evals["output_tokens"].sum()
    total_tokens = df_evals["total_tokens"].sum()
    total_cost = df_evals["total_cost"].sum()

    print("\nToken Usage & Cost Summary:")
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total cost: ${total_cost:.4f}")

    print("\nEvaluation results (percentage):")
    eval_columns = [check_name.value for check_name in CheckName]
    print(f"Evaluation columns: {eval_columns}")

    results = df_evals[eval_columns].mean() * 100
    # Display with clean column names
    for col in eval_columns:
        print(f"{col:<25} {results[col]:>6.1f}%")

    return df_evals


if __name__ == "__main__":
    df_results = asyncio.run(run_evaluations())
    df_results.to_csv("evals/evaluation_results.csv", index=False)
