import json
import asyncio

import pandas as pd

from pydantic import BaseModel
from pydantic_ai import Agent

from zc_agent.llm import read_prompt
from zc_agent.logs import ConversationJsonLogger
from zc_agent.eval.async_paralell import map_progress

logger = ConversationJsonLogger('evals/ai_logs')



class EvaluationCheck(BaseModel):
    check_name: str
    justification: str
    check_pass: bool


class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck]
    summary: str


evaluation_prompt = read_prompt('eval_checklist.md')


eval_agent = Agent(
    name='eval_agent',
    model='gpt-5-mini',
    instructions=evaluation_prompt,
    output_type=EvaluationChecklist
)


def load_log_file(log_file):
    with open(log_file, 'r') as f_in:
        log_data = json.load(f_in)
        log_data['log_file'] = log_file
        return log_data


def simplify_log_messages(messages):
    log_simplified = []

    for m in messages:
        parts = []
    
        for original_part in m['parts']:
            part = original_part.copy()
            kind = part['part_kind']
    
            if kind == 'user-prompt':
                del part['timestamp']
            if kind == 'tool-call':
                del part['tool_call_id']
            if kind == 'tool-return':
                del part['tool_call_id']
                del part['metadata']
                del part['timestamp']
            if kind == 'tool-return':
                part['content'] = 'RETURN_RESULTS_REDACTED'
            if kind == 'text':
                del part['id']
    
            parts.append(part)
    
        message = {
            'kind': m['kind'],
            'parts': parts
        }
    
        log_simplified.append(message)
    return log_simplified


def load_evaluation_set():
    """Load and filter log files for evaluation."""
    eval_set = []
    
    for log_file in logger.list_logs():
        if 'gh_agent' not in log_file.name:
            continue

        log_record = load_log_file(log_file)
        if log_record.get('source') != 'ai-generated':
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
    messages = log_record['messages']
    question = messages[0]['parts'][0]['content']
    answer = messages[-1]['parts'][0]['content']
    log_simplified = simplify_log_messages(messages)
    
    user_prompt = user_prompt_format.format(
        instructions=evaluation_prompt,
        question=question,
        answer=answer,
        log=json.dumps(log_simplified, indent=2)
    )
    
    result = await agent.run(user_prompt)
    return result.data


def process_evaluation_results(eval_results):
    """Convert evaluation results into DataFrame rows."""
    rows = []

    for log_record, eval_result in eval_results:
        messages = log_record['messages']

        row = {
            'file': log_record['log_file'].name,
            'question': messages[0]['parts'][0]['content'],
            'answer': messages[-1]['parts'][0]['content'],
        }

        checks = {c.check_name: c.check_pass for c in eval_result.checklist}
        row.update(checks)

        rows.append(row)
    
    return rows


async def run_evaluations():
    """Main function to run the evaluation process."""
    eval_set = load_evaluation_set()
    print(f"Loaded {len(eval_set)} evaluation records")
    
    eval_results = await map_progress(
        sequence=eval_set,
        function=lambda record: evaluate_log_record(eval_agent, record),
        desc="Evaluating log records",
    )

    rows = process_evaluation_results(eval_results)
    df_evals = pd.DataFrame(rows)
    
    print("Evaluation results (percentage):")
    print(df_evals.mean(numeric_only=True) * 100)
    
    return df_evals


if __name__ == "__main__":
    df_results = asyncio.run(run_evaluations())