import json
import asyncio
from typing import Dict

import pandas as pd

from pydantic import BaseModel
from pydantic_ai import Agent

from zc_agent.llm import read_prompt
from zc_agent.logs import ConversationJsonLogger
from zc_agent.eval.async_paralell import map_progress

from zc_agent.main import initialize_index, initialize_agent
from zc_agent.eval.async_paralell import map_progress


def load_questions():
    df = pd.read_csv('evals/generated_questions.csv')
    records = df.to_dict(orient='records')
    return records


async def run_and_log(agent: Agent, question: Dict, logger: ConversationJsonLogger):
    result = await agent.run(question['question'])

    log_file = logger.log(
        agent=agent,
        messages=result.new_messages(),
        source='ai-generated',
        extra=question
    )

    return log_file


async def run_agent(questions):
    index = initialize_index()
    agent = initialize_agent(index)
    logger = ConversationJsonLogger("evals/ai_logs") 

    logs = await map_progress(
        sequence=questions,
        function=lambda q: run_and_log(agent, q, logger),
        desc="Running evaluations"
    )

    for log in logs:
        print(f"Logged interaction to {log}")


def main():
    questions = load_questions()
    asyncio.run(run_agent(questions))


if __name__ == "__main__":
    main()

