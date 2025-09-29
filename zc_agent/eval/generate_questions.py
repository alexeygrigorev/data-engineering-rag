import random
import asyncio
import argparse

import pandas as pd

from pydantic import BaseModel
from pydantic_ai import Agent

from zc_agent.main import initialize_index
from zc_agent.llm import read_prompt
from zc_agent.eval.async_paralell import map_progress


index = initialize_index()


question_generation_prompt = read_prompt('eval_question_generator.md')



class QuestionsList(BaseModel):
    questions: list[str]


class QuestionItem(BaseModel):
    question: str
    filepath: str


question_generator = Agent(
    name="question_generator",
    instructions=question_generation_prompt,
    model='gpt-4o-mini',
    output_type=QuestionsList
)


async def generate_questions_for_doc(doc):
    filename = doc['filename']
    content = doc['content']

    result = await question_generator.run(user_prompt=content)
    output = result.output

    question_items = []
    for question in output.questions:
        item = QuestionItem(
            question=question,
            filepath=filename
        )

        question_items.append(item)

    return question_items


async def generate_questions(sample):
    return await map_progress(
        sequence=sample,
        function=generate_questions_for_doc,
        desc="Generating questions"
    )


def run(num_samples):
    sample = random.sample(index.docs, num_samples)

    results = asyncio.run(generate_questions(sample))
    flat_results = [item.dict() for sublist in results for item in sublist]

    for item in flat_results:
        print(f"{item['filepath']}: {item['question']}")

    df = pd.DataFrame(flat_results, columns=['filepath', 'question'])
    df.to_csv('evals/generated_questions.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Generate questions from documents')
    parser.add_argument('--sample-size', type=int, default=5, 
                       help='Number of documents to sample for question generation (default: 5)')
    
    args = parser.parse_args()

    run(args.sample_size)


if __name__ == "__main__":
    main()
