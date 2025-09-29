
from pathlib import Path
from openai import OpenAI

openai_client = OpenAI()

def llm(instructions, content, model='gpt-4o-mini'):
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": content}
    ]

    response = openai_client.responses.create(
        model='gpt-4o-mini',
        input=messages,
    )

    return response.output_text

def read_prompt(path):
    script_dir = Path(__file__).resolve().parent
    prompt_path = script_dir / 'prompts' / path
    return prompt_path.read_text(encoding='utf-8').strip()
