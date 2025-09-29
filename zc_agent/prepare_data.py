import io
import zipfile
import requests
import frontmatter

from tqdm.auto import tqdm

from zc_agent import llm
from zc_agent.prompts import read_prompt


doc_extensions = {'md', 'mdx'}
code_extensions = {'py', 'sql', 'java', 'ipynb'}

extensions = doc_extensions | code_extensions

def read_repo_data(repo_owner, repo_name):
    """
    Download and parse all markdown files from a GitHub repository.
    
    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name
    
    Returns:
        List of dictionaries containing file content and metadata
    """
    prefix = 'https://codeload.github.com' 
    url = f'{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)
    
    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    
    for file_info in zf.infolist():
        filepath = file_info.filename
        filepath_lower = filepath.lower()

        if filepath_lower.endswith('/'):
            continue

        filename = filepath_lower.split('/')[-1]

        if filename.startswith('.'):
            continue

        ext = filename.split('.')[-1]

        if ext not in extensions:
            continue

        filepath_edited = filepath.split('/', maxsplit=1)[1]

        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode('utf-8', errors='ignore')
                if ext in doc_extensions:
                    post = frontmatter.loads(content)
                    data = post.to_dict()
                    data['filename'] = filepath_edited
                elif ext in code_extensions:
                    data = {
                        'code': True,
                        'content': content,
                        'filename': filepath_edited
                    }

                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    zf.close()
    return repository_data


de_zoomcamp_data = read_repo_data('DataTalksClub', 'data-engineering-zoomcamp')

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ClearOutputPreprocessor
import os
from pathlib import Path
import json

exporter = MarkdownExporter()
exporter.register_preprocessor(ClearOutputPreprocessor(), enabled=True)

def format_notebook_as_md(raw_notebook: str) -> str:
    nb_parsed = nbformat.reads(
        raw_notebook,
        as_version=nbformat.NO_CONVERT,
    )
    md_body, _ = exporter.from_notebook_node(nb_parsed)
    return md_body

def strip_code_fence(text: str) -> str:
    text = text.strip()

    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    lines = lines[1:]

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines)


notebook_editing_instructions = read_prompt('notebook_edit.md')
code_doc_instructions = read_prompt('code_doc.md')

# process notebooks

ipynb_data = []

for record in de_zoomcamp_data:
    if record.get('code') == True and record['filename'].endswith('.ipynb'):
        ipynb_data.append(record)


print(f'processing {len(ipynb_data)} jupyter notebooks...')

for record in tqdm(ipynb_data):
    md_body = format_notebook_as_md(record['content'])
    new_content = llm(notebook_editing_instructions, md_body)
    new_content = strip_code_fence(new_content)
    record['content'] = new_content
    record['code'] = False

# process code

code_data = []

for record in de_zoomcamp_data:
    if record.get('code') != True:
        continue

    path = record['filename']
    ext = path.split('.')[-1]

    if ext not in code_extensions:
        continue

    if ext == 'ipynb':
        continue

    # print(path)
    code_data.append(record)

print(f'processing {len(code_data)} code files...')

for record in tqdm(code_data):
    code = record['content']

    new_content = llm(code_doc_instructions, code)
    new_content = strip_code_fence(new_content)

    record['content'] = new_content
    record['code'] = False

output_dir = Path('data')
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / 'de-zoomcamp-processed.json'

with output_file.open('w', encoding='utf-8') as f_out:
    json.dump(de_zoomcamp_data, f_out, indent=2)