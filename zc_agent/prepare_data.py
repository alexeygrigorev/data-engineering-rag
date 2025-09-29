import io
import json
import zipfile
from pathlib import Path

import nbformat
import requests
import frontmatter

from tqdm.auto import tqdm
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ClearOutputPreprocessor

from zc_agent.llm import llm, read_prompt


# Constants
DOC_EXTENSIONS = {"md", "mdx"}
CODE_EXTENSIONS = {"py", "sql", "java", "ipynb"}
ALL_EXTENSIONS = DOC_EXTENSIONS | CODE_EXTENSIONS

PROCESSED_DATA_PATH = "data/de-zoomcamp-processed.json"


class RepoDataReader:
    """
    Downloads and parses markdown and code files from a GitHub repository.
    """

    def __init__(self, repo_owner: str, repo_name: str):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.prefix = "https://codeload.github.com"
        self.url = (
            f"{self.prefix}/{self.repo_owner}/{self.repo_name}/zip/refs/heads/main"
        )

    def read(self) -> list[dict]:
        resp = requests.get(self.url)
        if resp.status_code != 200:
            raise Exception(f"Failed to download repository: {resp.status_code}")

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        repository_data = self._extract_files(zf)
        zf.close()

        return repository_data

    def _extract_files(self, zf: zipfile.ZipFile) -> list[dict]:
        data = []

        for file_info in zf.infolist():
            if self._should_skip_file(file_info):
                continue

            filepath = self._normalize_filepath(file_info.filename)

            try:
                with zf.open(file_info) as f_in:
                    content = f_in.read().decode("utf-8", errors="ignore")
                    record = self._process_file_content(filepath, content)
                    if record:
                        data.append(record)

            except Exception as e:
                print(f"Error processing {file_info.filename}: {e}")
                continue

        return data

    def _should_skip_file(self, file_info: zipfile.ZipInfo) -> bool:
        filepath = file_info.filename.lower()

        # directory
        if filepath.endswith("/"):
            return True

        # hidden file
        filename = filepath.split("/")[-1]
        if filename.startswith("."):
            return True

        # unsupported extension
        ext = self._get_extension(file_info.filename)
        if ext not in ALL_EXTENSIONS:
            return True

        return False

    def _get_extension(self, filepath: str) -> str:
        filename = filepath.lower().split("/")[-1]
        if "." in filename:
            return filename.split(".")[-1]
        else:
            return ""

    def _normalize_filepath(self, filepath: str) -> str:
        """
        Removes the top-level directory from the file path inside the zip archive.
        'repo-main/path/to/file.py' -> 'path/to/file.py'
        """
        parts = filepath.split("/", maxsplit=1)
        if len(parts) > 1:
            return parts[1]
        else:
            return parts[0]

    def _process_file_content(self, filepath: str, content: str) -> dict | None:
        ext = self._get_extension(filepath)

        if ext in DOC_EXTENSIONS:
            post = frontmatter.loads(content)
            data = post.to_dict()
            data["filename"] = filepath
            return data

        elif ext in CODE_EXTENSIONS:
            return {"code": True, "content": content, "filename": filepath}

        return None


def read_repo_data(repo_owner: str, repo_name: str) -> list[dict]:
    """
    Convenience function to use RepoDataReader.
    """
    reader = RepoDataReader(repo_owner, repo_name)
    return reader.read()


class NotebookMarkdownFormatter:
    """Converts Jupyter notebook content to markdown format."""

    def __init__(self):
        self.exporter = MarkdownExporter()
        self.exporter.register_preprocessor(ClearOutputPreprocessor(), enabled=True)

    def format(self, raw_notebook: str) -> str:
        nb_parsed = nbformat.reads(
            raw_notebook,
            as_version=nbformat.NO_CONVERT,
        )
        md_body, _ = self.exporter.from_notebook_node(nb_parsed)
        return md_body


def strip_code_fence(text: str) -> str:
    """Remove markdown code fence markers from text."""
    text = text.strip()

    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    lines = lines[1:]

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines)


def filter_notebook_data(data: list[dict]) -> list[dict]:
    """Filter for Jupyter notebook files."""
    return [
        record
        for record in data
        if record.get("code") and record["filename"].endswith(".ipynb")
    ]


def filter_code_data(data: list[dict]) -> list[dict]:
    """Filter for code files (excluding notebooks)."""
    code_data = []

    for record in data:
        if not record.get("code"):
            continue

        path = record["filename"]
        ext = path.split(".")[-1]

        if ext not in CODE_EXTENSIONS:
            continue

        if ext == "ipynb":
            continue

        code_data.append(record)

    return code_data


def process_notebooks(notebook_data: list[dict]) -> None:
    """Process Jupyter notebooks using LLM."""
    print(f"Processing {len(notebook_data)} Jupyter notebooks...")

    md_formatter = NotebookMarkdownFormatter()
    instructions = read_prompt("notebook_edit.md")

    for record in tqdm(notebook_data):
        md_body = md_formatter.format(record["content"])

        new_content = llm(instructions, md_body)
        new_content = strip_code_fence(new_content)

        record["content"] = new_content
        record["code"] = False


def process_code_files(code_data: list[dict]) -> None:
    """Process code files using LLM."""
    print(f"Processing {len(code_data)} code files...")

    instructions = read_prompt("code_doc.md")

    for record in tqdm(code_data):
        code = record["content"]

        new_content = llm(instructions, code)
        new_content = strip_code_fence(new_content)

        record["content"] = new_content
        record["code"] = False



def save_processed_data(
    data: list[dict], output_path: str = PROCESSED_DATA_PATH
) -> None:
    """Save processed data to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = Path(output_path)
    with output_file.open("w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=2)

    print(f"Data saved to {output_file}")


def run() -> None:
    """
    Main processing function that orchestrates the entire
    data preparation pipeline.
    """

    # Download and read repository data
    print("Downloading repository data...")
    de_zoomcamp_data = read_repo_data(
        "DataTalksClub",
        "data-engineering-zoomcamp"
    )
    print(f"Downloaded {len(de_zoomcamp_data)} files")

    # Filter and process notebooks
    notebook_data = filter_notebook_data(de_zoomcamp_data)
    process_notebooks(notebook_data)

    # Filter and process code files
    code_data = filter_code_data(de_zoomcamp_data)
    process_code_files(code_data)

    save_processed_data(de_zoomcamp_data)


if __name__ == "__main__":
    run()
