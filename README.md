# Data Engineering Zoomcamp AI Agent

This repository contains an AI Agents designed to work with data from the [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) repository.

## 📋 Project Overview

This project demonstrates the creation of an AI Agent that can process and analyze content from GitHub repositories, with a focus on educational materials and code from the Data Engineering Zoomcamp course.

### Key Features

- **Multi-format Data Pipeline**: Downloads and processes various file types from GitHub repositories
- **Code-aware Processing**: Handles Python, SQL, Java files alongside markdown content
- **Intelligent Content Extraction**: Uses frontmatter parsing and content analysis
- **RAG-ready Data Preparation**: Prepares data for chunking and vector search

### Supported File Types

- Markdown files (`.md`, `.mdx`)
- Python scripts (`.py`)
- SQL files (`.sql`)
- Java files (`.java`)
- Jupyter notebooks (`.ipynb`)


## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alexeygrigorev/data-engineering-rag.git
   cd data-engineering-rag
   ```

2. **Install dependencies**:
   ```bash
   uv sync --dev
   ```

3. **Set up environment variables**:
   ```bash
   cp .envrc_template .envrc
   ```
   
   Then edit `.envrc` and replace the placeholders with your actual API keys:
   ```bash
   export OPENAI_API_KEY='your-actual-openai-key'
   export ANTHROPIC_API_KEY='your-actual-anthropic-key'
   ```
   
   If you're using [direnv](https://direnv.net/), the environment variables will be loaded automatically when you enter the directory. Otherwise, you can source the file manually:

   ```bash
   source .envrc
   ```

4. **Run the data processing pipeline**:
   Open and run the `data-processing.ipynb` notebook to download and process repository data.

## 📚 Course Information

This project is part of the **AI Agents Course**. Want to follow along?

👉 **Sign up here**: [https://alexeygrigorev.com/aihero/](https://alexeygrigorev.com/aihero/)