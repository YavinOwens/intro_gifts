# Skills Gap Tool

## Introduction

The **Skills Gap Tool** is an introductory application designed to help users analyze, categorize, and bridge skills gaps in codebases and technical documents. While it offers a subset of the advanced features found in the Zen platform, it provides a robust workflow for ingesting files, extracting skills, running retrieval-augmented Q&A, and surfacing high-quality learning resources from top providers.

## Benefits

- **Automated Skill Extraction:** Quickly identify key programming, data, and technology skills present in your code or documents.
- **Gap Analysis:** Visualize which skills are present and which may be missing, helping guide upskilling or hiring decisions.
- **Retrieval-Augmented Q&A:** Ask questions about your own files and get answers grounded in your actual content, not just generic LLM knowledge.
- **Curated Learning Resources:** Instantly surface high-quality references and learning materials from gold standard providers (e.g., Harvard, MIT, AWS, Google, Accenture, etc.).
- **Easy Ingestion:** Upload a wide range of file types (code, text, PDFs, databases) and batch process them for analysis.
- **Report Generation:** Export your analysis and findings in Markdown or Quarto formats for sharing or further review.

## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) running locally with supported models (e.g., `phi3:mini`, `phi4-mini-reasoning:latest`, `nomic-embed-text:latest`)
- [ChromaDB](https://www.trychroma.com/) (handled automatically)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Application
```bash
streamlit run SKILLS_GAP_TOOL/app.py
```

---

## Application Walkthrough

The Skills Gap Tool is organized into several pages, each accessible from the sidebar:

### 1. **Analytics**
- **Purpose:** View live statistics about your ingested files and their content.
- **Features:**
  - Total files ingested
  - File type breakdown
  - Chunk counts per file

### 2. **Ingested Files**
- **Purpose:** Preview the content of files you have uploaded or ingested.
- **Features:**
  - Select and view any ingested file
  - Preview code, text, or table data

### 3. **Skill Categorization**
- **Purpose:** See which skills have been extracted from your files.
- **Features:**
  - Table of files, skills, and libraries
  - View skills by extraction model (Regex+NLP, Ollama LLMs)
  - Tabular and detailed views
  - Extract and cache skills for new files

### 4. **RAG Q&A**
- **Purpose:** Ask questions about your own files using Retrieval-Augmented Generation (RAG).
- **Features:**
  - Select a file for context (optional)
  - Enter a question or prompt
  - The system retrieves the most relevant content from your files and sends it, along with your question, to a local LLM (Ollama)
  - Get answers grounded in your actual data

### 5. **Web References**
- **Purpose:** Discover high-quality learning resources and references for upskilling.
- **Features:**
  - **All References:** View all collected references
  - **Filtered (High-Quality) References:**
    - Filter by extracted skill(s)
    - Filter by provider type (Education, Technology, Consultancy)
    - Select up to 5 gold standard providers
    - Run a web search for selected skills/providers and view curated results
  - **Skill-Driven Web References:**
    - Run targeted web searches based on extracted skills
    - View and filter results

### 6. **Generate Report**
- **Purpose:** Export your analysis and findings.
- **Features:**
  - Select which documents and references to include
  - Generate a Markdown or Quarto report
  - Download or view the report in your browser

---

## Example Workflow
1. **Upload or ingest files** (code, text, PDFs, etc.) using the sidebar.
2. **Analyze your files** in the Analytics and Ingested Files pages.
3. **Extract and review skills** in the Skill Categorization page.
4. **Ask questions** about your files in the RAG Q&A page.
5. **Find learning resources** in the Web References page, filtering by skill and provider.
6. **Generate a report** to share your findings.

---

## Limitations Compared to Zen Platform
- This tool is a demonstration and does not include advanced user management, workflow automation, or deep integration with enterprise systems.
- The skill extraction and RAG Q&A are designed for small to medium codebases and may not scale to very large datasets.
- Only a subset of LLMs and embedding models are supported (local Ollama only).

---

## Contributing
Pull requests and feedback are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License
See LICENSE file (if available) or contact the author for usage terms. 