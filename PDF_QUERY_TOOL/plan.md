# PDF Extraction & Knowledge Discovery 
Integrated Report 2024


# Project Overview

This project provides a robust, AI-powered workflow for extracting,
cleaning, embedding, and querying financial PDF reports. It combines
advanced PDF extraction, OCR, persistent vector storage, and natural
language querying in a modern Streamlit web app.

# PDF Structure Summary

The PDF consists of 59 pages and follows a standard UK statutory
financial report structure:

- **Pages 1:** Table of contents
- **Pages 2-8:** Core financial statements (income statement, balance
  sheet, cash flow, changes in equity)
- **Pages 9-51:** Notes to the financial statements (detailed
  breakdowns, accounting policies, segment information, etc.)
- **Pages 52-59:** Independent auditor’s report and regulatory/legal
  disclosures

# Main Types of Tables

1.  **Primary Financial Statements (Pages 2-8):**
    - Group income statement
    - Group statement of comprehensive income
    - Group and company balance sheets
    - Group and company statements of changes in equity
    - Cash flow statement
2.  **Notes Tables (Pages 9-51):**
    - Detailed breakdowns of line items (e.g., revenue, operating costs,
      tax, assets, liabilities)
    - Multi-year and multi-entity tables (group vs. company)
    - Sensitivity analyses, actuarial assumptions, segmental information
    - Some tables span multiple pages or have multi-row headers
3.  **Audit/Regulatory Tables (Pages 52-59):**
    - Key audit matters
    - Materiality and scope
    - Directors’ remuneration

# End-to-End Workflow

## 1. Extraction & Cleaning

- Use `pdfplumber` for line-based table detection and text extraction on
  all pages.
- Apply OCR (via `pytesseract`) for scanned/image-based pages.
- Extract tables as markdown for context-rich chunking.
- Clean tables: remove empty rows/columns, strip whitespace, convert
  numeric columns, handle multi-row headers.
- Save each table as a CSV (by page/table), and text blocks as `.txt`
  files.
- Merge multi-page tables, ensuring consistent headers and no
  duplicates.

## 2. Embedding & Storage

- Chunk all extracted text and tables for semantic search.
- Generate embeddings for each chunk using `nomic-embed-text` via the
  Ollama SDK.
- Store embeddings, documents, and metadata in a persistent ChromaDB
  collection for efficient retrieval.

## 3. Querying & AI Reasoning

- Provide a Streamlit UI for uploading PDFs, processing, and querying.
- On query, embed the question, retrieve the most relevant chunks from
  ChromaDB, and pass them as context to Microsoft Phi (via Ollama) for
  answer generation.
- Display the answer, top relevant chunks, and a “Think” expander with
  the model’s full reasoning in a `<think>...</think>` block.
- Warn users if OCR was used (possible extraction inaccuracies).

# Knowledge Discovery in Data (KDD) Steps

1.  **Selection:**
    - User uploads one or more financial PDF reports.
2.  **Preprocessing:**
    - Extract tables and text, apply OCR as needed, clean and normalize
      data, chunk for embedding.
3.  **Transformation:**
    - Convert tables to markdown, chunk text, generate vector embeddings
      for all content.
4.  **Data Mining:**
    - Use ChromaDB for semantic search and retrieval of relevant
      information.
    - Query with natural language; retrieve and rank the most relevant
      document chunks.
5.  **Interpretation/Evaluation:**
    - Use Microsoft Phi to synthesize answers and reasoning from
      retrieved context.
    - Present answers, relevant evidence, and model “thoughts” in the UI
      for user interpretation.

# Modern Streamlit App Features

- Multi-file PDF upload
- Robust extraction (tables, text, OCR fallback)
- Persistent vector storage (ChromaDB)
- Fast, interactive querying (Ollama + Phi)
- Session state for performance
- Expander UI for model reasoning
- User warnings for OCR

# Summary of Sprints

**Sprint 1:** Environment setup, initial extraction, and raw output
(CSV/txt)

**Sprint 2:** Data cleaning, header handling, multi-index headers, and
text block detection

**Sprint 3:** Merging multi-page tables, persistent embedding storage
(ChromaDB), and summary report generation

**Final:** Streamlit app for upload, extraction, embedding, querying,
and interactive exploration

# Next Steps

- Further refine extraction and cleaning logic for edge cases
- Expand KDD/analytics features (e.g., trend detection, anomaly
  detection)
- Add more advanced querying or visualization as needed
