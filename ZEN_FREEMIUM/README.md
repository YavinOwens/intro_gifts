# ZEN_FREEMIUM

A Streamlit-based digital product release dashboard and document assistant for advanced analytics, workforce insights, and AI-powered document Q&A.

## Features
- **Dashboard Tabs:** Visualize product, team, release, simulation, and workforce analytics.
- **Document Assistant (RAG):** Ask questions about indexed files (CSV, code, docs) using retrieval-augmented generation.
- **Product Assistant (Chatbot):** Chat with an AI about your data, tables, and charts. Generates code and answers using your loaded data.
- **Skill Extraction:** Extract and view code elements or skills from department/person files in the sidebar.

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YavinOwens/intro_gifts.git
   cd ZEN_FREEMIUM
   ```
2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the App
1. **Ingest files into ChromaDB:**
   ```bash
   python working_docs/chromadb_ingest.py
   ```
2. **Start the Streamlit dashboard:**
   ```bash
   streamlit run working_docs/streamlit_report.py
   ```

## Usage Tips
- Use the sidebar to extract skills and view the user guide.
- Explore all dashboard tabs for analytics and insights.
- Use the Document Assistant tab to see and query all indexed files.
- Keep your CSVs and workforce files up to date for best results.

---
For more details, see comments in the code or open an issue.

## License

This project is licensed under the Apache License, Version 2.0. See the LICENSE file for details.

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 