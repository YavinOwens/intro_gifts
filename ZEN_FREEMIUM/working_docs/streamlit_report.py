import streamlit as st
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import numpy as np
import re
import spacy
import requests
import chromadb
from sentence_transformers import SentenceTransformer
# --- Ensure ChromaDB is always up-to-date with workforce/Department files ---
# import importlib.util
# chromadb_ingest_path = os.path.join(os.path.dirname(__file__), 'chromadb_ingest.py')
# spec = importlib.util.spec_from_file_location('chromadb_ingest', chromadb_ingest_path)
# chromadb_ingest = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(chromadb_ingest)
# chromadb_ingest.main()  # This will re-index all files on app start

st.set_page_config(page_title="Digital Product Release Dashboard", layout="wide")

st.title("Digital Product Release Dashboard")

# --- File paths ---
BASE_DIR = Path(__file__).parent
csv_dir = BASE_DIR / "csv"
workforce_dir = BASE_DIR / "workforce"

# --- ChromaDB RAG Setup (moved to top so embedder is always defined) ---
CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../chromadb_store'))
COLLECTION_NAME = 'csv_docs'
EMBED_MODEL = 'all-MiniLM-L6-v2'

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL)

# --- Load Data ---
def load_csv(filename):
    try:
        return pd.read_csv(filename, dtype=str)
    except Exception as e:
        st.error(f"Could not load {filename}: {e}")
        return pd.DataFrame()

product_df = load_csv(csv_dir / "Product Tracker - Product.csv")
release_df = load_csv(csv_dir / "Product Tracker - Release.csv")
squad_df = load_csv(csv_dir / "Product Tracker - Squad.csv")
team_df = load_csv(csv_dir / "Product Tracker - Team.csv")
workforce_df = load_csv(workforce_dir / "Product Tracker - workforce.csv")

# --- Data Cleaning Helper ---
def clean_df(df):
    # Strip whitespace from headers and values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    return df

# Clean all loaded DataFrames
product_df = clean_df(product_df)
release_df = clean_df(release_df)
squad_df = clean_df(squad_df)
team_df = clean_df(team_df)
workforce_df = clean_df(workforce_df)

# Standardize department/team names for joins
for df in [product_df, squad_df, team_df]:
    if 'Department' in df.columns:
        df['Department'] = df['Department'].str.replace(' ', '').str.lower()
    if 'Team' in df.columns:
        df['Team'] = df['Team'].str.replace(' ', '').str.lower()

# Convert date columns
for col in ['Feature Released', 'Next Release']:
    if col in product_df.columns:
        product_df[col] = pd.to_datetime(product_df[col], errors='coerce', dayfirst=True)
    if col in release_df.columns:
        release_df[col] = pd.to_datetime(release_df[col], errors='coerce', dayfirst=True)

# Convert numeric columns
for col in ['Features', 'Features DOR', 'Feteaures DOD', 'Backlog']:
    if col in product_df.columns:
        product_df[col] = pd.to_numeric(product_df[col], errors='coerce')
    if col in release_df.columns:
        release_df[col] = pd.to_numeric(release_df[col], errors='coerce')

# --- Shared Chat History for Product Assistant (tab and sidebar) ---
if 'product_assistant_history' not in st.session_state:
    st.session_state['product_assistant_history'] = []

# --- Tabs for analytics ---
tabs = st.tabs(["Summary & Feature Metrics", "Team & Product Analytics", "Feature Assignment", "Simulation", "Workforce Skills Overview", "Product Assistant", "Document Assistant"])

with tabs[0]:
    # Keep summary metrics and advanced feature/timeline charts here
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Products", len(product_df))
    col2.metric("Releases", len(release_df))
    col3.metric("Squads", len(squad_df))
    col4.metric("Teams", len(team_df))
    col5.metric("Workforce", len(workforce_df))
    st.markdown("---")
    # Feature metrics and timeline charts
    feat_cols = ['Features', 'Features DOR', 'Feteaures DOD', 'Backlog']
    if not product_df.empty and 'Team' in product_df.columns:
        st.subheader("Feature Metrics per Team")
        feat_team = product_df.groupby('Team')[feat_cols].sum()
        st.bar_chart(feat_team)
    if not product_df.empty and 'Department' in product_df.columns:
        st.subheader("Feature Metrics per Department")
        feat_dept = product_df.groupby('Department')[feat_cols].sum()
        st.bar_chart(feat_dept)
    if not product_df.empty and 'Feature Released' in product_df.columns and 'Product' in product_df.columns:
        st.subheader("Feature Released Dates per Product")
        st.dataframe(product_df[['Product', 'Team', 'Department', 'Feature Released', 'Next Release']].sort_values('Feature Released'))
        import altair as alt
        if product_df['Feature Released'].notnull().any():
            chart = alt.Chart(product_df.dropna(subset=['Feature Released'])).mark_bar().encode(
                x='Feature Released:T',
                x2='Next Release:T',
                y=alt.Y('Product:N', sort='-x'),
                color='Team:N',
                tooltip=['Product', 'Team', 'Department', 'Feature Released', 'Next Release']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
    if not product_df.empty and 'Team' in product_df.columns:
        st.subheader("Features per Team (DOR, DOD, Released, Backlog)")
        feat_team_long = product_df.melt(id_vars=['Team'], value_vars=feat_cols, var_name='Metric', value_name='Value')
        chart = alt.Chart(feat_team_long).mark_bar().encode(
            x='Team:N',
            y='Value:Q',
            color='Metric:N',
            tooltip=['Team', 'Metric', 'Value']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    # Burndown chart
    st.subheader("Burndown Chart (Backlog over Time)")
    if 'Feature Released' in product_df.columns and 'Backlog' in product_df.columns:
        burndown_df = product_df[['Feature Released', 'Backlog']].dropna(subset=['Feature Released'])
        if burndown_df.empty:
            st.warning("No 'Feature Released' dates found in the data. Please ensure all products have a release date.")
        else:
            burndown_df = burndown_df.sort_values('Feature Released')
            burndown_df['Cumulative Backlog'] = burndown_df['Backlog'][::-1].cumsum()[::-1]
            import altair as alt
            chart = alt.Chart(burndown_df).mark_line(point=True).encode(
                x=alt.X('Feature Released:T', title='Release Date'),
                y=alt.Y('Cumulative Backlog:Q', title='Remaining Backlog'),
                tooltip=['Feature Released', 'Cumulative Backlog']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        # Warn if any products are missing release dates
        missing_dates = product_df[product_df['Feature Released'].isna()]
        if not missing_dates.empty:
            st.warning(f"Some products are missing 'Feature Released' dates. Please update the CSV to include these dates for a complete burndown chart.")
    else:
        st.warning("'Feature Released' and/or 'Backlog' columns not found in Product table. Please check your CSV headers.")

with tabs[1]:
    # # Team & Product Analytics tab
    # if not product_df.empty and 'Team' in product_df.columns:
    #     st.subheader("Products per Team")
    #     st.bar_chart(product_df['Team'].value_counts().sort_values(ascending=False))
    # if not product_df.empty and not product_df.empty and 'Product' in release_df.columns and 'Product' in product_df.columns and 'Team' in product_df.columns:
    #     rel_team_df = release_df.merge(product_df[['Product', 'Team']], how='left', on='Product')
    #     if 'Team' in rel_team_df.columns:
    #         rel_team_counts = rel_team_df['Team'].value_counts().sort_values(ascending=False)
    #         st.subheader("Releases per Team")
    #         st.bar_chart(rel_team_counts)
    # if not squad_df.empty and 'Team' in squad_df.columns:
    #     squad_team_counts = squad_df['Team'].value_counts().sort_values(ascending=False)
    #     st.subheader("Squads per Team")
    #     st.bar_chart(squad_team_counts)
    # if not workforce_df.empty and 'Department' in workforce_df.columns:
    #     dept_counts = workforce_df['Department'].value_counts().sort_values(ascending=False)
    #     st.subheader("Workforce by Department")
    #     st.bar_chart(dept_counts)
    # if not team_df.empty and 'Department' in team_df.columns:
    #     dept_member_counts = team_df['Department'].value_counts().sort_values(ascending=False)
    #     st.subheader("Team Members per Department")
    #     st.bar_chart(dept_member_counts)
    # if not squad_df.empty and 'Department' in squad_df.columns and 'Team' in squad_df.columns:
    #     teams_per_dept = squad_df.groupby('Department')['Team'].nunique().sort_values(ascending=False)
    #     st.subheader("Number of Teams per Department")
    #     st.bar_chart(teams_per_dept)
    # if not product_df.empty and 'Department' in product_df.columns:
    #     st.subheader("Products per Department")
    #     st.bar_chart(product_df['Department'].value_counts().sort_values(ascending=False))
    # if not product_df.empty and 'Status' in product_df.columns:
    #     st.subheader("Product Status Distribution")
    #     st.bar_chart(product_df['Status'].value_counts())

    # --- Person-level analytics (Simulated) ---
    st.markdown("---")
    st.header("Person-Level Analytics (Simulated)")
    assign_sim_path = BASE_DIR / 'csv' / 'Product Tracker - Team_Feature_Assignment_simulated.csv'
    try:
        assign_sim_df = pd.read_csv(assign_sim_path)
        if not assign_sim_df.empty:
            products = assign_sim_df['Product'].unique().tolist()
            selected_products = st.multiselect("Filter by Product", products, default=products)
            filtered_df = assign_sim_df[assign_sim_df['Product'].isin(selected_products)] if selected_products else assign_sim_df

            st.subheader("Number of Features Assigned per Team Member (Simulated)")
            import altair as alt
            if not filtered_df.empty:
                chart = alt.Chart(filtered_df).mark_bar().encode(
                    x=alt.X('Team Member:N', sort='-y'),
                    y=alt.Y('Features Assigned:Q'),
                    color=alt.Color('Product:N', legend=alt.Legend(title="Product")),
                    tooltip=['Team Member', 'Product', 'Features Assigned']
                ).properties(height=400)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No data to display for selected product(s).")

            st.subheader("Assignments Table (Person, Team, Product, Features Assigned)")
            st.dataframe(filtered_df[['Team Member', 'Team', 'Product', 'Features Assigned']])
        else:
            st.info("No simulated person-level assignment data available.")
    except Exception as e:
        st.warning(f"Could not load Team_Feature_Assignment_simulated.csv: {e}")

with tabs[2]:
    st.header("Feature Assignment Overview")
    import_path = BASE_DIR / 'csv' / 'Product Tracker - Team_Feature_Assignment.csv'
    try:
        assign_df = pd.read_csv(import_path)
        st.dataframe(assign_df)
        # Bar chart: Number of features assigned per team member
        st.subheader("Number of Features Assigned per Team Member")
        member_counts = assign_df['Team Member'].value_counts().sort_values(ascending=False)
        st.bar_chart(member_counts)
        # Bar chart: Number of features assigned per team
        st.subheader("Number of Features Assigned per Team")
        team_counts = assign_df['Team'].value_counts().sort_values(ascending=False)
        st.bar_chart(team_counts)
    except Exception as e:
        st.warning(f'Could not load Team_Feature_Assignment.csv: {e}')

with tabs[3]:
    st.header("Feature Metrics Simulation Over Time")
    sim_path = BASE_DIR / 'csv' / 'Product Tracker - Product_simulation.csv'
    try:
        sim_df = pd.read_csv(sim_path)
        if not sim_df.empty:
            products = sim_df['Product'].unique().tolist()
            selected_products = st.multiselect("Select Products", products, default=products[:1])
            if selected_products:
                import altair as alt
                filtered = sim_df[sim_df['Product'].isin(selected_products)]
                # Combined chart: all metrics, all products (move this above individual charts)
                st.subheader("All Metrics Over Time (All Products)")
                melted = filtered.melt(id_vars=['Product', 'Date'], value_vars=['Features', 'Features DOR', 'Feteaures DOD', 'Backlog'], var_name='Metric', value_name='Value')
                chart = alt.Chart(melted).mark_line().encode(
                    x='Date:T',
                    y=alt.Y('Value:Q', title='Value'),
                    color=alt.Color('Product:N', legend=alt.Legend(title="Product")),
                    strokeDash=alt.StrokeDash('Metric:N', legend=alt.Legend(title="Metric")),
                    tooltip=['Product', 'Metric', 'Date', 'Value']
                ).properties(height=400)
                st.altair_chart(chart, use_container_width=True)
                # Separate charts for each metric
                for metric in ['Features', 'Features DOR', 'Feteaures DOD', 'Backlog']:
                    st.subheader(f"{metric} Over Time")
                    chart = alt.Chart(filtered).mark_line().encode(
                        x='Date:T',
                        y=alt.Y(f'{metric}:Q', title=metric),
                        color=alt.Color('Product:N', legend=alt.Legend(title="Product")),
                        tooltip=['Product', 'Date', metric]
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)
                # Show data for selected products
                st.dataframe(filtered)
            else:
                st.info("Please select at least one product.")
        else:
            st.info("Simulation data is empty.")
    except Exception as e:
        st.warning(f"Could not load simulation data: {e}")

with tabs[4]:
    st.header("Workforce Skills Overview")
    import glob
    import re
    # Scan all department files folders
    base_dir = BASE_DIR / 'workforce' / 'Department'
    departments = [d for d in base_dir.iterdir() if d.is_dir()]
    skill_rows = []
    for dept in departments:
        dept_name = dept.name
        files_dir = dept / 'files'
        people_csv = dept / 'people' / 'people.csv'
        if not files_dir.exists():
            continue
        for file_path in files_dir.glob('*.*'):
            match = re.match(r'(person\d+)\.(\w+)$', file_path.name)
            if match:
                person = match.group(1)
                ext = match.group(2)
                skill_rows.append({
                    'department': dept_name,
                    'person': person,
                    'file_type': ext,
                    'file_name': file_path.name
                })
    skills_df = pd.DataFrame(skill_rows)
    # Join with people.csv for HR info
    hr_rows = []
    for dept in departments:
        people_csv = dept / 'people' / 'people.csv'
        if people_csv.exists():
            hr_df = pd.read_csv(people_csv)
            hr_df['department_folder'] = dept.name
            hr_rows.append(hr_df)
    if hr_rows:
        hr_df = pd.concat(hr_rows, ignore_index=True)
    else:
        hr_df = pd.DataFrame()
    # Merge skills with HR info
    merged = skills_df.merge(hr_df, left_on=['department', 'person'], right_on=['department_folder', 'name'], how='left')
    # Left join with product assignments
    assign_sim_path = BASE_DIR / 'csv' / 'Product Tracker - Team_Feature_Assignment_simulated.csv'
    try:
        assign_sim_df = pd.read_csv(assign_sim_path)
        assign_sim_df['person_lower'] = assign_sim_df['Team Member'].str.lower()
        merged['person_lower'] = merged['person'].str.lower()
        merged = merged.merge(assign_sim_df, left_on=['department', 'person_lower'], right_on=['Department', 'person_lower'], how='left')
    except Exception as e:
        st.warning(f"Could not load Team_Feature_Assignment_simulated.csv: {e}")
    # Display table
    st.subheader("People, Skills (File Types), and Product Assignments")
    # Only show columns that exist in merged
    desired_cols = ['department', 'person', 'file_type', 'file_name', 'id', 'email', 'role', 'Product', 'Features Assigned']
    available_cols = [col for col in desired_cols if col in merged.columns]
    st.dataframe(merged[available_cols])
    # Summary chart: count of file types per department
    st.subheader("File Types per Department")
    if not skills_df.empty:
        chart = skills_df.groupby(['department', 'file_type']).size().reset_index(name='count')
        import altair as alt
        c = alt.Chart(chart).mark_bar().encode(
            x=alt.X('department:N', title='Department'),
            y=alt.Y('count:Q', title='Count'),
            color='file_type:N',
            tooltip=['department', 'file_type', 'count']
        ).properties(height=400)
        st.altair_chart(c, use_container_width=True)

with tabs[5]:
    st.header('Product Assistant (Chatbot)')
    # Add expander for dataframe schemas
    with st.expander('Available DataFrames and Columns (for prompt writing)'):
        df_list = [
            ('product_df', product_df),
            ('release_df', release_df),
            ('team_df', team_df),
            ('squad_df', squad_df),
            ('workforce_df', workforce_df),
            ('assign_sim_df', assign_sim_df if 'assign_sim_df' in locals() else None),
            ('assign_df', assign_df if 'assign_df' in locals() else None),
            ('sim_df', sim_df if 'sim_df' in locals() else None),
            ('skills_df', skills_df if 'skills_df' in locals() else None),
            ('hr_df', hr_df if 'hr_df' in locals() else None),
            ('merged', merged if 'merged' in locals() else None),
            ('skill_df', skill_df if 'skill_df' in locals() else None),
        ]
        for name, df in df_list:
            if df is not None and hasattr(df, 'columns'):
                st.markdown(f"**{name}**: {', '.join([str(col) for col in df.columns])}")
    # Display chat history using chat bubbles
    for idx, message in enumerate(st.session_state['product_assistant_history']):
        role, text = message
        st.chat_message(role).write(text)
        # If assistant and contains python code, show code and run button
        if role == 'assistant':
            code_blocks = re.findall(r'```python\n([\s\S]+?)```', text)
            for i, code in enumerate(code_blocks):
                st.code(code, language='python')
                run_key = f'run_code_{idx}_{i}'
                if st.button('Run this code', key=run_key):
                    # Safe local namespace for code execution
                    local_vars = {
                        'pd': pd,
                        'np': np,
                        'product_df': product_df,
                        'release_df': release_df,
                        'team_df': team_df,
                        'squad_df': squad_df,
                        'workforce_df': workforce_df,
                        'assign_sim_df': assign_sim_df if 'assign_sim_df' in locals() else None,
                        'assign_df': assign_df if 'assign_df' in locals() else None,
                        'sim_df': sim_df if 'sim_df' in locals() else None,
                        'skills_df': skills_df if 'skills_df' in locals() else None,
                        'hr_df': hr_df if 'hr_df' in locals() else None,
                        'merged': merged if 'merged' in locals() else None,
                        'skill_df': skill_df if 'skill_df' in locals() else None,
                    }
                    # Preprocess code: remove markdown headings and non-Python lines, and fix common typos
                    def clean_code(code):
                        lines = code.split('\n')
                        cleaned = []
                        for line in lines:
                            # Remove markdown headings (but keep python comments)
                            if line.strip().startswith('#') and not line.strip().startswith('# '):
                                cleaned.append(line)
                            elif not line.strip().startswith('#') and not line.strip().startswith('```') and not line.strip().startswith('---') and not line.strip().startswith('Assuming') and not line.strip().startswith('Group by'):
                                cleaned.append(line)
                        code_str = '\n'.join(cleaned)
                        # Auto-fix common hallucinated typo: plt0r -> plt
                        code_str = code_str.replace('plt0r', 'plt')
                        # Auto-replace 'df' with the correct dataframe name if only one is referenced in the user prompt
                        import re as _re
                        # Try to infer which dataframe is being referenced
                        user_prompt = st.session_state['product_assistant_history'][idx-1][1] if idx > 0 and st.session_state['product_assistant_history'][idx-1][0] == 'user' else ''
                        df_names = ['product_df', 'release_df', 'team_df', 'squad_df', 'workforce_df', 'assign_sim_df', 'assign_df', 'sim_df', 'skills_df', 'hr_df', 'merged', 'skill_df']
                        referenced = [name for name in df_names if name in user_prompt or name in code_str]
                        if 'df' in code_str and len(referenced) == 1:
                            code_str = _re.sub(r'\bdf\b', referenced[0], code_str)
                        return code_str
                    code_to_run = clean_code(code)
                    try:
                        exec(code_to_run, {}, local_vars)
                        # If code creates a matplotlib or Altair chart, try to display it
                        import matplotlib.pyplot as plt
                        fig = plt.gcf()
                        if fig.get_axes():
                            st.pyplot(fig)
                            plt.clf()
                        if 'alt' in local_vars:
                            for v in local_vars.values():
                                if hasattr(v, 'to_altair'):  # Altair chart
                                    st.altair_chart(v)
                    except ValueError as ve:
                        if 'pie requires either y column or' in str(ve):
                            st.error("Pie chart error: You must call .plot.pie() on a Series (e.g., df['col'].value_counts().plot.pie()), not a DataFrame. Or use plotly for more flexibility. Example: skills_df['skill'].value_counts().plot.pie() or use plotly.express.pie.")
                        else:
                            st.error(f"Error running code: {ve}")
                    except SyntaxError as se:
                        st.error("Syntax error: There may be extra text, markdown, or incomplete code in the code block. Please ensure only valid Python code is present.\n" + str(se))
                    except Exception as e:
                        st.error(f"Error running code: {e}")
    # Chat input
    user_input = st.chat_input('Ask about your data, tables, charts, or graphs:')
    if user_input:
        # RAG: Retrieve top 3 relevant CSV rows from ChromaDB
        query_emb = embedder.encode([user_input])[0]
        results = collection.query(query_embeddings=[query_emb], n_results=3)
        rag_context = ''
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            rag_context += f"[From {meta['filename']} row {meta['row']}]: {doc}\n"
        # Compose context from all dataframes
        context = ''
        try:
            context += '\n--- Product Data ---\n' + product_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- Release Data ---\n' + release_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- Team Data ---\n' + team_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- Squad Data ---\n' + squad_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- Workforce Data ---\n' + workforce_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- Simulated Assignment ---\n' + assign_sim_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- Feature Assignment ---\n' + assign_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- Simulation Data ---\n' + sim_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- Skills Extracted ---\n' + skill_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- Skills Table ---\n' + skills_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- HR Table ---\n' + hr_df.head(10).to_csv(index=False)
        except Exception: pass
        try:
            context += '\n--- Merged Table ---\n' + merged.head(10).to_csv(index=False)
        except Exception: pass
        # Compose schema info for each dataframe
        schema_info = ""
        for name, df in [
            ("product_df", product_df),
            ("release_df", release_df),
            ("team_df", team_df),
            ("squad_df", squad_df),
            ("workforce_df", workforce_df),
            ("assign_sim_df", assign_sim_df if 'assign_sim_df' in locals() else None),
            ("assign_df", assign_df if 'assign_df' in locals() else None),
            ("sim_df", sim_df if 'sim_df' in locals() else None),
            ("skills_df", skills_df if 'skills_df' in locals() else None),
            ("hr_df", hr_df if 'hr_df' in locals() else None),
            ("merged", merged if 'merged' in locals() else None),
            ("skill_df", skill_df if 'skill_df' in locals() else None),
        ]:
            if df is not None and not df.empty:
                schema_info += f"\n{name} columns: {', '.join([f'{col} ({df[col].dtype})' for col in df.columns])}"
        visualization_instruction = (
            "When the user requests a visualization, generate only the code to create the chart "
            "Assume the relevant dataframes {schema_info} (such as skill_df, product_df, etc.) are already loaded and available. "
            "Do not include Streamlit UI code (such as st.write, st.button, st.plotly_chart, etc.). "
            "Return only the code for the visualization logic and plotting. "
            "Always return code in a code block with triple backticks and the word python (e.g., ```python ... ```), so it can be executed directly. "
            "Never use a variable named df—always use the actual dataframe variable names provided (e.g., skills_df, product_df, etc.). "
            "Only use matplotlib for all visualizations. Do not use plotly, seaborn, or any other library. "
            "Always check that arrays or lists used for plotting or DataFrame creation are the same length to avoid errors. "
            "You can also help the user find files in the workforce directory. If the user asks for a file, search the RAG context and return the file name, type, department, and relative path if found."
        )
        prompt = (
            "You are a helpful data assistant. The user can ask about the data, tables, charts, or graphs in the Streamlit dashboard. "
            f"{visualization_instruction} "
            "When generating Python code, use the dataframe variables as defined below. Do not re-read CSVs. "
            f"{schema_info}\n\n"
            f"RAG Context (retrieved from CSVs):\n{rag_context}\n"
            "Context:\n"
            f"{context}\n\n"
            f"User: {user_input}\nBot:"
        )
        # Call Ollama phi4-mini:latest
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'phi4-mini:latest',
                    'prompt': prompt,
                    'stream': False
                },
                timeout=180
            )
            if response.status_code == 200:
                bot_reply = response.json().get('response', '').strip()
            else:
                bot_reply = f"[Error from Ollama: {response.status_code}]"
        except requests.exceptions.Timeout:
            bot_reply = "[Error: Ollama timed out. The model may be slow to load or busy. Try again or increase the timeout.]"
        except Exception as e:
            bot_reply = f"[Error connecting to Ollama: {e}]"
        st.session_state['product_assistant_history'].append(('user', user_input))
        st.session_state['product_assistant_history'].append(('assistant', bot_reply))
        st.rerun()

with tabs[6]:
    st.header('Document Assistant (RAG)')
    # List all available files in ChromaDB (only indexed files)
    all_metas = collection.get(include=['metadatas'])['metadatas']
    file_list = []
    table_rows = []
    for meta in all_metas:
        if meta and 'filename' in meta:
            file_list.append(f"{meta.get('filename')} (type: {meta.get('file_type', 'unknown')}, dept: {meta.get('department', '-')}, path: {meta.get('relative_path', '-')})")
            table_rows.append({
                'Filename': meta.get('filename'),
                'File Type': meta.get('file_type', 'unknown'),
                'Department': meta.get('department', '-'),
                'Relative Path': meta.get('relative_path', '-')
            })
    with st.expander('Available Files in Database (for prompt writing)', expanded=True):
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows))
        else:
            st.info('No files found in ChromaDB.')
    # Chat UI for document Q&A
    if 'doc_assistant_history' not in st.session_state:
        st.session_state['doc_assistant_history'] = []
    for message in st.session_state['doc_assistant_history']:
        role, text = message
        st.chat_message(role).write(text)
    doc_user_input = st.chat_input('Ask about your documentation or files:')
    if doc_user_input:
        # RAG: Retrieve top 3 relevant files/chunks from ChromaDB
        doc_query_emb = embedder.encode([doc_user_input])[0]
        doc_results = collection.query(query_embeddings=[doc_query_emb], n_results=3)
        doc_rag_context = ''
        for doc, meta in zip(doc_results['documents'][0], doc_results['metadatas'][0]):
            doc_rag_context += f"[From {meta.get('filename', '?')} (type: {meta.get('file_type', '?')}, dept: {meta.get('department', '-')}, path: {meta.get('relative_path', '-')})]: {doc}\n"
        doc_prompt = (
            "You are a helpful document assistant. The user can ask about any documentation or file in the database. "
            "Use the RAG context below to answer the user's question. If the answer is not in the context, say you don't know.\n"
            f"RAG Context (retrieved from files):\n{doc_rag_context}\n"
            f"User: {doc_user_input}\nBot:"
        )
        # Call Ollama phi4-mini:latest
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'phi4-mini:latest',
                    'prompt': doc_prompt,
                    'stream': False
                },
                timeout=180
            )
            if response.status_code == 200:
                doc_bot_reply = response.json().get('response', '').strip()
            else:
                doc_bot_reply = f"[Error from Ollama: {response.status_code}]"
        except requests.exceptions.Timeout:
            doc_bot_reply = "[Error: Ollama timed out. The model may be slow to load or busy. Try again or increase the timeout.]"
        except Exception as e:
            doc_bot_reply = f"[Error connecting to Ollama: {e}]"
        st.session_state['doc_assistant_history'].append(('user', doc_user_input))
        st.session_state['doc_assistant_history'].append(('assistant', doc_bot_reply))
        st.rerun()

# --- User Guide/Info Section (at top of sidebar) ---
st.sidebar.header('How to Use This App')
st.sidebar.markdown('''
**Main Features:**
- **Dashboard Tabs:** Navigate the main tabs to view product, team, release, simulation, and workforce analytics.
- **Document Assistant (RAG):** In the last tab, see all indexed files and ask questions about your documentation and files. The assistant uses retrieval-augmented generation for accurate answers.
- **Product Assistant (Chatbot):** Use the Product Assistant tab to chat with an AI about your data, tables, and charts. It can generate code and answer questions using your loaded data.
- **Skill Extraction:** Use the sidebar to select a department and person to extract and view code elements or skills from their files.

**Tips:**
- Use the expanders and tables to explore available data and files.
- For best results, keep your CSVs and workforce files up to date.
- Indexed files are shown in the Document Assistant tab for easy reference.
''')
st.sidebar.markdown('---')

# --- Sidebar: Department/Person Skill Extraction ---
st.sidebar.header("Department File Skill Extractor")
base_dir = BASE_DIR / 'workforce' / 'Department'
departments = [d for d in base_dir.iterdir() if d.is_dir()]
dep_names = [d.name for d in departments]
selected_dept = st.sidebar.selectbox("Select Department", dep_names)

# Get people in selected department
people_csv = base_dir / selected_dept / 'people' / 'people.csv'
if people_csv.exists():
    people_df = pd.read_csv(people_csv)
    people = people_df['name'].tolist()
else:
    people = []
selected_person = st.sidebar.selectbox("Select Person (optional)", ['All'] + people)

# Scan files for selected department
files_dir = base_dir / selected_dept / 'files'
skill_rows = []
if files_dir.exists():
    for file_path in files_dir.glob('*.*'):
        match = re.match(r'(person\d+)\.(\w+)$', file_path.name)
        if match:
            person = match.group(1)
            ext = match.group(2)
            if selected_person != 'All' and person != selected_person:
                continue
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Extraction logic
            if ext == 'py':
                imports = re.findall(r'^import\s+\w+|^from\s+\w+\s+import', content, re.MULTILINE)
                classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                defs = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
                for imp in imports:
                    skill_rows.append({'department': selected_dept, 'person': person, 'file_type': ext, 'file_name': file_path.name, 'skill_type': 'import', 'skill': imp})
                for cls in classes:
                    skill_rows.append({'department': selected_dept, 'person': person, 'file_type': ext, 'file_name': file_path.name, 'skill_type': 'class', 'skill': cls})
                for d in defs:
                    skill_rows.append({'department': selected_dept, 'person': person, 'file_type': ext, 'file_name': file_path.name, 'skill_type': 'def', 'skill': d})
            elif ext == 'css':
                selectors = re.findall(r'([\w\.#-]+)\s*\{', content)
                for sel in selectors:
                    skill_rows.append({'department': selected_dept, 'person': person, 'file_type': ext, 'file_name': file_path.name, 'skill_type': 'selector', 'skill': sel})
            elif ext == 'html':
                tags = re.findall(r'<(\w+)', content)
                for tag in set(tags):
                    skill_rows.append({'department': selected_dept, 'person': person, 'file_type': ext, 'file_name': file_path.name, 'skill_type': 'tag', 'skill': tag})
            elif ext == 'yaml':
                keys = re.findall(r'^(\w+):', content, re.MULTILINE)
                for key in keys:
                    skill_rows.append({'department': selected_dept, 'person': person, 'file_type': ext, 'file_name': file_path.name, 'skill_type': 'key', 'skill': key})
            else:
                # Use spaCy for generic NLP extraction (keywords)
                doc = nlp(content)
                for ent in doc.ents:
                    skill_rows.append({'department': selected_dept, 'person': person, 'file_type': ext, 'file_name': file_path.name, 'skill_type': 'entity', 'skill': ent.text})

skill_df = pd.DataFrame(skill_rows)
if not skill_df.empty:
    st.sidebar.subheader("Extracted Skills/Code Elements")
    st.sidebar.dataframe(skill_df)
else:
    st.sidebar.info("No skills/code elements extracted for this selection.") 