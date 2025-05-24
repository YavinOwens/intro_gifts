import streamlit as st
import sqlite3
import pandas as pd
import polars as pl
import os
import ollama
import traceback

st.set_page_config(page_title="Database Query Tool (AI-Powered)", layout="wide")
st.title("Database Query Tool (AI-Powered)")

DB_PATH = os.path.join("DATABASE_QUERY_TOOL", "road_network_demo.db")
connect_btn = st.sidebar.button("Connect to Database")

if 'db_connected' not in st.session_state:
    st.session_state['db_connected'] = False
if 'table_list' not in st.session_state:
    st.session_state['table_list'] = []
if 'selected_tables' not in st.session_state:
    st.session_state['selected_tables'] = []
if 'custom_tables' not in st.session_state:
    st.session_state['custom_tables'] = {}  # name: DataFrame
if 'query_language' not in st.session_state:
    st.session_state['query_language'] = ['SQL']

# Set model names
BASE_MODEL = "phi3:mini"
CODE_MODEL = "codellama:7b-instruct"

# Remove persistent DB connection from session state
# Only store DB_PATH if needed

if connect_btn:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
        st.session_state['db_connected'] = True
        st.session_state['table_list'] = tables
        st.success(f"Connected to {DB_PATH}. Found {len(tables)} tables.")
    except Exception as e:
        st.session_state['db_connected'] = False
        st.error(f"Failed to connect: {e}")

if st.session_state['db_connected']:
    tables = st.session_state['table_list']
    custom_tables = list(st.session_state['custom_tables'].keys())
    all_tables = tables + custom_tables
    selected_tables = st.sidebar.multiselect(
        "Select up to 4 tables to query/compare:", all_tables, max_selections=4
    )
    st.session_state['selected_tables'] = selected_tables
    table_schemas = {}
    table_samples = {}
    table_dfs = {}
    if selected_tables:
        st.write("### Table Previews:")
        for t in selected_tables:
            st.write(f"**{t}**")
            try:
                if t in tables:
                    with sqlite3.connect(DB_PATH) as conn:
                        df = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 10", conn)
                        # Get schema
                        cursor = conn.execute(f"PRAGMA table_info({t})")
                        schema = [(row[1], row[2]) for row in cursor.fetchall()]
                else:
                    df = st.session_state['custom_tables'][t]
                    schema = list(zip(df.columns, [str(dtype) for dtype in df.dtypes]))
                st.dataframe(df)
                table_schemas[t] = schema
                table_samples[t] = df.head(3).to_dict(orient='records')
                table_dfs[t] = df
            except Exception as e:
                st.error(f"Error loading {t}: {e}")

    language_options = ['SQL', 'Python', 'Polars']
    st.sidebar.write('## Query Language')
    st.session_state['query_language'] = st.sidebar.multiselect(
        'Select preferred query language(s):',
        language_options,
        default=['SQL']
    )

    st.write("### Ask a question about the selected table(s):")
    user_query = st.text_input("Enter your question:")
    save_table_name = st.session_state.get('save_table_name', '')
    query_langs = st.session_state['query_language']
    if user_query and selected_tables:
        prompt = f"You are a database assistant. The user has selected the following tables from a SQLite database.\n"
        for t in selected_tables:
            prompt += f"\nTable: {t}\nSchema: {table_schemas.get(t, [])}\nSample Rows: {table_samples.get(t, [])}\n"
        prompt += f"\nUser question: {user_query}\n"
        if query_langs:
            if len(query_langs) == 1:
                prompt += f"\nThe user prefers the answer/code in {query_langs[0]}.\n"
            else:
                prompt += f"\nThe user is open to answers/code in any of: {', '.join(query_langs)}.\n"
        prompt += ("\nIf code is needed, specify if it should be SQL, Python (with pandas or polars), and what it should do. "
                   "If you need to run multiple queries, generate Python (with pandas or polars) code that executes each query separately and combines the results, rather than multiple SQL statements in one block. "
                   "You may generate SQL to read, query, or create tables, but you must NOT delete tables or data. "
                   "If you generate code, show it clearly. Then answer the question in natural language.\n")
        st.info("Querying base agent (Phi3-mini)...")
        try:
            phi_response = ollama.chat(
                model=BASE_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            phi_content = phi_response['message']['content']
            st.write("**Base Agent (Phi3-mini) Reasoning:**")
            st.write(phi_content)
            import re
            code_block = re.search(r'```(sql|python|polars)?\s*([\s\S]+?)```', phi_content)
            code_type = None
            code_str = None
            if code_block:
                code_type = code_block.group(1) or "sql"
                code_str = code_block.group(2).strip()
                st.write(f"**Generated {code_type} code:**")
                st.code(code_str, language=code_type)
                # Use codellama for code agent if code is present
                code_agent_response = ollama.chat(
                    model=CODE_MODEL,
                    messages=[{"role": "user", "content": code_str}]
                )
                code_str = code_agent_response['message']['content']
                st.write(f"**Code Agent (CodeLlama) Output:**")
                st.code(code_str, language=code_type)
            else:
                st.warning("No code block found in the base agent's response.")
            executed_sql = None
            sql_result = None
            py_result = None
            error_msg = None
            result_df = None
            if code_type == "sql" and code_str:
                if not re.search(r'\b(DELETE|DROP)\b', code_str, re.IGNORECASE):
                    try:
                        with sqlite3.connect(DB_PATH) as conn:
                            sql_result = pd.read_sql_query(code_str, conn)
                        st.write("**SQL Result:**")
                        st.dataframe(sql_result)
                        executed_sql = code_str
                        result_df = sql_result
                    except Exception as e:
                        error_msg = f"SQL execution error: {e}"
                        st.error(error_msg)
                else:
                    st.warning("SQL contains DELETE or DROP statement and was not executed.")
            elif code_type in ["python", "polars"] and code_str:
                exec_env = {"pd": pd, "pl": pl}
                for t, df in table_dfs.items():
                    exec_env[t] = df
                    exec_env[f"{t}_pl"] = pl.from_pandas(df)
                try:
                    import io
                    import contextlib
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        exec(code_str, exec_env)
                    py_result = output.getvalue()
                    st.write("**Python/Polars Output:**")
                    st.text(py_result if py_result else "(No output)")
                    for v in exec_env.values():
                        if isinstance(v, (pd.DataFrame, pl.DataFrame)):
                            result_df = v.to_pandas() if isinstance(v, pl.DataFrame) else v
                except Exception as e:
                    error_msg = f"Python/Polars execution error: {e}\n{traceback.format_exc()}"
                    st.error(error_msg)
            if result_df is not None:
                st.write("#### Save this result as a custom table:")
                save_table_name = st.text_input("Custom table name:", value=save_table_name, key="save_table_name_input")
                if st.button("Save Table"):
                    if not save_table_name:
                        st.error("Please provide a name for the custom table.")
                    elif save_table_name in st.session_state['custom_tables'] or save_table_name in tables:
                        st.error("A table with this name already exists. Please choose a different name.")
                    else:
                        st.session_state['custom_tables'][save_table_name] = result_df
                        st.success(f"Custom table '{save_table_name}' saved for this session.")
                        st.session_state['save_table_name'] = ''
            os.makedirs("DATABASE_QUERY_TOOL/outputs", exist_ok=True)
            with open(f"DATABASE_QUERY_TOOL/outputs/query_log.txt", "a") as f:
                f.write(f"Tables: {selected_tables}\nQuery: {user_query}\nPrompt: {prompt}\nPhi: {phi_content}\nCodeType: {code_type}\nCode: {code_str}\nSQLResult: {sql_result}\nPyResult: {py_result}\nError: {error_msg}\n---\n")
        except Exception as e:
            st.error(f"AI agent error: {e}") 