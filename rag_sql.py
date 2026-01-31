from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re


# -------------------- PDF TEXT --------------------
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


# -------------------- VECTOR STORE --------------------
def build_vector_store(text):
    chunks = [c for c in text.split("\n\n") if len(c.strip()) > 30]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return chunks, model, index


# -------------------- SQL GENERATOR --------------------
def generate_sql(user_query, chunks, model, index):
    q = user_query.lower()

    # ----- TABLE DETECTION -----
    tables = []
    if "employee" in q:
        tables.append("employees")
    if "department" in q:
        tables.append("departments")
    if "job" in q:
        tables.append("jobs")

    if not tables:
        return "❌ No table detected from query"

    # ----- BASE SELECT -----
    select_clause = "SELECT *"

    if "first name" in q or "last name" in q or "name" in q:
        select_clause = "SELECT e.first_name, e.last_name"

    if "salary" in q and "name" in q:
        select_clause = "SELECT e.first_name, e.last_name, e.salary"

    if "count" in q:
        select_clause = "SELECT COUNT(*)"

    # ----- FROM + JOIN -----
    from_clause = "FROM employees e"

    join_clause = ""

    if "department" in q:
        join_clause += " JOIN departments d ON e.department_id = d.department_id"

    if "job" in q:
        join_clause += " JOIN jobs j ON e.job_id = j.job_id"

    # ----- WHERE CONDITIONS -----
    conditions = []

    # salary condition
    salary_match = re.search(r"salary\s*(>|>=|<|<=)\s*(\d+)", q)
    if salary_match:
        op, value = salary_match.groups()
        conditions.append(f"e.salary {op} {value}")

    # hire date condition
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", q)
    if "after" in q and date_match:
        conditions.append(f"e.hire_date > '{date_match.group()}'")

    # department name condition
    dept_match = re.search(r"department\s+(\w+)", q)
    if dept_match:
        conditions.append(f"d.department_name = '{dept_match.group(1)}'")

    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    # ----- FINAL SQL -----
    sql = f"""
{select_clause}
{from_clause}
{join_clause}
{where_clause};
""".strip()

    return sql
