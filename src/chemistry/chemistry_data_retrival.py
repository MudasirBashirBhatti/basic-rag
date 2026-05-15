import re

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# =========================
# 1. EMBEDDINGS + VECTOR DB
# =========================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="./chemistry_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# =========================
# 2. SECTION DETECTOR
# =========================

def extract_section(query):
    match = re.search(r"\d+\.\d+", query)
    return match.group() if match else None

# =========================
# 3. SMART RETRIEVAL
# =========================

def get_docs(query):
    docs = retriever.invoke(query)

    section = extract_section(query)

    # filter by section if present (17.1 etc.)
    if section:
        filtered = [
            d for d in docs
            if d.metadata.get("section") and section in str(d.metadata["section"])
        ]

        # fallback if filtering removes everything
        if filtered:
            docs = filtered

    return docs

# =========================
# 4. FORMAT CONTEXT
# =========================

def format_docs(docs):
    context = ""

    for d in docs:
        context += f"""
(Page: {d.metadata.get('page', 'N/A')})
(Section: {d.metadata.get('section', 'N/A')})

{d.page_content}

-------------------------
"""
    return context

# =========================
# 5. LLM (STRICT MODE)
# =========================

llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template("""
You are a STRICT CHEMISTRY TEACHER AI.

RULES:
- Use ONLY the provided context
- Do NOT use outside knowledge
- If answer is not in context → say INSUFFICIENT CONTEXT
- If someone asks for a section → return the section content from context, do NOT answer from memory
- Always include the page number in your answer (e.g. "As per page 23...")
- If multiple sections are relevant, include all of them with page numbers
- If the question is about a specific section (e.g. "What is 17.1?") → return that section content with page number, do NOT answer from memory
- If someone asks to generate questions -> return the output in a JSON array format like: ["Question 1?", "Question 2?"]
- If the question is not about chemistry → say THIS IS A CHEMISTRY ASSISTANT, I CANNOT ANSWER NON-CHEMISTRY QUESTIONS
- If someone asks for generating questions and answers → return the output in this JSON format:
  [{{"question": "Question 1?", "answer": "Answer 1."}},
   {{"question": "Question 2?", "answer": "Answer 2."}}]

- If someone asks for MCQ questions → return the output in this exact JSON format:
  [{{"question": "Question 1?",
     "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
     "answer": "A"}}]

Context:
{context}

Question:
{question}

Answer:
""")

def run_llm(context, question):
    chain = prompt | llm

    return chain.invoke({
        "context": context,
        "question": question
    }).content

# =========================
# 6. MAIN ASK FUNCTION
# =========================

def ask(query):
    docs = get_docs(query)
    context = format_docs(docs)
    return run_llm(context, query)

# =========================
# 7. CLI RUNNER
# =========================

if __name__ == "__main__":

    print("\n--- AI CHEMISTRY RAG ASSISTANT ---")

    while True:
        q = input("\nAsk: ")

        if q.lower() == "exit":
            break

        print("\n", ask(q))