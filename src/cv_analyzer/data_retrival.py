from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. embeddings (must match ingestion)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. load vector db
vector_db = Chroma(
    persist_directory="./my_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

# 3. Llm
llm = ChatGroq(model="llama-3.1-8b-instant")

# 4. prompt
prompt = ChatPromptTemplate.from_template("""
You are a strict Resume AI assistant.

RULES:
- Only use given context
- Do not guess
- If answer not in context say: "Mujhe nahi pata"

Context:
{context}

Question:
{question}

Answer:
""")

# 5. format context
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# 6. rag pipeline
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# 8. querying
# for cli
# print("\n--- AI Resume Assistant ---")

# while True:
#     query = input("\nAsk question (or type exit): ")

#     if query.lower() == "exit":
#         break

#     # debug(query)  # uncomment to inspect retrieval

#     response = rag_chain.invoke(query)

#     print("\nAI Answer:")
#     print(response.content)

# for api
def ask_question(query: str):
    return rag_chain.invoke(query).content