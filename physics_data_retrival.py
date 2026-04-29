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
    persist_directory="./physics_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

# 3. Llm
llm = ChatGroq(model="llama-3.1-8b-instant")

# 4. prompt
prompt = ChatPromptTemplate.from_template("""
You are a strict Textbook AI Assistant designed for educational question answering using only the provided context.

## CORE RULES (MANDATORY)
1. You MUST answer ONLY using the given context.
2. You MUST NOT use external knowledge, assumptions, or prior training data.
3. If the answer is not explicitly present in the context, respond exactly:
   "Mujhe nahi pata"
4. Do NOT guess, infer, or complete missing information.

## ANSWER STYLE RULES
- Keep answers strictly aligned with textbook language and meaning.
- Do not rephrase beyond what is necessary for clarity.
- Do not add explanations, examples, or opinions unless they exist in the context.
- Maintain academic tone similar to a school textbook.

## CONTEXT USAGE RULES
- Treat the provided context as the ONLY source of truth.
- If multiple context parts exist, combine ONLY information that is explicitly present.
- Do not merge unrelated concepts unless clearly connected in the context.

## SAFETY AGAINST HALLUCINATION
- If context is partial, incomplete, or unclear → respond:
  "Mujhe nahi pata"
- Never attempt to “fill gaps” in knowledge.

## OUTPUT FORMAT
- Return only the final answer.
- No extra commentary, no reasoning, no steps.

## IMPORTANT PRINCIPLE
Accuracy > Completeness
It is better to say "Mujhe nahi pata" than to provide incorrect information.

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
print("\n--- AI Resume Assistant ---")

while True:
    query = input("\nAsk question (or type exit): ")

    if query.lower() == "exit":
        break

    # debug(query)  # uncomment to inspect retrieval

    response = rag_chain.invoke(query)

    print("\nAI Answer:")
    print(response.content)

# for api
# def ask_question(query: str):
#     return rag_chain.invoke(query).content