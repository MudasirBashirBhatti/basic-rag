from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant") 

response = llm.invoke("What is the capital of France?")
print(response.content)
