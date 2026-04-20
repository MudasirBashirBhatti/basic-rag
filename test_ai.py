from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant") 

user_input = input("Enter your question: ")

response = llm.invoke(user_input)
print(response.content)
