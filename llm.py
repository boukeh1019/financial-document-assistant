# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.2,
)

response = llm.invoke("Find me information about Boke Lets'oara")
print(response.content)
