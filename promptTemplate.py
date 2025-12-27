from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate 

# Instantiation of the model
llm = ChatOllama(
    temperature=0.7,
    model="llama3.1:8b",
)

# building a basic template
prompt = ChatPromptTemplate.from_template("Tell a joke about {subject}")

# Create LLM chain
chain = prompt | llm


response = chain.invoke({"subject": "dog"})

print(response)