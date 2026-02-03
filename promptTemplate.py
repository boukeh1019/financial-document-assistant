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
# prompt = ChatPromptTemplate.from_template("Tell a joke about {subject}")

# An advanced prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("System", "You are a computer Science student at MIT. Write a unique prompt structure to follow when prompting the following LLM."),
        ("human", "{input}")
    ]
)

# Create LLM chain
chain = prompt | llm


response = chain.invoke({"input": "chatgpt"})

print(response)