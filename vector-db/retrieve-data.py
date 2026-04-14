import os

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from pinecone import Pinecone



# 1. Initialize Google Generative AI LLM
# Ensure GOOGLE_API_KEY is set in your environment
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# 2. Initialize Pinecone and create a retriever
# This part uses your existing Pinecone setup logic
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

pc = Pinecone(api_key=pinecone_api_key)
index_name = "developer-quickstart-py" # Use your index name

# Assuming embeddings are consistent with what was used for indexing
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# Connect to the existing Pinecone index
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# 3. Create a Tool from the retriever
# The tool's description is crucial for the agent to know when to use it
retrieval_tool = Tool(
    name="pinecone_retriever",
    func=retriever.invoke,
    description="Useful for answering questions about uploaded documents by searching the Pinecone vector database.",
)

# 4. Define the agent's prompt
# The prompt guides the agent's behavior and tool usage
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Use the provided tools to answer questions."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# 5. Create the agent
# This agent can call tools based on the prompt and LLM's reasoning
agent = create_tool_calling_agent(llm, [retrieval_tool], prompt)

# 6. Create an AgentExecutor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=[retrieval_tool], verbose=True)

# Example usage
response = agent_executor.invoke({"input": "What is mentioned about file uploads?"})
print(response["output"])