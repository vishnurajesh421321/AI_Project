import os

from langchain_core.tools import Tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    index_name = "developer-quickstart-py"

    # Connect to the existing Pinecone index
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"]
    )
    return vectorstore


def get_pinecone_tool():
    # This assumes you have the logic from your loader.py accessible
    vectorstore = get_vectorstore()  # Function we defined earlier
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return Tool(
        name="document_search",
        func=retriever.invoke,  # LangChain's standard retrieval method
        description="Use this tool to search the uploaded PDF or CSV documents for specific information."
    )

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
tools = [get_pinecone_tool()]

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 100),
            keep=("messages", 1)
        )
    ],
)