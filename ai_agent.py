
# Step 1 - Load API keys
import os
from dotenv import load_dotenv

load_dotenv()  # this line was missing — without it, everything below returns None

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Step 2 - Set up the LLM
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
    
groq_llm = ChatGroq(model="openai/gpt-oss-120b")  # llama-3.3-70b-versatile was just deprecated by Groq
#groq_llm = ChatGroq(model="openai/gpt-oss-20b")
openai_llm = ChatOpenAI(model="gpt-4o-mini")  # gpt-4o-mini is the most cost-effective OpenAI model for agents
search_tool = TavilySearch(max_results=5)  # this tool will allow the agent to perform web searches

# Step 3 - Set up the agent
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

system_prompt = "Act as a helpful assistant that can answer questions clearly and concisely."

agent = create_agent(
    model=groq_llm,
    tools=[search_tool],
    system_prompt=system_prompt,
)

query = "what is the latest news of USA?"
state = {"messages": query}
response = agent.invoke(state)
print("Agent Response:", response["messages"][-1].content)