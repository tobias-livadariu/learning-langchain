from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# set env key
load_dotenv()

# initialize search tool
search_tool = TavilySearch(max_results=2)
# # sample usage
# query = "What is the weather in Los Angeles"
# search_results = search_tool.invoke(query)
# print(search_results)

# initialize model
model = init_chat_model("gpt-4o-mini", model_provider="openai")
tools = [search_tool]
# memory = MemorySaver()

# create agent
agent = create_react_agent(model, tools, 
                          #  checkpointer=memory
                          )

# response = agent.invoke({"messages": [("user", "Hi")]})
# response["messages"][-1].pretty_print()

# response = agent.invoke({"messages": [("user", "What is the weather in Ontario, North York today?")]})
# response["messages"][-1].pretty_print()

# response = agent.invoke({"messages": [("user", "Who is the current president of the USA?")]})
# response["messages"][-1].pretty_print()

response = agent.invoke({"messages": [("user", "Who is the current president of the USA?")]})
for res in response["messages"]:
  res.pretty_print()