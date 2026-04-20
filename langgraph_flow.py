import os
from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=  os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

# 🔹 Define state
class GraphState(TypedDict):
    topic: str
    search_query: str
    search_result: str
    final_post: str


# 🔹 Mock search tool
def mock_search(query: str):
    if "crypto" in query.lower():
        return "Bitcoin surges after ETF approval"
    elif "ai" in query.lower():
        return "New AI model beats GPT-4 benchmarks"
    elif "stock" in query.lower():
        return "Stock market crashes due to global recession fears"
    return "General tech news"


# Decide search query
def decide_search(state: GraphState):
    prompt = f"Convert this topic into a search query.Only return query,nothing else: {state['topic']}"

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "search_query": response.content
    }


# Call tool
def search_node(state: GraphState):
    result = mock_search(state["search_query"])
    return {
        "search_result": result
    }


# Generate post
def generate_post(state: GraphState):
    prompt = f"""
    You are a social media bot.

    Topic: {state['topic']}
    Search Info: {state['search_result']}

    Write a short engaging post.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "final_post": response.content
    }


# 🔹 Build Graph
graph = StateGraph(GraphState)

graph.add_node("decide", decide_search)
graph.add_node("search", search_node)
graph.add_node("generate", generate_post)

graph.set_entry_point("decide")

graph.add_edge("decide", "search")
graph.add_edge("search", "generate")

graph.set_finish_point("generate")

app = graph.compile()
if __name__ == "__main__":
    result = app.invoke({
        "topic": "AI replacing jobs"
    })

    print(result)