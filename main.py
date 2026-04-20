from router import pick_bot_for_post
from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()



llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)


personas = {
    "A": "Tech maximalist. Loves AI, crypto, startups, Elon Musk. Very optimistic about future.",
    "B": "Skeptic. Critical of big tech, capitalism, privacy issues, inequality.",
    "C": "Finance expert. Talks about markets, ROI, trading, investments."
}

class GraphState(TypedDict):
    topic: str
    bot_id: str
    search_query: str
    search_result: str
    final_post: str

def mock_search(query: str):
    if "crypto" in query.lower():
        return "Bitcoin surges after ETF approval"
    elif "ai" in query.lower():
        return "New AI model beats GPT-4 benchmarks"
    elif "stock" in query.lower():
        return "Stock market crashes due to global recession fears"
    return "General tech news"

def decide_search(state: GraphState):
    prompt = f"""
    Convert this topic into ONE short search query.
    Only return the query.

    Topic: {state['topic']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"search_query": response.content}


def search_node(state: GraphState):
    return {"search_result": mock_search(state["search_query"])}


def generate_post(state: GraphState):
    persona = personas[state["bot_id"]]

    prompt = f"""
    You are a social media persona.

    Persona: {persona}

    Topic: {state['topic']}
    Info: {state['search_result']}

    Write a short engaging post in this persona's style.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"final_post": response.content}


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
    topic = "AI replacing jobs"

    
    route = pick_bot_for_post(topic)

    
    result = app.invoke({
        "topic": topic,
        "bot_id": route["bot_id"]
    })

    
    final_output = {
        "bot_id": route["bot_id"],
        "topic": topic,
        "post_content": result["final_post"]
    }

    print(final_output)