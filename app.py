import warnings
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!*")

from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Gemini + dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

# ---------------------- STATE ----------------------
class PlayCheckState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    game_name: Optional[str]
    user_specs: Optional[Dict[str, str]]
    game_requirement: Optional[Dict[str, Dict[str, str]]]

# ---------------------- TOOL ----------------------
HEADERS = {"User-Agent": "Mozilla/5.0"}

@tool
def scrape_steam_requirements(game_name: str) -> dict:
    """Scrape system requirements for a PC game from Steam Store page."""
    try:
        search_url = f"https://store.steampowered.com/search/?term={game_name}"
        search_res = requests.get(search_url, headers=HEADERS)
        search_soup = BeautifulSoup(search_res.text, "html.parser")

        game_link_tag = search_soup.select_one(".search_result_row")
        if not game_link_tag:
            return {"error": "Game not found on Steam"}

        game_url = game_link_tag["href"].split("?")[0]
        game_page = requests.get(game_url, headers=HEADERS)
        game_soup = BeautifulSoup(game_page.text, "html.parser")

        min_req = game_soup.select_one(".game_area_sys_req_leftCol")
        rec_req = game_soup.select_one(".game_area_sys_req_rightCol")

        return {
            "game": game_name,
            "steam_url": game_url,
            "minimum": min_req.get_text("\n", strip=True) if min_req else "Not found",
            "recommended": rec_req.get_text("\n", strip=True) if rec_req else "Not found"
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------- LLM SETUP ----------------------
tools = [scrape_steam_requirements]
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    convert_system_message_to_human=True
).with_config({
    "system_message": """You are PlayCheck, a gaming assistant. 
When a user asks whether their PC can run a specific game, you must:
1. Extract the game name and user's PC specs.
2. Call the 'scrape_steam_requirements' tool to fetch system requirements.
3. Use the tool result to generate a compatibility response.

If system requirements are missing, explain that clearly. If they are available, compare minimum/recommended specs against the user's hardware and suggest settings (Low/Medium/High).
"""
})
llm_with_tools = llm.bind_tools(tools)

# ---------------------- AGENT NODES ----------------------
def chatbot_node(state: PlayCheckState) -> Dict:
    try:
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)

        # Debug output
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("🧪 Gemini requested a tool call:", response.tool_calls)
        else:
            print("🤔 No tool was called by Gemini.")

        return {"messages": [*messages, response]}
    except Exception as e:
        print(f"❌ Error in chatbot_node: {e}")
        return {"messages": state["messages"]}

def should_continue(state: PlayCheckState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return "end"

# ---------------------- GRAPH ----------------------
def build_playcheck_graph():
    graph = StateGraph(PlayCheckState)
    graph.add_node("agent", chatbot_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()

# ---------------------- RUN LOOP ----------------------
def run_playcheck():
    app = build_playcheck_graph()

    print("\n🎮 Welcome to PlayCheck — Can Your PC Run It?")
    print("💡 Try: Can I run God of War on i5 with 8GB RAM and GTX 1650?\n")

    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        state = {
            "messages": [HumanMessage(content=user_input)],
            "game_name": None,
            "user_specs": None,
            "game_requirement": None
        }

        result = app.invoke(state)
        messages = result["messages"]
        last_msg = messages[-1]

        if isinstance(last_msg, AIMessage):
            print(f"🤖 PlayCheck: {last_msg.content}")
        elif isinstance(last_msg, ToolMessage):
            print(f"🛠️ Tool Result: {last_msg.content}")

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    run_playcheck()