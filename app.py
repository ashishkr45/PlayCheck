import warnings
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!*")

from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import random
import os

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

# Google Gemini and LangSmith imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()

class PlayCheckState(TypedDict):
	message: Annotated[List[BaseMessage], add_message]
	game_name: Optional[str]
	user_specs: Optional[Dict[str, str]]
	game_requirement: Optional[Dict[str, Dict[str, str]]]

@tool
def get_system_specifications()