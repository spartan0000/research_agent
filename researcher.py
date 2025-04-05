
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


load_dotenv()

llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.2)

class State(TypedDict):
    text: str
    articles: List[str]
    summary: str

