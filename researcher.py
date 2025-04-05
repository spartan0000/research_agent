
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


def summarize_node(state: State) -> State:
    """
    Summarize the text which is a compilation of abstracts from pubmed.  
    The summary should focus on how the abstract text is related to the query terms and should be as relevant as possible to the
    query terms.

    Parameters: state(State): The current state which contains the abstract text to summarize

    Returns: state(State): the updated state which includes a summary as well as a list of the articles
    """

    #define prompt template

    prompt = PromptTemplate(
        input_variables = ['text'],
        template = f"Summarize the following text: {summary}",
    )

    message = HumanMessage(content = prompt.format(text = state['text']))

    response = llm.invoke([message]).content.strip()

    return {'summary': response}

