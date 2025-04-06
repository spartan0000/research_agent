
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from Bio import Entrez

#load functions from abstract.py
from abstract import get_articles, make_query, create_string

load_dotenv()
Entrez.email = os.environ['Entrez.email']

llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.2)



class State(TypedDict):
    query: str
    start_date: str
    end_date: str
    num_articles: int
    abstract_text: str
    summary: str


def get_article_node(state: State) -> State:
    """
    Get the abstract from pubmed and add it to the state
    """
    queries = make_query(state['query'], state['start_date'], state['end_date'])
    articles = get_articles(queries, state['num_articles'])
    abstract_text = create_string(articles)
    return {'abstract_text': abstract_text}


def summarize_node(state: State) -> State:
    """
    Summarize the text which is a compilation of abstracts from pubmed.  
    The summary should focus on how the abstract text is related to the query terms and should be as relevant as possible to the
    query terms.

    Parameters: state(State): The current state which contains the abstract text to summarize

    Returns: state(State): the updated state which includes a summary
    """
    text = state['abstract_text']

    #define prompt template

    prompt = PromptTemplate(
        input_variables = ['text'],
        template = f"Summarize the following abstracts: {text}",
    )

    message = HumanMessage(content = prompt.format(text = text))

    response = llm.invoke([message]).content.strip()

    return {'summary': response}

builder = StateGraph(State)

builder.add_node("get_abstracts", get_article_node)
builder.add_node("summarize", summarize_node)

builder.set_entry_point("get_abstracts")
builder.add_edge("get_abstracts", "summarize")

graph = builder.compile()



if __name__ == "__main__":
    initial_state = {"query": "thyroidectomy", "start_date": "2023/01/01", "end_date": "2025/01/01", "num_articles": 1}
    result = graph.invoke(initial_state)
    print(result['summary'])