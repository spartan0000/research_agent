
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

from openai import OpenAI

import json

#load functions from abstract.py
from abstract import get_articles, make_query, create_string, convert_query

load_dotenv()
Entrez.email = os.environ['EMAIL']

llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.2)



class State(TypedDict):
    user_input: str
    query: str
    start_date: str
    end_date: str
    num_articles: int
    abstract_text: str
    summary: str

#need a way to get free text query into the format we want it to query pubmed

def format_query(state: State) -> State:
    """
    take the user input query and return a dictionary that contains the query, start date, end date, and number of articles
    """
    system_prompt = """Extract the search query terms of interest, the start date, end date, and number of articles required.
    
    """

    input_text = state['user_input']
    result = llm.invoke(
        messages = [
            {'role':'system', 'content': system_prompt},
            {'role':'user', 'content': input_text},
        ],
        functions = [
            {
                "name": "generate_query_parameters",
                "description": "Extracts query terms, dates, and number of articles",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type":"string"},
                        "start_date": {"type":"string"},
                        "end_date": {"type":"string"},
                        "num_articles": {"type":"integer"},
                    },
                    "required": ["query", "start_date", "end_date", "num_articles"]
                }
            }
        ],
        function_call = {"name": "generate_query_parameters"}
    )

    args = result.additional_kwargs['function_call']['arguments']
    parsed = json.loads(args)

    return {
        "query":parsed["query"],
        "start_date":parsed["start_date"],
        "end_date":parsed["end_date"],
        "num_articles":parsed["num_articles"],
    }







def get_article_node(state: State) -> State:
    """
    Get the abstract from pubmed and add it to the state
    """
    #print('input state:', state)
    date_range = f'("{state['start_date']}"[Date - Publication] : "{state['end_date']}"[Date - Publication])'
    
    queries = f'{state['query']}[Title/Abstract] AND {date_range}'



    #print('queries:', queries)
    articles = get_articles(queries, state['num_articles'])
    abstract_text = create_string(articles)
    return {'abstract_text': abstract_text}


def summarize_node(state: State) -> State:
    """
    Summarize the text which is a compilation of abstracts from pubmed.  
    

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

builder.add_node("format_query", format_query)
builder.add_node("get_abstracts", get_article_node)
builder.add_node("summarize", summarize_node)

builder.set_entry_point("format_query")
builder.add_edge("format_query", "get_abstracts")
builder.add_edge("get_abstracts", "summarize")


graph = builder.compile()



if __name__ == "__main__":
    query_text = """
    I need some articles about robotic colon resection starting january 1, 2023 to dec 31, 2024
    """


    
    initial_state = convert_query(query_text)
    result = graph.invoke(initial_state)
    print(result)