
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

import argparse

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
    
    extract_prompt = PromptTemplate(
        input_variables = ['user_input'],
        template = """
        Given the following user input, extract the search query terms of interest, the start date, end date, and the number of articles requested
        Return a **valid JSON** object with the following key:value pairs
        {{
        "query": the search term of interest as a string
        "start_date": a string in the format YYYY-MM-DD
        "end_date": a string in the format YYYY-MM-DD
        "num_articles": an integer
        }}
        
        user_input: {user_input}"""
    )
    user_input = state['user_input']

    
    prompt = extract_prompt.format(user_input = user_input) #user_input var for the prompt = user_input from state['user_input']

    response = llm.invoke(prompt)

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        raise ValueError("Could not parse LLM output as JSON")

    #print(parsed['query'])
    return {
        'query': parsed['query'],
        'start_date': parsed['start_date'],
        'end_date': parsed['end_date'],
        'num_articles': parsed['num_articles'],

    }


def get_article_node(state: State) -> State:
    """
    Get the abstract from pubmed and add it to the state
    """
    #print('input state:', state)
    date_range = f'("{state['start_date']}"[Date - Publication] : "{state['end_date']}"[Date - Publication])'
    
    queries = f'{state['query']}[Title/Abstract] AND {date_range}'

    print(queries)



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



def main(): 
    parser = argparse.ArgumentParser(description = "Generate summary from pubmed query")
    parser.add_argument("--topic", required = True, help = "Topic of interest")
    parser.add_argument("--start_date", default = "2023/01/01", help = "Start date in YYYY/MM/DD format")
    parser.add_argument("--end_date", default = 2024/12/31, help = "End date in YYY/MM/DD format")
    parser.add_argument("--n_articles", default = 5, type = int, help = "Number of articles to review")



if __name__ == "__main__":
    query_text = """
    I need some articles about robotic colon resection starting january 1, 2023 to dec 31, 2024
    """


    
    initial_state = {'user_input': query_text}
    result = graph.invoke(initial_state)
    print(result)