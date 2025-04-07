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


#just storing some stuff that is a work in progress here

def get_query_parameters(state):
    system_prompt = """Extract the search query terms of interest, the start date, end date, and number of articles required."""
    
    

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
