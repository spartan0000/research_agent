from openai import OpenAI
from typing import TypedDict

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from tavily import TavilyClient

import argparse
import json
import requests
from dataclasses import dataclass
import datetime

load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
NEWSAPI_API_KEY = os.getenv('NEWSAPI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY ')


llm = ChatOpenAI(model = 'gpt-3.5-turbo', temperature = 0.6)

@dataclass
class NewsState:
    user_input: str 
    user_bias: str   
    query: str = None
    news_aggregate: str = None
    tavily_aggregate: str = None
    combined_text: str = None
    news_summary: str = None
    neutral_summary: str = None
    right_summary: str = None
    left_summary: str = None
    final_summary:str = None
    
        

def format_query_node(state: NewsState) -> NewsState:  #formatted query needed for NEWSAPI.  Tavily takes free text query
    extract_prompt = PromptTemplate(
        input_variables = ['user_input'],
        template = """
        Given the following user input, extract the search term of interest.  Return a **valid JSON** object with the follwing key:value pairs
        {{
            "query": the search term of interest as a string
            }}

        user_input : {user_input}

        """)
    user_input = state.user_input

    prompt = extract_prompt.format(user_input = user_input)

    response = llm.invoke(prompt)

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        raise ValueError('Could not parse LLM output as JSON')
    
    state.query = parsed['query']
    
    return state
                         

def get_news_api_node(state: NewsState) -> NewsState:
    '''Get news from NEWSAPI and convert it to a large string for the LLM'''

    query = state.query
    language = 'en'
    today = (datetime.datetime.now())
    d = datetime.timedelta(days = 7)
    last_week = today - d
    today_date = f'{today.year}-{today.month}-{today.day}'
    lw_date = f'{last_week.year}-{last_week.month}-{last_week.day}'
    url = f'https://newsapi.org/v2/everything?q={query}&from={lw_date}&to={today_date}&sortBy=relevancy&language={language}&apiKey={NEWSAPI_API_KEY}'

    response = requests.get(url)
    
    output = response.json()

    articles = output['articles']
    #print(len(articles))
    news_aggregate = ''.join(f'{articles[i].get("title", "")} {articles[i].get("description", "")} {articles[i].get("content", "")} {articles[i].get("url", "")}' for i in range(len(articles)))

    state.news_aggregate = news_aggregate
    
    return state

    
def get_tavily_node(state: NewsState) -> NewsState:
    '''Get news from Tavily and convert to large string for the LLM'''
    
    TAVILY_API_KEY = os.environ['TAVILY_API_KEY']
    client = TavilyClient(TAVILY_API_KEY)

    response = client.search(
        query = state.user_input
    )

    results = response['results']
    
    aggregate = ' '.join(f'{results[i].get("title", "")} {results[i].get("content", "")}' for i in range(len(results)))

    state.tavily_aggregate = aggregate

    return state
    
    
def summarize_news_node(state: NewsState) -> NewsState:
    '''
    Summarizes the aggregate text string of news information from the query
    '''
    combined_text = str(state.news_aggregate) + str(state.tavily_aggregate)    
    
    
    prompt_text = 'Summarize the following text as if writing an article for an objective, unbiased news source.'

    
    prompt = PromptTemplate(
        input_variables = ['combined_text'],
        template = f"{prompt_text}:{combined_text}"
    )

    message = HumanMessage(content = prompt.format(combined_text = combined_text))
    response = llm.invoke([message]).content.strip()
    
    state.news_summary = response
    return state
    

def right_leaning_news_node(state: NewsState) -> NewsState:
    '''
    takes neutral news summary and spins it with a right leaning bias
    '''

    summary = state.news_summary


    prompt_text = '''Take this summary and rewrite with a right leaning bias as if you are a reporter for a right wing news outlet such as OAN or Breitbart.
    The summary should be based on facts but can use hyperbole or exaggeration.  The biased summary should be something that a sterotypical Trump supporter 
    would enjoy reading and one that a Democratic voter would find unappealing
    '''

    prompt = PromptTemplate(
        input_variables = ['summary'],
        template = f"{prompt_text}:{summary}",
    )

    message = HumanMessage(content = prompt.format(summary = summary))
    
    response = llm.invoke([message]).content.strip()
    
    state.right_summary = response
    
    return state

def left_leaning_news_node(state: NewsState) -> NewsState:
    '''Takes a neutral summary and spins it with a left leaning bias'''
    
    summary = state.news_summary
    
    prompt_text = """Take this summary and rewrite it with a left leaning bias as if you are a reporter for a left leaning news outlet such as MSNBC or if you were a political operative
    aligned with radical progressive and left wing ideologies.  The biased summary should be something that a stereotypical Trump supporter would find repulsive and one that someone who 
    voted for Hillary Clinton, Joe Biden and Kamala Harris would find appealing"""

    prompt = PromptTemplate(
        input_variables = ['summary'],
        template = f"{prompt_text}:{summary}",
    )
    message = HumanMessage(content = prompt.format(summary = summary))
    response = llm.invoke([message]).content.strip()
    state.left_summary = response
    
    return state

def neutral_news_node(state: NewsState) -> NewsState:
    '''Takes a neutral summary and spits out the same thing'''
    summary = state.news_summary
    prompt_text = '''Take this summary and make minor edits while keeping it largely the same'''
    prompt = PromptTemplate(
        input_variables = ['summary'],
        template = f"{prompt_text}:{summary}",
    )
    message = HumanMessage(content = prompt.format(summary = summary))
    response = llm.invoke([message]).content.strip()
    state.neutral_summary = response
    return state


def publish(state: NewsState) -> NewsState:
    '''Takes one of 3 summaries and polishes it'''
    if state.user_bias == 'left':
        summary = state.left_summary
    elif state.user_bias == 'right':
        summary = state.right_summary
    else:
        summary = state.news_summary
      
    prompt_text = """
    Take the following summary and expand it and rewrite it as a polished article for a general audience.  Make it clear and insightful like a blog post.
    Use a bright and natural tone like something you'd hear in a TED talk or in a popular magazine.
    You can expand the summary as needed but do not make up any new facts.
    """
    prompt = PromptTemplate(
        input_variables = ['summary'],
        template = f"{prompt_text}:{summary}",
    )
    message = HumanMessage(content = prompt.format(summary = summary))
    response = llm.invoke([message]).content.strip()
    state.final_summary = response
    return state

def run_news_agent(state):
    builder = StateGraph(NewsState)
    builder.add_node("format_query", format_query_node)
    builder.add_node("get_news", get_news_api_node)
    builder.add_node("summarize", summarize_news_node)
    builder.add_node("right_leaning_news", right_leaning_news_node)
    builder.add_node("left_leaning_news", left_leaning_news_node)
    builder.add_node("neutral_news", neutral_news_node)
    builder.add_node("publish", publish)

    def route(state):
        bias = state.user_bias
        if bias == 'right':
            return "right_leaning_news"
        elif bias == 'left':
            return "left_leaning_news"
        else:
            return "neutral_news"
    builder.add_conditional_edges("summarize", route)

    builder. add_edge("format_query", "get_news")
    builder.add_edge("get_news", "summarize")
    #builder.add_edge("summarize", "right_leaning_news")
    #builder.add_edge("summarize", "left_leaning_news")
    #builder.add_edge("summarize", "neutral_news")

    builder.add_edge("right_leaning_news", "publish")
    builder.add_edge("left_leaning_news", "publish")
    builder.add_edge("neutral_news", "publish")

    builder.set_entry_point("format_query")
    builder.set_finish_point("publish")

    graph = builder.compile()
    result = graph.invoke(state)
    return result

if __name__ == '__main__':
    state = NewsState(user_input = 'What has donald trump been up to?', user_bias = 'left')
    news = run_news_agent(state)
    print(news['final_summary'])