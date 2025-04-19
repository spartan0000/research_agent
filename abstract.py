import requests
import os
from Bio import Entrez
from typing import List
import time
from dotenv import load_dotenv

from openai import OpenAI


load_dotenv()

Entrez.email = 'bobthebuilder@mail.com'

def parse_pub_date(pub_date):
    if 'Year' in pub_date:
        year = pub_date['Year']
        month = pub_date.get('Month', '01')
        day = pub_date.get('Day', '01')
        return f'{year} - {month} - {day}'
    return 'Not Available'


def convert_query(text):

    
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {'role':'system', 'content':query_prompt},
            {'role':'user', 'content':text},
        ]
    )
    return response


def make_query(topic: str, start_date: str, end_date: str):
    #dates need to be entered as string in the format YYYY/MM/DD
    #topics = ['robotic colon resection']
    #start_date = '2018/01/01'
    #end_date = '2023/12/31'
    
    #format the date range for pubmed
    
    date_range = f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
    
    queries = f'{topic}[Title/Abstract] AND {date_range}'
    #populate queries based on topic and date range
    
    
    #just in case there are more than one topic in the query
    #for topic in topics:
    #    queries.append(f'{topic}[Title/Abstract] AND {date_range}')


    return queries
    #the queries are ready to send to pubmed


#process each query
def get_articles(query, n_results):
    results = []
    
    handle = Entrez.esearch(db = 'pubmed', term = query, retmax = n_results)
    record = Entrez.read(handle)
    id_list = record['IdList']
        
        #for each pmid, get information about the article
    for pmid in id_list:
        try:
            handle = Entrez.efetch(db = 'pubmed', id = pmid, retmode = 'xml')
            records = Entrez.read(handle)
        except Exception as e:
            print(f'Error fetching PMID {pmid}: {e}')

            
            #process each article
        for record in records['PubmedArticle']:
            article = record['MedlineCitation']['Article']
            title = article.get('ArticleTitle', 'Title Not Available')
            abstract = ' '.join(article['Abstract']['AbstractText']) if 'Abstract' in article else ''
            authors_list = ', '.join(a.get('ForeName', '') + ' ' + a.get('LastName', '') for a in article.get('AuthorList', [])) or 'Authors Not Available'
            journal = article['Journal'].get('Title', 'Journal Not Available')
            keywords = ', '.join(k['DescriptorName'] for k in record['MedlineCitation'].get('MeshHeadingList', [])) or 'Keyword Not Available'
            pub_date = parse_pub_date(article['Journal']['JournalIssue']['PubDate'])
            url = f"https://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
                
            new_result = {
                'PMID':pmid,
                'Title':title,
                'Abstract':abstract,
                'Journal':journal,
                'Keywords':keywords,
                'URL':url,
                'Publication Date':pub_date,
                }
            
            results.append(new_result)
             
    return results

def create_string(result): #need to convert the results above to a long string to input into the LLM

    text = ' '.join(f"{result[i].get('Title', '')} {result[i].get('Abstract', '')} {result[i].get('Keywords', '')}" for i in range(len(result)))

    return text
    
#text then can be directly sent to the LLM as an input

client = OpenAI(api_key = os.environ['OPENAI_API_KEY'])

system_prompt = '''You are a research assistant tasked with summarizing scientific articles.  Summarize the abstracts'''

def summarize_text(text):
    response = client.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {'role': 'system', 'content':system_prompt},
            {'role': 'user', 'content':text},
        ]
    )
    return response

if __name__ == '__main__':
    #run a test case when calling the script directly
    topics = ['robotic colon resection']
    start_date = '2020/01/01'
    end_date = '2023/12/31'
    queries = make_query(topics, start_date, end_date)
    results = get_articles(queries, 1)
    text = create_string(results)
    summary = summarize_text(text)
    print(summary.choices)