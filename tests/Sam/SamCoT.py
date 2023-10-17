import os
import traceback
import requests
import json
import nyt
import ipinfo
import random
import socket
import time
import numpy as np
import faiss
import pickle
import hashlib
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from promptrix.ConversationHistory import ConversationHistory
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.ChoiceResponseValidator import ChoiceResponseValidator
from alphawave.TOMLResponseValidator import TOMLResponseValidator
from alphawave_pyexts import utilityV2 as ut
from alphawave_pyexts import LLMClient as llm
from alphawave_pyexts import Openbook as op
from alphawave.OSClient import OSClient
from alphawave.alphawaveTypes import PromptCompletionOptions

# Encode titles to vectors using SentenceTransformers 
from sentence_transformers import SentenceTransformer
from scipy import spatial

NYT_API_KEY = os.getenv("NYT_API_KEY")
sections = ['arts', 'automobiles', 'books/review', 'business', 'fashion', 'food', 'health', 'home', 'insider', 'magazine', 'movies', 'nyregion', 'obituaries', 'opinion', 'politics', 'realestate', 'science', 'sports', 'sundayreview', 'technology', 'theater', 't-magazine', 'travel', 'upshot', 'us', 'world']


host = '127.0.0.1'
port = 5004

def generate_faiss_id(document):
    hash_object = hashlib.sha256()
    hash_object.update(document.encode("utf-8"))
    hash_value = hash_object.hexdigest()
    faiss_id = int(hash_value[:8], 16)
    return faiss_id

class SamInnerVoice():
    def __init__(self, model):
        self.client = OSClient(api_key=None)
        self.functions = FunctionRegistry()
        self.tokenizer = GPT3Tokenizer()
        self.memory = VolatileMemory({'input':'', 'history':[]})
        self.max_tokens = 4000
        self.nytimes = nyt.NYTimes()
        self.news, self.details = self.nytimes.headlines()
        self.keys_of_interest = ['title', 'abstract', 'uri']
        self.model = model
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        docHash_loaded = False
        try:
            self.docHash = {}
            self.metaData = {}
            with open('SamDocHash.pkl', 'rb') as f:
                data = pickle.load(f)
                self.docHash = data['docHash']
                self.metaData = data['metaData']
            docHash_loaded = True
            try:
                self.semanticIDMap = faiss.read_index("SamVectors.faiss")
            except:
                # tbd: recover faiss index from docHash!
                self.semanticIDMap = faiss.IndexIDMap(faiss.IndexFlatL2(384))
                faiss.write_index(self.semanticIDMap, "SamVectors.faiss")
        except Exception as e:
            # no docHash, so faiss is useless, reinitialize both
            self.docHash = {}
            self.metaData = {}
            with open('SamDocHash.pkl', 'wb') as f:
                data = {}
                data['docHash'] = self.docHash
                data['metaData'] = self.metaData
                pickle.dump(data, f)
            self.semanticIDMap = faiss.IndexIDMap(faiss.IndexFlatL2(384))
            faiss.write_index(self.semanticIDMap, "SamVectors.faiss")
            
    def search_titles(self, query, news_details):
        titles = []; articles = []
        for key in news_details.keys():
            for item in news_details[key]:
                titles.append(item['title'])
                articles.append(item)
        title_embeddings = self.embedder.encode(titles)
        query_embedding = self.embedder.encode(query)
        # Find closest title by cosine similarity
        cos_sims = spatial.distance.cdist([query_embedding], title_embeddings, "cosine")[0]
        #for n, title in enumerate(titles):
        #    print(f'{n}, {cos_sims[n]}, {titles[n]}')
        
        most_similar = cos_sims.argmin()
        print(f'similar title {titles[most_similar]}')
        return articles[most_similar]
    
    def sentiment_analysis(self, history):
        pass


    def recall(self, tags, query):
        if tags =='cancel':
            return []

        tags = tags.split(',')
        embed = self.embedder.encode(query)
        distances, ids = self.semanticIDMap.search(embed.reshape(1,-1), 4)
        print("Distances:", distances)
        print("Id:",ids)
        texts = []
        for id in ids[0]:
            if id in self.docHash:
                if id in self.metaData and 'tags' in self.metaData[id]:
                    itemTags = self.metaData[id]['tags']
                    for tag in tags:
                        if tag in itemTags:
                            texts.append(self.docHash[id])
                            break
        return texts
    
    def remember(self, tags, text):
        if tags =='cancel':
            return
        tags = tags.split(',')
        embed = self.embedder.encode(text)
        id = generate_faiss_id(text)
        if id in self.docHash:
            print('duplicate, skipping')
            return
        self.semanticIDMap.add_with_ids(embed.reshape(1,-1), np.array([id]))
        self.docHash[id] = text
        self.metaData[id] = {"tags": tags}
        # and save - write docHash first, we can always recover from that.
        with open('SamDocHash.pkl', 'wb') as f:
            data = {}
            data['docHash'] = self.docHash
            data['metaData'] = self.metaData
            pickle.dump(data, f)
        faiss.write_index(self.semanticIDMap, "SamVectors.faiss")

    def action_selection(self, input, response, profile, news_details):
        #
        ## see if an action is called for given conversation context and most recent exchange
        #
        #print(news_details.keys())
        self.articles = []
        for key in news_details.keys():
            for item in news_details[key]:
                self.articles.append({"title": item['title'],
                                      #"abstract": item['abstract']
                                      })
        print(f'SamCoT profile {profile}')
        prompt_text = f"""Given the following user-provided recent exchange, your task is to determine if profile\n{profile}\n should act at this time. Only select an action if explicity called for by the user in the recent exchange. The usual default should be to choose 'none' as the action.
The following New York Times articles are available:
{self.articles}

Respond using the following TOML format:
[RESPONSE]
action="<action name>"
value= "<action argument>"
[STOP]


The actions available are:
action_name\naction_argument\tdescription
article\t<article title>\t# retrieve a NYTimes article.
web\t<search query string>\t# perform a web search, using the <search query string> as the subject of the search.
wiki\t<search query string>\t# search the local wikipedia database.
none\t'none'\t# no action is needed.
"""
        action_validation_schema={
            "action": {
                "type":"string",
                "required": True,
                "meta": "<action to perform>"
            },
            "value": {
                "type":"string",
                "required": True,
                "meta": "argument for action"
            }
        }


        #print(f'action_selection {input}\n{response}')
        prompt = Prompt([
            SystemMessage(prompt_text),
            UserMessage('Exchange: {{$input}}'),
            AssistantMessage(' ')
        ])
        prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, max_tokens=50)

        summary_prompt_text = f"""Summarize the following text with respect to its title {{$title}}"""

        summary_prompt = Prompt([
            SystemMessage(summary_prompt_text),
            UserMessage('{{$input}}'),
            AssistantMessage(' ')
        ])
        summary_prompt_options = PromptCompletionOptions(completion_type='chat', model=self.model, max_tokens=200)

        text = 'doc: '+input+'\nSam: '+response
        analysis = ut.run_wave (self.client, {"input": text}, prompt, prompt_options,
                              self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                              logRepairs=False, validator=TOMLResponseValidator(action_validation_schema))
      
        print(f'SamCoT analysis {analysis}')
        if type(analysis) == dict and 'status' in analysis and analysis['status'] == 'success':
            content = analysis['message']['content']
            print(f'SamCoT content {content}')
            if type(content) == dict and 'action' in content.keys() and content['action']=='article':
                title = content['value']
                print(f'Sam wants to read {title}')
                article = self.search_titles(title, news_details)
                url = article['url']
                print(f' requesting url from server {title} {url}')
                response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={title}&url={url}')
                data = response.json()
                summary = ut.run_wave(self.client, {"input":data['result']}, summary_prompt, summary_prompt_options,
                              self.memory, self.functions, self.tokenizer, max_repair_attempts=1,
                              logRepairs=False, validator=DefaultResponseValidator())
                                      
                if type(summary) is dict and 'status' in summary.keys() and summary['status']=='successful':
                    return {"result":  summary['message']['content']}
                else:
                    return {"result": ''}

            if type(content) == dict and 'action' in content.keys() and content['action']=='web':
                return {"web":content['value']}
            
    
if __name__ == '__main__':
    sam = SamInnerVoice(model='alpaca')
    #print(sam.news)
    #print(sam.action_selection("Hi Sam. We're going to run an experiment ?",  'I would like to explore ', sam.details))
    #print(generate_faiss_id('a text string'))
    #sam.remember('language models','this is a sentence about large language models')
    #sam.remember('language models', 'this is not a sentence about large language models')
    print(sam.recall('doc','something about large language models'))
