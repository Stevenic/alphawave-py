import os
import gc
import pandas as pd
import numpy as np
import re
import math
import time
import traceback
from tqdm.auto import tqdm
import blingfire as bf

from collections.abc import Iterable
import torch
import faiss
from faiss import write_index, read_index
from datasets import load_dataset
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Union
import datasets

from sklearn.preprocessing import normalize

import ctypes
libc = ctypes.CDLL("libc.so.6")

import warnings
warnings.filterwarnings("ignore")

def process_documents(documents: Iterable[str],
                      document_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param document_type: String denoting the document type to be processed
    :param document_sections: List of sections for a given document type to process
    :param split_sentences: Flag to determine whether to further split sections into sentences
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """
    
    print(documents)
    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values, 
                        df.document_id.values,
                        df.offset.values, 
                        filter_len, 
                        disable_progress_bar)
    return df


def sectionize_documents(documents: Iterable[str],
                         document_ids: Iterable,
                         disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the 
    selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for document_id, document in zip(document_ids, documents):
        row = {}
        text, start, end = (document, 0, len(document))
        row['document_id'] = document_id
        row['text'] = text
        row['offset'] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(['document_id', 'offset']).reset_index(drop=True)
    else:
        return _df


def sentencize(documents: Iterable[str],
               document_ids: Iterable,
               offsets: Iterable[tuple[int, int]],
               filter_len: int = 3,
               disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param offsets: Iterable tuple of the start and end indices
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for document, document_id, offset in zip(documents, document_ids, offsets):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1]-o[0] > filter_len:
                    sentence = document[o[0]:o[1]]
                    abs_offsets = (o[0]+offset[0], o[1]+offset[0])
                    row = {}
                    row['document_id'] = document_id
                    row['text'] = sentence
                    row['offset'] = abs_offsets
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)

class OpenBook():
    def __init__(self,top_k_articles:int=3,
                top_k_matches:int=3,
                model_name:str="BAAI/bge-small-en", 
                device:str='cuda',
                max_length:int=512, 
                batch_size:int=16,
                wikipedia_data_path:str="/home/bruce/Downloads/kaggle/openbook/en_wiki_202212",
                wikipedia_index_path:str="/home/bruce/Downloads/kaggle/openbook/en_wiki_202212/index.parquet",
                wikipedia_faiss_index_path:str="/home/bruce/Downloads/kaggle/openbook/en_wiki_202212_ivf256_sq8.index",
                nprobe:int=8):
        self.top_k_articles=top_k_articles
        self.top_k_matches=top_k_matches
        self.model_name=model_name
        self.device=device
        self.max_length=max_length
        self.batch_size=batch_size
        self.wikipedia_data_path=wikipedia_data_path
        self.wikipedia_index_path=wikipedia_index_path
        self.wikipedia_faiss_index_path=wikipedia_faiss_index_path
        self.nprobe=nprobe

        ## Setup the model and tokenizer
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        ## Setup the wikipedia articles index
        self.wikipedia_faiss_index = read_index(self.wikipedia_faiss_index_path)
        self.wikipedia_faiss_index_ivf = faiss.extract_index_ivf(self.wikipedia_faiss_index)
        self.wikipedia_faiss_index_ivf.nprobe = nprobe
        ## Load the wikipedia article index file
        self.wikipedia_article_index = pd.read_parquet(self.wikipedia_index_path)

    def get_embeddings(self, text:Iterable[str]):    
        sentence_embeddings = []
        n_batches = len(text) // self.batch_size
    
        # Compute token embeddings
        with torch.no_grad():
            for i in range(n_batches+1):
                if i*self.batch_size == len(text):
                    break # empty batch when len(text) is exact multiple, too lazy to fix properly
                encoded_input = self.tokenizer(
                    text[i*self.batch_size:(i+1)*self.batch_size], 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length, 
                    return_tensors='pt'
                ).to(self.device)

                model_output = self.model(**encoded_input)
            
                # Perform pooling. In this case, cls pooling.
                _sentence_embeddings = model_output[0][:, 0, :]
            
                # normalize embeddings
                _sentence_embeddings = torch.nn.functional.normalize(_sentence_embeddings, p=2, dim=1)
                sentence_embeddings.append(_sentence_embeddings.detach().cpu().numpy())
        sentence_embeddings = np.concatenate(sentence_embeddings)
        return sentence_embeddings    

    def search1(self, text):
        ## Get question choice embeddings - will be used later
        self.question_choice_embeddings = []
        embeddings = self.get_embeddings(text)
        self.question_choice_embeddings.append(embeddings)
        ## Get the indices corresponding to the wikipedia article that best matches
        self.question_article_indices = []
        for question_choice_embedding in self.question_choice_embeddings:
            _, search_index = self.wikipedia_faiss_index.search(question_choice_embedding, self.top_k_articles)
            self.question_article_indices.append(list(set(search_index.flatten())))

        
    def search2(self, text):
        ## Identifying which files to perform look up and which question questions are associated with which articles
        self.question_article_data = []
        for article_index in self.question_article_indices:
            ## Within the Wikipedia Index get the articles (and associated file values) that are closest to the choices for each question
            _df = self.wikipedia_article_index.loc[article_index].copy()
        self.question_article_data.append(_df)
        self.question_article_data = pd.concat(self.question_article_data).reset_index(drop=True)
    
        ## Create the data to tell us which files to look up
        self.wikipedia_article_data = self.question_article_data[['id','file_id']].drop_duplicates().sort_values(['file_id', 'id']).reset_index(drop=True)
    
        ## Obtaining the article text data 
        self.wikipedia_article_text = []
        for file_id in self.wikipedia_article_data.file_id.unique():
            ## For the file, get all the ids pertinent that exist in that file
            _id = [i for i in self.wikipedia_article_data[self.wikipedia_article_data['file_id']==file_id]['id'].tolist()]
            _df = pd.read_parquet(f"{self.wikipedia_data_path}/{file_id}.parquet")
            _df = _df[_df['id'].isin(_id)]
            self.wikipedia_article_text.append(_df)

        self.wikipedia_article_text = pd.concat(self.wikipedia_article_text).drop_duplicates().reset_index(drop=True)
        self.wikipedia_article_text['document_id'] = self.wikipedia_article_text.apply(lambda x: f"{x['id']}_{x['paragraph_id']}", axis=1)

    def search3(self):
        ## Parse documents into sentences
        self.wikipedia_sentence_text = process_documents(self.wikipedia_article_text.text.values, self.wikipedia_article_text.document_id.values)

        ## Have to split document_id back into wiki article ids and paragraph ids
        self.wikipedia_sentence_text['id'] = self.wikipedia_sentence_text['document_id'].apply(lambda x: np.int64(x.split('_')[0]))
        self.wikipedia_sentence_text['paragraph_id'] = self.wikipedia_sentence_text['document_id'].apply(lambda x: np.int64(x.split('_')[1]))
        ## Get embeddings of the wiki text data
        self.wikipedia_sentence_embeddings = self.get_embeddings(self.wikipedia_sentence_text.text.tolist())
        #else:
        #    self.wikipedia_sentence_embeddings = self.get_embeddings(('empty string just to create the right output object'))
                    
    def search(self, text):
        self.search1(text)
        self.search2(text)
        self.search3()
        ## Now creating Wikipedia lookups for each question in the dataframe
        question_contexts = []
        id=text
        question_id=id
        sentence_indices = self.wikipedia_sentence_text.index.values
        ## Only if there was associated text
        if sentence_indices.shape[0] > 0:
            ## Per question Index
            try:
                ## Perform semantic search over every item
                question_index = faiss.index_factory(self.wikipedia_sentence_embeddings.shape[-1], "Flat")
                question_index.train(self.wikipedia_sentence_embeddings)
                question_index.add(self.wikipedia_sentence_embeddings)
                ## Get the best matches from the articles that most closely matched the question
                # 0 in below because there is only 1 sample, and qce shape is [num_samples, num_choice_vectors/sample, 384]
                _question_match_scores, _question_match_indices = question_index.search(np.array(self.question_choice_embeddings[0]), self.top_k_matches)
                for _question_match_score, _question_match_index in zip(_question_match_scores, _question_match_indices):
                    ## Get the pertinent sentencized chunks
                    _text_data = self.wikipedia_sentence_text.loc[sentence_indices].iloc[_question_match_index]

                    _ids = _text_data['id'].values
                    _paragraph_ids = _text_data['paragraph_id'].values
                    _sentences = _text_data['text'].values

                    _paragraphs = []
                    for _id, _paragraph_id, _sentence in zip(_ids, _paragraph_ids, _sentences):
                        _paragraph_source = self.wikipedia_article_text[(self.wikipedia_article_text['id']==_id)&(self.wikipedia_article_text['paragraph_id']==_paragraph_id)]['text'].values

                        ## Check if there is a paragraph
                        if _paragraph_source.shape[0] > 0:
                            _paragraphs.append(_paragraph_source[0])
                        else:
                            _paragraphs.append("")

                    for _score, _id, _paragraph_id, _sentence, _paragraph in zip(_question_match_score, _ids, _paragraph_ids, _sentences, _paragraphs):
                        _question_context = {}
                        _question_context['question_id'] = question_id
                        _question_context['wiki_id'] = _id
                        _question_context['paragraph_id'] = _paragraph_id
                        _question_context['score'] = _score
                        _question_context['sentence'] = _sentence
                        _question_context['paragraph'] = _paragraph
                        question_contexts.append(_question_context)
            except ValueError as e:
                print(e)
                traceback.print_exc()

        torch.cuda.empty_cache()
    
        question_contexts_df = pd.DataFrame(question_contexts)
        question_contexts_df = question_contexts_df.sort_values(['question_id', 'wiki_id', 'paragraph_id', 'score']).reset_index(drop=True)
        question_contexts_df = question_contexts_df.drop_duplicates(subset=['question_id', 'wiki_id', 'paragraph_id'], keep='first').reset_index(drop=True)
        return question_contexts_df

    
op = OpenBook(device='cpu')
gc.collect()
import time
start = time.time()
print(op.search('Einstein Rosen Podolsky Paradox'))
print(time.time()-start)
