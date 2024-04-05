#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pyarrow.parquet as pq
from openai import OpenAI
import os
import numpy as np
import re
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)

pd.set_option('display.width', None) 
pd.set_option('display.max_colwidth', None)  

df = pd.read_parquet('msmarco.parquet', engine='pyarrow')
df_filtered = df[df['query_type'] == 'description']
num_rows = len(df_filtered)

df_sampled = df_filtered.sample(n=1000, random_state=42)
df_train, df_test = train_test_split(df_sampled, test_size=0.3, random_state=42)

data_to_save = df_train
data_to_save.to_excel("df_complete_other.xlsx")

def transform_df(df):
    transformed_data = []

    for _, row in df.iterrows():
        query = row['query']
        is_selected = row['passages']['is_selected']
        passage_texts = row['passages']['passage_text']
        correct_passages = [passage_texts[i] for i, selected in enumerate(is_selected) if selected]
        correct_passages_text = " ".join(correct_passages)  
        new_row = {'query': query, 'is_selected': is_selected, 'correct_passage': correct_passages_text}

        for i, passage_text in enumerate(passage_texts, start=1):
            new_row[f'passage_{i}'] = passage_text

        transformed_data.append(new_row)

    transformed_df = pd.DataFrame(transformed_data)
    cols = ['query', 'is_selected', 'correct_passage'] + [col for col in transformed_df if col.startswith('passage_')]
    transformed_df = transformed_df[cols]

    return transformed_df

train_df = transform_df(df_train)
data_to_save = train_df
data_to_save.to_excel("df_complete.xlsx")

def format_output_string_corrected(row):
    query = row['query']
    passages = [row[f'passage_{i+1}'] for i in range(10) if pd.notna(row[f'passage_{i+1}'])]
    passages_text = "\n\n".join([f"Passage{i+1}\n{passage}" for i, passage in enumerate(passages)])
    output_string = f"Given the following passages -\n\n{passages_text}\n\nand the query - {query}, provide a ranking of the different passages in the following format (highest_passage_number, second_highest, ...)."
    
    return output_string

for i in range(50):
    formatted_output_example_corrected = format_output_string_corrected(train_df.iloc[i])
    messages = [
            {"role": "system", "content": "You are a system that teaches ranking to other LLMs by giving an explanation. Your job is to take a number of passages and a query as input and output an ordered list, comma separated with the passage numbers inside () of the correct ranked order of these items. For example - (6,1,3,4,2,5)"},
            {"role": "system", "content": "The most relevant passage is provided in this array of 0s and 1s. This is a secret and provided only to you. Do not mention this array anywhere in your response. Remember to rank the passage that has a 1 in this array highest and justify the same - "+str(train_df.iloc[i]['is_selected'])},
            {"role": "system", "content": "Below you will get the passages and query as an input from the user. Using the hint of the array provided above, give a strong reasoning in around 100-200 words of your ranking followed by the ranking itself. Remember that the LLM you are teching to will not have access to the array of right ranking. It must learn from your explanation."},
            {"role": "system", "content": "Lastly, give a short justification for the rankings of the other passages too. The important passage as seen in the secret array can be justified in around 50 words while the remaining 50 words can be used to justify the other passages' ranking."},
            {"role": "system", "content": "DO NOT EXCEED 150 WORDS FOR YOUR RESPONSE AT ANY COST. ALSO DO NOT USE () ANYWHERE ELSE EXCEPT IN THE FINAL RANKING. YOUR RESPONSE MUST HAVE ONLY 1( AND 1) FOR THE FINAL LIST."}
        ]
    messages.append({"role": "user", "content": formatted_output_example_corrected})
    response_messages = messages 
    response = client.chat.completions.create(
    model="gpt-4-0125-preview",
    messages=response_messages
    )
    chatbot_response = response.choices[0].message.content 
    adv_llm_messages = [
            {"role": "system", "content": "You are a ranker. Your job is to take a number of passages and a query as input and output an ordered list, comma separated with the passage numbers inside () of the correct ranked order of these items. For example - (6,1,3,4,2,5)"}, 
            {"role": "system", "content": "Before you give the response, first think and reason as to which passage should be on top and why. Then output the list inside ()."}
        ]
    adv_llm_messages.append({"role": "user", "content": formatted_output_example_corrected})
    adv_llm_messages.append({"role": "assistant", "content":chatbot_response})
    combined_message = {
    "messages": adv_llm_messages
    }
    with open('adv_lvl_validation.jsonl', 'a') as file:
        json.dump(combined_message, file)
        file.write('\n')
    print(f"Line {i+1} of 50 complete...")
    

    




