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

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)

pd.set_option('display.width', None) 
pd.set_option('display.max_colwidth', None)  
pd.DataFrame(columns=['BM25', 'GPT3.5', 'GPT3.5 Finetuned', 'Embeddings']).to_csv('lists_saved.csv', mode='w', index=False)
df = pd.read_parquet('msmarco.parquet', engine='pyarrow')
df_filtered = df[df['query_type'] == 'description']
num_rows = len(df_filtered)

df_sampled = df_filtered.sample(n=1000, random_state=42)
df_train, df_test = train_test_split(df_sampled, test_size=0.3, random_state=42)

data_to_save = df_train
data_to_save.to_excel("df_complete_other.xlsx")

queries_train = []
passage_texts_train = []
is_selected_labels_train = []

for index, row in df_train.iterrows():
    query = row['query']
    passages_info = row['passages']
    passage_texts_row = passages_info['passage_text']
    is_selected_row = passages_info['is_selected']
    for passage_text, is_selected in zip(passage_texts_row, is_selected_row):
        queries_train.append(query)
        passage_texts_train.append(passage_text)
        is_selected_labels_train.append(is_selected)

df_ranking_train = pd.DataFrame({
    'query': queries_train,
    'passage_text': passage_texts_train,
    'is_selected': is_selected_labels_train
})

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
test_df = transform_df(df_test)
data_to_save = train_df
data_to_save.to_excel("df_complete.xlsx")

def format_output_string_corrected(row):
    query = row['query']
    passages = [row[f'passage_{i+1}'] for i in range(10) if pd.notna(row[f'passage_{i+1}'])]
    passages_text = "\n\n".join([f"Passage{i+1}\n{passage}" for i, passage in enumerate(passages)])
    output_string = f"Given the following passages -\n\n{passages_text}\n\nand the query - {query}, provide a ranking of the different passages in the following format (highest_passage_number, second_highest, ...)."
    
    return output_string
total_mrr_basic = 0
total_mrr_advanced = 0
total_mrr_bm = 0
total_mrr_em = 0
bm_arr = []
g35_arr = []
g35_ft_arr = []
em_arr=[]
def rank_passages_with_bm25(passages, query):
    tokenized_passages = [passage.split(" ") for passage in passages]
    tokenized_query = query.split(" ")
    bm25 = BM25Okapi(tokenized_passages)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]
    return (ranked_indices + 1).tolist()

def rank_passages_with_embeddings(passages, query):
    passage_embeddings = []
    for passage in passages:
        passage_embedding = client.embeddings.create(input = passage,model='text-embedding-3-small').data[0].embedding
        passage_embeddings.append(passage_embedding)
    query_embedding = client.embeddings.create(input = query,model='text-embedding-3-small').data[0].embedding
    similarities = cosine_similarity([query_embedding], passage_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    embeddings_ranking = sorted_indices + 1
    return embeddings_ranking.tolist()
num = 50
for i in range(num):
    formatted_output_example_corrected = format_output_string_corrected(test_df.iloc[i])
    passages_info_bm = df_test.iloc[i]['passages']  
    passages_bm = passages_info_bm['passage_text'].tolist()
    query_bm = df_test.iloc[i]['query']
    bm_ranking = rank_passages_with_bm25(passages_bm, query_bm)
    embeddings_ranking = rank_passages_with_embeddings(passages_bm, query_bm) 
    print(bm_ranking)
    print(embeddings_ranking)
    basic_llm_messages = [
            {"role": "system", "content": "You are a ranker. Your job is to take a number of passages and a query as input and output an ordered list, comma separated with the passage numbers inside () of the correct ranked order of these items. For example - (6,1,3,4,2,5)"}, 
            {"role": "system", "content": "ALWAYS ENCLOSE THE FINAL LIST INSIDE () AND INSIDE THERE SHOULD ONLY BE DIGITS AND , IN THE RIGHT ORDER."}
        ]
    basic_llm_messages.append({"role": "user", "content": formatted_output_example_corrected})
    response_messages = basic_llm_messages 
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=response_messages,
    temperature = 0.2
    )
    chatbot_response_basic = response.choices[0].message.content 
    print(chatbot_response_basic)
    advanced_llm_messages = [
            {"role": "system", "content": "You are a ranker. Your job is to take a number of passages and a query as input and output an ordered list, comma separated with the passage numbers inside () of the correct ranked order of these items. For example - (6,1,3,4,2,5)"}, 
            {"role": "system", "content": "Before you give the response, first think and reason as to which passage should be on top and why. Then output the list inside ()."}, 
            {"role": "system", "content": "ALWAYS ENCLOSE THE FINAL LIST INSIDE ()."}, 
        ]
    advanced_llm_messages.append({"role": "user", "content": formatted_output_example_corrected})
    response_messages = advanced_llm_messages 
    response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:tu-delft:ir-50:98wqt5iU",
    messages=response_messages,
    temperature = 0.2
    )
    chatbot_response_advanced = response.choices[0].message.content 
    ranking_messages = [
            {"role": "system", "content": "You are a ranking extractor. You will be given a block of text generated by an LLM which has a lot of fluff and you will return only one output - the ranking as seen in the text inside () with all digits comma separated."}, 
            {"role": "system", "content": "YOUR RESPONSE AND THE FINAL LIST MUST BE INSIDE () AND THERE SHOULD ONLY BE DIGITS AND , IN THE RIGHT ORDER. Like example (6,1,2,5,4,3). NOTHING ELSE SHOULD BE THERE."}
        ]
    ranking_messages.append({"role": "user", "content": chatbot_response_advanced})
    response_messages = ranking_messages 
    response_ranking = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=response_messages,
    temperature = 0.2
    )
    chatbot_response_advanced = response_ranking.choices[0].message.content
    print(chatbot_response_advanced)
    def parse_rankings(ranking_str):
        ranking_str = ranking_str.strip("()")  
        return [int(x) for x in ranking_str.split(',')]

    def compute_mrr(is_selected, predicted_ranking):
        predicted_ranking_1_indexed = [x - 1 for x in predicted_ranking]
        reciprocal_rank = 0
        for idx in predicted_ranking_1_indexed:
            if is_selected[idx] == 1:
                reciprocal_rank = 1 / (predicted_ranking_1_indexed.index(idx) + 1)
                break
        return reciprocal_rank

    predicted_rankings_basic = parse_rankings(chatbot_response_basic)
    print(predicted_rankings_basic)
    mrr_score_basic = compute_mrr(test_df.iloc[i]['is_selected'], predicted_rankings_basic)
    print(f"MRR Score basic: {mrr_score_basic}")
    g35_arr.append(mrr_score_basic)
    total_mrr_basic+=mrr_score_basic
    predicted_rankings_advanced = parse_rankings(chatbot_response_advanced)
    mrr_score_advanced = compute_mrr(test_df.iloc[i]['is_selected'], predicted_rankings_advanced)
    print(f"MRR Score fine-tuned reasoning: {mrr_score_advanced}")
    g35_ft_arr.append(mrr_score_advanced)
    total_mrr_advanced+=mrr_score_advanced
    predicted_rankings_bm = bm_ranking
    mrr_score_bm = compute_mrr(test_df.iloc[i]['is_selected'], predicted_rankings_bm)
    print(f"MRR Score BM25: {mrr_score_bm}")
    bm_arr.append(mrr_score_bm)
    total_mrr_bm+=mrr_score_bm
    predicted_rankings_em = embeddings_ranking
    mrr_score_em = compute_mrr(test_df.iloc[i]['is_selected'], predicted_rankings_em)
    print(f"MRR Score embeddings: {mrr_score_em}")
    em_arr.append(mrr_score_em)
    total_mrr_em+=mrr_score_em
    new_row = {'BM25': mrr_score_bm, 'GPT3.5': mrr_score_basic, 'GPT3.5 Finetuned': mrr_score_advanced, 'Embeddings': mrr_score_em}
    df_new_row = pd.DataFrame([new_row])
    df_new_row.to_csv('lists_saved.csv', mode='a', header=False, index=False)
    print(f"Iteration {i+1} done and saved.")
    


print(f"Average MRR score with 3.5 - {total_mrr_basic/num}")
print(f"Average MRR score with 3.5 and reasoning - {total_mrr_advanced/num}")
print(f"Average MRR score with BM25 - {total_mrr_bm/num}")
print(f"Average MRR score with embeddings - {total_mrr_em/num}")
print(bm_arr)
print(g35_arr)
print(g35_ft_arr)
print(em_arr)

df_ranking_mrr = pd.DataFrame([bm_arr, g35_arr, g35_ft_arr, em_arr])
df_ranking_mrr.index = ['BM25', 'GPT3.5', 'GPT3.5 Finetuned', 'Embeddings']
df_ranking_mrr.to_csv('lists_saved.csv', index=True)




