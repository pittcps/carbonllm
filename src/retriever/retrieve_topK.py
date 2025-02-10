import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

embeddings_path = '../output/tfidf_embeddings.pkl'
with open(embeddings_path, 'rb') as f:
    saved_data = pickle.load(f)

embeddings = saved_data['embeddings']
ids = saved_data['id']
vectorizer = saved_data['vectorizer']

QA_file = '../input/test_relevant.csv'
QA_data = pd.read_csv(QA_file)
output_csv = '../output/test_relevant_top10.csv'

products_file = '../input/products_text.csv'
products_data = pd.read_csv(products_file)

column_names = ['Product name', 'Question', 'Top K IDs', 'Ground truth ID', 'Top K texts', 'Ground truth', 'Ground truth index among K list']
topk_dataset = pd.DataFrame(columns=column_names)

k = 10
matches = 0

for index, row in QA_data.iterrows():
    query = row['Question']
    gt_id = row['Product ID']

    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_k_indices = np.argsort(similarities)[::-1]

    top_k_ids = [ids[i] for i in top_k_indices[:k]]
    top_k_texts = []
    for i in top_k_ids:
        if not products_data.loc[products_data['Product ID'] == i].empty:
            top_k_texts.append(products_data.loc[products_data['Product ID'] == i, 'Text'].iloc[0])
        else:
            print("Top K text not found for ID:", i)

    gt_text = ""
    if not products_data.loc[products_data['Product ID'] == gt_id].empty:
        gt_text = products_data.loc[products_data['Product ID'] == gt_id, 'Text'].iloc[0]
    else:
        print("GT text not found for ID:", gt_id)

    product_name = ""
    if not products_data.loc[products_data['Product ID'] == gt_id].empty:
        product_name = products_data.loc[products_data['Product ID'] == gt_id, 'Product name'].iloc[0]
    else:
        print("Product name not found for ID:", gt_id)

    topk_dataset.at[index, 'Product name'] = product_name
    topk_dataset.at[index, 'Question'] = query
    topk_dataset.at[index, 'Top K IDs'] = top_k_ids
    topk_dataset.at[index, 'Ground truth ID'] = gt_id
    topk_dataset.at[index, 'Top K texts'] = top_k_texts

    if gt_text in top_k_texts:
        index_gt = [i for i, text in enumerate(top_k_texts) if text == gt_text]
        topk_dataset.at[index, 'Ground truth'] = gt_text
        topk_dataset.at[index, 'Ground truth index among K list'] = [i+1 for i in index_gt]
    else:
        topk_dataset.at[index, 'Ground truth'] = "N/A"
        topk_dataset.at[index, 'Ground truth index among K list'] = [-1]

    if gt_text in top_k_texts:
        matches += 1

yes_percentage = matches / len(QA_data) * 100
no_percentage = (len(QA_data) - matches) / len(QA_data) * 100

topk_dataset.to_csv(output_csv, index = False)
print('# of total samples', len(QA_data))
print(f'# of not finding a match in Top {k} and percentages', len(QA_data) - matches, f"; {no_percentage:.3f}%")
print(f'# of finding a match in Top {k} and percentages', matches,f"; {yes_percentage:.3f}%")
