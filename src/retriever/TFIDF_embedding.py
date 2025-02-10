import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

file_path = '../input/products_text.csv'
data = pd.read_csv(file_path)

vectorizer = TfidfVectorizer(stop_words='english')
embeddings = vectorizer.fit_transform(data['Text'].tolist())

embeddings_path = '../output/tfidf_embeddings.pkl'
with open(embeddings_path, 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'id': list(range(1, len(data) + 1)), 'vectorizer': vectorizer}, f)
