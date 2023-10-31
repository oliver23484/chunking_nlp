from sentence_transformers import SentenceTransformer, util
import pandas as pd
import faiss
import numpy as np
import fasttext

df = pd.read_csv("dataset/false_answers.csv")
print(len(df))

embedder = SentenceTransformer('all-MiniLM-L6-v2')

corpus = df["Question"] + ". " + df["Answer"]

print(corpus)
nb = 10
# Create embeddings from titles and texts of articles
# It takes about 10 minutes for 192368 articles
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
print(corpus_embeddings)

index = faiss.IndexFlatL2(384)
print(index.is_trained)
index.add(corpus_embeddings)
print(index.ntotal)
faiss.write_index(index, 'vectorstore/sample.index')


query = "What is the capital of australia?"
query_embedding = embedder.encode(query, convert_to_tensor=True)
# query_embedding = np.array(query_embedding)
print(query_embedding)
index_out = faiss.read_index('vectorstore/sample.index')
print(index_out.ntotal)
k = 4                          # we want to see 4 nearest neighbors
D, I = index_out.search(query_embedding, k)
print(D)
print(I)
# D, I = index_out.search(query_embedding, top_k)
# # print(I)
# print(D)
# hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
# hits = hits[0]
# print(f"\nTop {top_k} most similar sentences in corpus:")
# for hit in hits:
#     hit_id = hit['corpus_id']
#     article_data = df.iloc[hit_id]
#     title = article_data["Answer"]
#     print("-", title, "(Score: {:.4f})".format(hit['score']))
