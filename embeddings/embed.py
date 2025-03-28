# embed.py
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer

model_name = 'sentence-transformers/all-mpnet-base-v2'
embedder = SentenceTransformer(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# pt_model = AutoModel.from_pretrained(model_name)

sentence = "He plays soccer"
words = list(sentence.split())
embedding = embedder.encode(words + [sentence])

print(embedding.shape)