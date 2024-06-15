from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BertRetriever:
   # def __init__(self, model_name='bert-base-uncased'):
    def __init__(self, model_name='bert-large-uncased'):

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    def retrieve(self, query, documents):
        query_embedding = self.embed_text(query).reshape(1, -1)  # reshape to 2D array
        doc_embeddings = np.array([self.embed_text(doc) for doc in documents])
        similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
        ranked_docs = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:3]]  # 返回最相似的前3个文档

# 示例文档集合
documents = [
    "BERT is a transformer model for natural language understanding.",
    "Python is a popular programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning is used in many AI applications.",
    "BERT improves the accuracy of search engines.",
    "BERT是由Google 研究人员于2018 年推出的,是一种使用Transformer 架构的强大语言模型.",
    "我喜欢吃苹果,because it is delicious."
]

# 初始化 BERT 召回模型
retriever = BertRetriever()

# 用户查询
query = "BERT is a deep learning model, What is BERT used for?"
#query = "BERT是什么?"
# 检索相关文档
retrieved_docs = retriever.retrieve(query, documents)
print("Retrieved documents:")
for doc in retrieved_docs:
    print(doc)
