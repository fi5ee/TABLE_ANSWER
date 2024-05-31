# 日期:2024/5/31

from sentence_transformers import SentenceTransformer, util
import torch
bert_path = "../bert-base-uncased"
sim_path = '../Semantic-Textual-Relatedness-Telugu'
sentences = ["1", "2"]

model = SentenceTransformer(sim_path)
embedding_1= model.encode(sentences, convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

sim = util.pytorch_cos_sim(embedding_1, embedding_2).squeeze(-1)
a,b=torch.max(sim, dim=-1)
print(b.item())
"""
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Nationality': ['ukraine', 'american', 'ukraine']
}
sql_query = "SELECT * FROM df WHERE Nationality = 'ukraine'"
df = pd.DataFrame(data)
result = psql.sqldf(sql_query, globals())
print(result)
"""