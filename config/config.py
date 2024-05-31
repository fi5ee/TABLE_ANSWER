# 日期:2024/5/27
from transformers import BertTokenizer
import torch
bert_path = "bert-base-uncased"
tokenizer =  BertTokenizer.from_pretrained(bert_path)
device = torch.device("cuda")
col_type_dic = {"text": "[unused11]", "real": "[unused12]"}
agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
new_agg_ops = {'': 6, 'AVG': 1, "MAX": 2, "MIN": 3, "COUNT": 4, "SUM": 5}
cond_ops = ['=', '>', '<', 'OP']
cond_ops_dic = {0: '=', 1: '>', 2: '<', 3: 'not_anotate'}
sim_path = 'Semantic-Textual-Relatedness-Telugu'