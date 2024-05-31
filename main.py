# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
import pandas as pd
from model import att_classifier, wiki_decode
from data import CustomDataset
from config.config import *
import os

import pandasql as psql
from sentence_transformers import SentenceTransformer, util
"Welcome to xiaowu's web"
@st.cache_resource
def load_model():
    model = att_classifier(bert_path)
    sim_model = SentenceTransformer(sim_path)
    sim_model.to(device)
    ck = torch.load("checkpoint.pth")
    model.load_state_dict(ck['net'])
    model.to(device)
    return model,sim_model
if __name__ == '__main__':
    st.title("SQL Query Predictor")
    st.write("请上传csv类型的表格文件，并输入英文查询问题")
    question = None
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        header = df.columns.tolist()
        types = df.dtypes.apply(lambda x: 'text' if x == object else 'real').tolist()
        question = st.text_input("Enter your question:")
    if question:
        y = {"table_id": "1", "question": question,
             "sql": {"sel": None, "conds": [[None, None, None]], "agg": None}}
        query = [y]
        # 构建JSON格式的数据结构
        table_dic = {
            "header": header,
            "types": types,
            "id": "1"
        }
        dic_z = {"1": table_dic}
        item = CustomDataset(query, dic_z, 1, tokenizer)[0]
        # print(item)

        model, sim_model = load_model()
        model.eval()
        with torch.no_grad():
            x = item[0].to(device)
            att = item[1].to(device)
            sel_new = item[4].to(device)
            qu_new = item[6].to(device)
            qu = item[7].to(device)
            head_len = item[-2]
            qu_len = item[-1]
            p_head, p_qu, head_part, qu_part, p_con_head = model(x, att, qu_new, sel_new, head_len=head_len, qu_origin=qu)
            m = wiki_decode(p_head[0], p_qu[0], p_con_head[0], head_len[0], qu_len[0], qu[0])
            dic = {}
            dic['sel'] = m['sel']
            dic['conds'] = [[m['cons_head'][i], m['ops'][i], m['conds_val'][i]] for i in range(len(m['ops']))]  # dic即为最终结果
            dic['agg'] = m['agg']
            #print(dic)
            sql = ['SELECT']
            if dic['agg'] is None:
                sql.append('*')
            else:
                sel_col = df.keys().tolist()[dic["sel"]]
                sql.append(dic['agg']+f'("{sel_col}")')
            sql.append("FROM")
            sql.append('df')
            st.write("Predicted SQL Query:")
            print("-------------------")
            if len(dic['conds'])>0 and dic['conds'][0] is not None:
                print(dic["conds"])
                for item in dic['conds']:
                    sql.append("WHERE")
                    sql.append(f'"{df.keys().tolist()[item[0]]}"')
                    sql.append(cond_ops_dic[item[1]])
                    sentence = [item for item in df.iloc[:, item[0]].tolist()]
                    sentence1 = [str(item) for item in df.iloc[:, item[0]].tolist()]
                    if str(item[2]) in sentence:
                        if isinstance(item[2],int):
                            sql.append(f"{item[2]}")
                        else:
                            sql.append(f'"{item[2]}"')
                    else:
                        reference_embed = sim_model.encode(f'{item[2]}', convert_to_tensor=True)
                        embedding = sim_model.encode(sentence1, convert_to_tensor=True)
                        sim = util.pytorch_cos_sim(embedding, reference_embed).squeeze(-1)
                        _, con_val_index = torch.max(sim, -1)
                        con_val_index = con_val_index.item()
                        if isinstance(sentence[con_val_index], str):
                            sql.append(f'"{sentence[con_val_index]}"')
                        else:
                            sql.append(f'{item[2]}')
                    sql.append("and")

                sql.pop()
            st.write(" ".join(sql))
            result = psql.sqldf(" ".join(sql), globals())
            st.write(result)


