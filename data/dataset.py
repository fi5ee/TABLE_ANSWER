# 日期:2024/5/26
from torch.utils.data import Dataset
import torch
from config.config import *
class CustomDataset(Dataset):

    def __init__(self, query_dic, table_dic,batch_size,tokenizer):
        super(CustomDataset,self).__init__()
        self.querydic=query_dic
        self.tabledic=table_dic
        self.batch_size = batch_size
        self.num_samples=len(query_dic)
        self.tokenizer=tokenizer
    def __len__(self):
        return self.num_samples // self.batch_size
    def __getitem__(self, index):#我们需要x(bs,max_seq_len),att(bs,max_seq_len),head_id(bs,h_len),sel_id(bs,h_len),new(bs,h_len,768)
        # 生成批量数据
        if index > len(self):
            raise IndexError("Index out of range")
        a = []
        b = []
        c = []
        d = []
        e = []
        q = []
        r=[]
        l_index=[]
        h_len_list=[]
        qu_len=[]
        max_len = 0
        h_len = 0
        cond_len = 0
        qu_id_len = 0
        r_len=0
        for i in range(self.batch_size):
            a1, b1, c1 = self.serialize(self.querydic[i+index*self.batch_size], self.tabledic, self.tokenizer)
            d1, e1, q1 = self.cond_data_process(self.querydic[i+index*self.batch_size]['sql']['conds'], self.querydic[i+index*self.batch_size]['question'])
            r1,l1 = self.duohao(self.querydic[i + index * self.batch_size], self.tabledic)
            r.append(r1)
            l_index.append(l1)
            a.append(a1)
            b.append(b1)
            c.append(c1)
            d.append(d1)
            e.append(e1)
            q.append(q1)
            qu_len.append(len(q1))
            h_len_list.append(len(b1))
            if max_len < len(a1):
                max_len = len(a1)
            if h_len < len(b1):
                h_len = len(b1)
            if cond_len < len(d1):
                cond_len = len(d1)
            if qu_id_len < len(e1):
                qu_id_len = len(e1)
            if r_len < len(r1):
                r_len = len(r1)
        x = torch.zeros(self.batch_size, max_len).type(torch.long)
        att_mask = torch.zeros(self.batch_size, max_len).type(torch.long)
        head_id = torch.zeros(self.batch_size, h_len).type(torch.long)
        qu_id = torch.zeros(self.batch_size, qu_id_len).type(torch.long)
        sel_id = torch.full((self.batch_size, h_len), 10).type(torch.long)
        cond_id = torch.full((self.batch_size, cond_len), 10).type(torch.long)
        qu = torch.zeros(self.batch_size, qu_id_len).type(torch.long)
        sel_cond_id = torch.full((self.batch_size, r_len), 10).type(torch.long)
        for i in range(self.batch_size):
            x[i, :len(a[i])] = a[i]
            att_mask[i, :len(a[i])] = torch.ones(len(a[i]))
            head_id[i, :len(b[i])] = b[i]  # 需要对表头做padding,(4,h_len)
            sel_id[i, :len(c[i])] = c[i]
            cond_id[i, :len(d[i])] = d[i]
            qu_id[i, :len(e[i])] = e[i]
            qu[i, :len(q[i])] = q[i]
            sel_cond_id[i, :len(r[i])] = r[i]
        new = torch.empty(self.batch_size, h_len, 768).type(torch.long)  # 该张量用于做torch.gather的输入
        new1 = torch.empty(self.batch_size, qu_id_len,768).type(torch.long)
        for i in range(self.batch_size):
            for j in range(h_len):
                new[i, j, :] = head_id[i, j].repeat(768)
            for k in range(qu_id_len):
                new1[i, k, :] = qu_id[i, k].repeat(768)
        return x, att_mask, head_id, sel_id, new, cond_id, new1, qu, sel_cond_id, l_index, h_len_list, qu_len
    @staticmethod
    def duohao(query_line, table):
        a, _, b = CustomDataset.cond_data_process(query_line['sql']['conds'], query_line['question'])
        p = 0
        a = torch.nonzero(a != 3).squeeze(-1)
        qu_index_set = []
        if a.numel() > 0:
            qu_index_set = [[a[0].item()]]  # cond_val 列索引
        for l in range(1, a.size(0)):  # 算法：利用 每一个条件值的多个索引具有连续性
            if a[l].item() == a[l - 1].item() + 1:
                qu_index_set[p].append(a[l].item())
            else:
                qu_index_set.append([a[l].item()])
                p += 1
        conds = query_line['sql']['conds']

        head = 4*len(table[query_line["table_id"]]['header'])*[10]
        if len(qu_index_set)<=4:
            head[:len(qu_index_set)*len(table[query_line["table_id"]]['header'])] = len(qu_index_set)*len(table[query_line["table_id"]]['header'])*[0]
        else:
            return torch.tensor(head), qu_index_set

        for i, item in enumerate(qu_index_set):  # 按条件值*h_len，在同一行堆叠(堆叠过程不含mask)
            jtem = [b[j] for j in item]
            q = tokenizer.decode(jtem, skip_special_tokens=True).replace(" ",'')  # 找到句子中与q相等token的index
            #print(q)  # 用操作符定位一样的条件值 列29286
            for cond in conds:  # 是否这里可以更丰富一点？改成
                if isinstance(cond[2],str):
                    cond[2] = tokenizer.decode(tokenizer.encode(cond[2],add_special_tokens= False),skip_special_tokens=True)  # 转换非英文字符
                if isinstance(cond[2], str) and cond[2].replace(' ', '').lower() == q:
                    head[i*len(table[query_line["table_id"]]['header']) + cond[0]] += 1
                    break
                if isinstance(cond[2], int) and str(cond[2]) == q:
                    head[i*len(table[query_line["table_id"]]['header']) + cond[0]] += 1
                    break
                if isinstance(cond[2], float) and str(cond[2]) == q:
                    head[i * len(table[query_line["table_id"]]['header']) + cond[0]] += 1
                    break
        return torch.tensor(head), qu_index_set
    @staticmethod
    def serialize(query_dic1, table_dic, tokenizer):
        # query_dic:字典格式，包含一行查询数据。table：字典格式{id：表}
        col_type_dic = {"text": "[unused11]", "real": "[unused12]"}
        a_serial = []
        header_ids = []#head_id是指[text][real]这些表头的位置
        sel_ids = []
        a_serial.extend(tokenizer.encode(query_dic1["question"]))
        header_ids.append(len(a_serial))
        sel = query_dic1['sql']['sel']

        for i, head in enumerate(table_dic[query_dic1["table_id"]]["header"]):
            # print(table_dic[a["table_id"]]["header"][i])
            a_serial.extend(tokenizer.encode(["[CLS]"], add_special_tokens=False))
            # BERT认为输出序列的i = 0位置的Token对应的词向量包含了整个句子的信息
            a_serial.extend(tokenizer.encode([col_type_dic["text"]], add_special_tokens=False) if
                            table_dic[query_dic1["table_id"]]["types"][i] == "text" else tokenizer.encode(
                [col_type_dic["real"]], add_special_tokens=False))
            # print(head,tokenizer.encode(head,add_special_tokens=False))
            a_serial.extend(tokenizer.encode(head, add_special_tokens=False))
            a_serial.extend(tokenizer.encode(["[SEP]"], add_special_tokens=False))
            header_ids.append(len(a_serial))
            segment_ids = [0] * len(a_serial)
            head1 = header_ids[:-1]
            # head_ids=torch.tensor([[j]*model.config.hidden_size for j in head1])
            if i == sel:
                ids = agg_ops[query_dic1["sql"]["agg"]]
                sel_ids.append(new_agg_ops[ids])
            else:
                sel_ids.append(0)
        return torch.tensor(a_serial), torch.tensor(head1), torch.tensor(sel_ids)
    @staticmethod
    def cond_data_process(conds, question):  # conds:[[col,operator,value]],[3,3,3,3,3,0,0,3]
        qu = tokenizer.encode(question, add_special_tokens=False)
        qu_= tokenizer.encode(question)#为了找到含特殊字符的句子长度
        cond_anotate = torch.full((len(qu),), 3).type(torch.long)
        qu_select=torch.arange(1,len(qu_)-1).type(torch.long)
        for cond in conds:
            a = cond[2]  # 如果只有一个词，返回含一个元素的列表
            a = str(a)
            a = tokenizer.encode(a, add_special_tokens=False)
            #print(tokenizer.decode(a[1],skip_special_tokens=True))
            #print(tokenizer.decode(a[2], skip_special_tokens=True))
            index = []
            for j in range(len(qu)):  # 字符串查找
                for i, item in enumerate(a):
                    if j + i < len(qu) and item == qu[j + i]:
                        #print(item)
                        index.append(j + i)
                    else:
                        index = []
                        break
                if len(index) == len(a):
                    cond_anotate[index] = cond[1]
        return cond_anotate,qu_select,torch.tensor(qu).type(torch.long)