# 日期:2024/5/26
import torch

from config.config import *
from transformers import BertModel
import torch.nn as nn


class att_classifier(nn.Module):

    def __init__(self, model_name):
        super(att_classifier, self).__init__()
        device = torch.device("cuda")
        self.head = None
        self.qu = None
        self.bert = BertModel.from_pretrained(model_name)
        self.sel_classifier = nn.Linear(self.bert.config.hidden_size, 7)
        self.con_classifier = nn.Linear(self.bert.config.hidden_size, 4)
        self.active = nn.LogSoftmax(dim = -1)
        self.multiattention = nn.MultiheadAttention(768,12,batch_first=True,device=device)
        self.two_classifier = nn.Linear(768, 2)
        self.fnn1 = nn.Linear(768, 2048)
        self.fnn2 = nn.Linear(2048,768)
        self.layernorm = nn.LayerNorm(768)
        self.gelu = nn.GELU()
        self.soft = nn.Softmax

    def forward(self, x, attention_mask, qu_sel, head_sel,l_index=None,head_len=None,qu_origin=None):

        cls_hidden_states = self.bert(x, attention_mask=attention_mask).last_hidden_state
        head_part = torch.gather(cls_hidden_states, 1, head_sel)
        qu_part = torch.gather(cls_hidden_states, 1, qu_sel)  # 不包含cls
        p_head = self.active(self.sel_classifier(head_part))
        p_qu = self.active(self.con_classifier(qu_part))
        #self.head = head_part.detach()
        #self.qu = qu_part.detach()
        self.head = head_part
        self.qu = qu_part
        self.p_qu = p_qu
        if l_index is not None:
            p_con_head = self.train_state(l_index=l_index, head_len=head_len)
        else:
            index = self.inference_state(qu_origin)
            p_con_head = self.train_state(l_index=index, head_len=head_len)
        return p_head, p_qu, head_part, qu_part, p_con_head
    def train_state(self, l_index, head_len):  # 4*[[cond1_index],[cond2_index],[cond3_index]] ,[len(h1),len(h2),...]
        device = torch.device("cuda")
        size = 0
        val_len = 0
        val_num = 0
        label=0
        for i in l_index:#判断batch中的条件值是否全部为空
            if len(i)!=0:
                label=1
                break
        if label == 0:
            return [1]
        for j, jtem in enumerate(l_index):
            size = max(size, len(jtem)*head_len[j])
            val_num = max(val_num, len(jtem))
            for i, item in enumerate(jtem):
                val_len = max(val_len, len(item))
        key = torch.zeros(4*len(l_index), val_len, 768).to(device)#默认cond个数小于等于4，已经统计过训练集
        h = torch.zeros(4*len(l_index), self.head.size(1), 768).to(device)
        for i in range(len(l_index)):#将一句话的4个条件值拆分为4个batch
            for j in range(4):
                h[i*len(l_index)+j] = self.head[i]
        #h = torch.cat([self.head]*4, dim=-1).to(device)
        mask = torch.ones(4*len(l_index), val_len).bool()
        mask = mask.to(device)
        mask[:,0]=False
        for j in range(len(l_index)):
            if len(l_index[j])>4:
                continue
            for i, item in enumerate(l_index[j]):  # [[cond1_index],[cond2_index],[cond3_index]]
                if len(item)==0:
                    break
                q_tensor = self.qu[j, item]
                try :
                    key[j*len(l_index)+i, :q_tensor.size(0)] = q_tensor
                except Exception as e:
                    print(l_index,val_len,self.qu.size(),q_tensor.size(),key.size())
                mask[j*len(l_index)+i, :len(item)] = False
        att, _ = self.multiattention(h, key, key, key_padding_mask=mask)


        p_con_head = self.layernorm((h+self.fnn2(self.fnn1(self.layernorm(h+att)))))
        logit = self.two_classifier(p_con_head)
        p_con_head1 = self.active(logit)

        ans = torch.zeros(len(l_index), max(head_len)*4, 2).to(device)
        for i in range(len(l_index)):
            h_l = head_len[i]
            for j in range(len(l_index[i])):
                if j>=4:
                    break
                try:
                    ans[i, j*h_l:j*h_l+h_l] = p_con_head1[i*len(l_index)+j, :h_l]
                except Exception as e:
                    print(l_index,len(l_index[i]))
                    print(head_len, ans.size(), p_con_head1.size())
                    raise ValueError("error")
        return ans
    def inference_state(self,qu_origin):  # 返回问句中被标记的条件值token索引  4*[[cond1_index],[cond2_index],[cond3_index]]
        _, indice2 = torch.max(self.p_qu, dim=-1)
        l_index = []
        for j in range(self.qu.size(0)):
            anti_pad_size1 = (qu_origin[j, :] != 0).sum().item()  # 求出问句中除去pad=0的数组长度
            one_qu_temp = indice2[j, :anti_pad_size1]  # 单一问句的预测标签值
            qu_index_set = []

            one_qu = torch.nonzero(one_qu_temp != 3).squeeze(-1)  # 找到被词标记为条件值（!=3）的索引
            if one_qu.numel() > 0:
                qu_index_set = [[one_qu[0].item()]]  # [[cond1_index],[cond2_index],[cond3_index]]
                # print(one_qu_temp,one_qu,qu_index_set)

            p = 0
            for l in range(1, one_qu.size(0)):  # 算法：利用 每一个条件值的多个索引具有连续性
                if one_qu[l].item() == one_qu[l - 1].item() + 1:
                    qu_index_set[p].append(one_qu[l].item())

                else:
                    qu_index_set.append([one_qu[l].item()])
                    p += 1
            l_index.append(qu_index_set)
        return l_index



def wiki_decode(p_head,p_qu,p_con,head_mask,qu_mask,qu):  # 所有数据都是一维的  #item[-2],item[-1],item[-5]
    def header_decode(indice):
        new_agg_ops = ['pass','AVG',"MAX","MIN","COUNT","SUM",""]
        for i, item in enumerate(indice):
            if item != 0:
                return i, new_agg_ops[item]
        return None, None
    def qu_decode(indice,qu):
        op=[]
        qu_index_set = []
        ## squeeze:移除掉指定维度大小为一的维度（如果指定维度大小不为一，则返回原张量）
        one_qu = torch.nonzero(indice != 3).squeeze(-1)  # 找到被词标记为条件值（!=3）的索引
        if one_qu.numel() > 0:
            qu_index_set = [[one_qu[0].item()]]  # [[cond1_index],[cond2_index],[cond3_index]]
            # print(one_qu_temp,one_qu,qu_index_set)
        p = 0
        for l in range(1, one_qu.size(0)):  # 算法：利用 每一个条件值的多个索引具有连续性
            if one_qu[l].item() == one_qu[l - 1].item() + 1:
                qu_index_set[p].append(one_qu[l].item())
            else:
                qu_index_set.append([one_qu[l].item()])
                p += 1
        val=[]
        for i,item in enumerate(qu_index_set):  # [[cond1_index],[cond2_index],[cond3_index]]
            w=[]
            value=''
            for j in range(len(item)):
                try:
                    if tokenizer.decode(qu[item[j]], skip_special_tokens=False)[0] == "#":
                        if len(w) > 0:
                            w.pop()
                        w.append(
                            tokenizer.decode(qu[item[j]], skip_special_tokens=False).replace(" ", "").replace(
                                "#", ""))
                        w.append(" ")
                    else:
                        w.append(tokenizer.decode(qu[item[j]], skip_special_tokens=False).replace(" ", ""))
                        w.append(" ")
                except Exception as e:
                    print(qu_index_set)
                    print(item,qu)
            for k in w:
                value += k
            val.append(value.strip())
        for item in qu_index_set:
            op.append(indice[item[0]].item())
        return val, op, len(op)
    def cons_decode(indice,n,h_len):
        lst=[]
        for i in range(n):
            con_heads=indice[i*h_len:i*h_len+h_len]
            index = torch.nonzero(con_heads!=0).squeeze(-1)  ##也许可以加一个全等于0的惩罚项
            if index.size(0):
                lst.append(index[0].item())
            else:
                lst.append(None)
        return lst
    _, indice1 = torch.max(p_head, dim=-1)
    _, indice2 = torch.max(p_qu, dim=-1)
    if isinstance(p_con,torch.Tensor):
        _, indice3 = torch.max(p_con, dim=-1)
    else:
        indice3 = ''
    indice1 = indice1[:head_mask]
    indice2 = indice2[:qu_mask]
    header, agg = header_decode(indice1)
    qu_val, op, num = qu_decode(indice2, qu)
    cons_head = cons_decode(indice3, num, h_len=head_mask)  ## [head1,head2,head3] 与条件值对应
    return {"sel":header,'agg':agg,'conds_val':qu_val,"ops":op,"cons_head":cons_head}