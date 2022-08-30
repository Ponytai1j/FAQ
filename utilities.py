
import pandas as pd
import re
import jieba
import pkuseg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from torch.utils.data import DataLoader, Dataset 
import torch

def filter_out_category(input):
    new_input = re.sub('[\u4e00-\u9fa5]{2,5}\\/','',input)
    return new_input

def filter_out_punctuation(input):
    new_input = re.sub('([a-zA-Z0-9])','',input)
    new_input = ''.join(e for e in new_input if e.isalnum())
    return new_input

def word_segmentation(input, seg):
    # 加载停用词
    with open('./hit_stopwords.txt', encoding = 'utf-8') as f:
        # f.readlines()返回一个list，每个element是文件中的一行会有换行符['!\n'，'*\n'...]
        stopwords = [line.strip('\n') for line in f.readlines()] 
    if seg == 'jieba':
        #temp = jieba.cut(input)
        new_input = ' '.join(jieba.cut(input, cut_all=True))
        line_list = new_input.split(' ')
        temp_line = []
        for i in line_list:
            if i not in stopwords:
                temp_line.append(i)
        temp_line = ' '.join(temp_line)
        line = temp_line
        new_input = line
        
    elif seg == 'pkuseg':
        file = open("temp.txt", 'w').close() # 清空
        file = open("temp_list.txt", 'w').close() # 清空
        textfile = open('temp.txt','w',encoding='utf-8') # array读入txt
        for element in input:
            textfile.write(element + '\n')
        textfile.close()
        pkuseg.test('temp.txt', 'temp_list.txt', nthread=20) # 进行分词
        new_input = []
        f = open('temp_list.txt',encoding='utf-8')
        while True:
            line = f.readline()
            if line:
                line = line.strip('\n')
                line_list = line.split(' ')
                temp_line = []
                for i in line_list:
                    if i not in stopwords:
                        temp_line.append(i)
                temp_line = ' '.join(temp_line)
                line = temp_line
                new_input.append(line)
            else:
                break
        f.close()
#         seg = pkuseg.pkuseg()  # 可以用细分领域 pkuseg.pkuseg(model_name='medicine') 
#         new_input = ','.join(seg.cut(input))
    return new_input

def preprocess_text(data, seg):
    if seg == 'pkuseg':
        new_data = []
        for q in data:
            q = filter_out_category(q)
            q = filter_out_punctuation(q)
            new_data.append(q)
        new_data = word_segmentation(new_data, seg)
            
    elif seg == 'jieba':
        new_data = []
        for q in data:
            q = filter_out_category(q)
            q = filter_out_punctuation(q)
            q = word_segmentation(q, seg)
            new_data.append(q)
  
    return new_data


def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def conver2tfidf(data):
    new_data = []
    for q in data:
        new_data.append(q)
    tfidf_vectorizer, tfidf_X = tfidf_extractor(new_data)
    return tfidf_vectorizer, tfidf_X

class LCQMC_Dataset(Dataset):
    def __init__(self, tokenized_q1s, tokenized_q2s, with_label, data=None):
        self.tokenized_q1s = tokenized_q1s
        self.tokenized_q2s = tokenized_q2s
        self.data = data
        self.max_q1_len = 35
        self.max_q2_len = 35
        self.with_label = with_label
        # [CLS] + q1 + [SEP] + q2 + [SEP]
        self.max_seq_len = 1 + self.max_q1_len + 1 + self.max_q2_len + 1
        self.with_label = with_label
    
    def __len__(self):
        #return len(self.data)
        return len(self.tokenized_q1s.input_ids)

    def __getitem__(self, idx):
        if self.with_label:
            this_data = self.data[idx]
        tokenized_q1 = self.tokenized_q1s[idx]
        tokenized_q2 = self.tokenized_q2s[idx]
        
        input_ids_q1 = [101] + tokenized_q1.ids[:self.max_q1_len] + [102]
        input_ids_q2 = tokenized_q2.ids[:self.max_q2_len] + [102]
        
        # Pad sequence and obtain inputs to model 
        input_ids, token_type_ids, attention_mask = self.padding(input_ids_q1, input_ids_q2)
        if self.with_label:
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), this_data['label']
        else:
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)

    def padding(self, input_ids_q1, input_ids_q2):
        padding_len = self.max_seq_len - len(input_ids_q1) - len(input_ids_q2)
        # input token
        input_ids = input_ids_q1 + input_ids_q2 + [0] * padding_len
        # segment token
        token_type_ids = [0] * len(input_ids_q1) + [1] * len(input_ids_q2) + [0] * padding_len
        # attention mask
        attention_mask = [1] * (len(input_ids_q1) + len(input_ids_q2)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask

class LCQMC_Dataset4encode(Dataset):
    def __init__(self, tokenized_q1s, tokenized_q2s, with_label, label=None):
        self.tokenized_q1s = tokenized_q1s
        self.tokenized_q2s = tokenized_q2s
        self.label = label
        # self.max_q1_len = 35
        # self.max_q2_len = 35
        self.with_label = with_label
        # [CLS] + q1 + [SEP] + q2 + [SEP]
        #self.max_seq_len = 1 + self.max_q1_len + 1 + self.max_q2_len + 1

    def __len__(self):
        #return len(self.data)
        return len(self.tokenized_q1s.input_ids)

    def __getitem__(self, idx):
        if self.with_label:
            this_label = self.label[idx]
        tokenized_q1 = self.tokenized_q1s[idx]
        tokenized_q2 = self.tokenized_q2s[idx]
        
        ids_q1 = tokenized_q1.ids
        typeid_q1 = tokenized_q1.type_ids
        attention_q1 = tokenized_q1.attention_mask

        ids_q2 = tokenized_q2.ids
        typeid_q2 = tokenized_q2.type_ids
        attention_q2 = tokenized_q2.attention_mask
        #input_ids_q1 = [101] + tokenized_q1.ids[:self.max_q1_len] + [102]

        #input_ids_q2 = tokenized_q2.ids[:self.max_q2_len] + [102]
        
        # Pad sequence and obtain inputs to model 
        #input_ids, token_type_ids, attention_mask = self.padding(input_ids_q1, input_ids_q2)
        if self.with_label:
            return [torch.tensor(ids_q1), torch.tensor(typeid_q1), torch.tensor(attention_q1)], [torch.tensor(ids_q2), torch.tensor(typeid_q2), torch.tensor(attention_q2)], this_label
        else:
            return [torch.tensor(ids_q1), torch.tensor(typeid_q1), torch.tensor(attention_q1)], [torch.tensor(ids_q2), torch.tensor(typeid_q2), torch.tensor(attention_q2)]