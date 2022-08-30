from re import X
from telnetlib import DO
from turtle import mode
from sklearn.ensemble import AdaBoostClassifier
import pickle
from DataLoader import read_data_xxwx
from Jaccard import Jaccard
from inverted import InvertedRetrieve
from bm25 import BM25
import jieba
import numpy as np
from tqdm import tqdm
from sensitive_detecter import Sensitive_detector
import ipdb
class FAQ(object):
    def init(self, datapath=None, indexpath=None):
        self.datapath = datapath
        self.data = None 
        self.indexpath = indexpath
        self.inverted_retriever= None
        self.bm25 = None
        self.topk = 5
        self.adaboost = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, 
                                            algorithm='SAMME.R', random_state=None)  # adaboost
        self.sensitive_detector = Sensitive_detector(self.indexpath)
    def init_all(self):
        if self.datapath: 
            self.data = read_data_xxwx(self.datapath) # 读入数据整理成字典格式
            with open(self.indexpath+'corpus_dict.pkl', 'wb') as f:
                pickle.dump(self.data, f)
        else: 
            with open(self.indexpath+"corpus_dict.pkl", "rb") as tf: # 若没有输入datapath，则读入已配置的数据
                self.data = pickle.load(tf)
        self.inverted_retriever = InvertedRetrieve(corpus_dict=self.data)
        self.inverted_retriever.create_index() # 保存index
        self.bm25 = BM25(self.inverted_retriever)
        self.bm25.create_index()# 保存index
        
    def load_all(self):
        with open(self.indexpath+"corpus_dict.pkl", "rb") as tf: # 若没有输入datapath，则读入已配置的数据
                self.data = pickle.load(tf)
        self.inverted_retriever = InvertedRetrieve(index_path='./config/')
        self.bm25 = BM25(index_path='./config/')
        with open(self.indexpath+'adaboost.pickle', 'rb') as f:
            self.adaboost = pickle.load(f)
    def train(self, train_data):
        """
        训练adaboost时使用的是Q2[0]作为待匹配候选,Q2[1]作为输入的query,所以需要重建新的倒排,bm25,因为FAQ初始化使用的是Q1
        """
        inverted_retriever_for_train = InvertedRetrieve(corpus_dict=self.data, mode='train')
        bm25_for_train = BM25(inverted_retriever_for_train)
        self.adaboost = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, 
                                            algorithm='SAMME.R', random_state=None)  # adaboost
        print('******正在构建特征******')
        # 倒排召回
        X_train = []
        y_train = []
        for key, value in train_data.items(): # value['Q2']是相似问题list 
            this_query = value['Q2'][1]
            bm25_result = bm25_for_train.search(this_query)
            if len(bm25_result) < 2: continue # 搜索到的query不到两个无法构造训练对
            for each_result in bm25_result:
                if each_result[0] == key:
                    # 构建正样本
                    query1 = ' '.join(jieba.lcut(value['Q2'][0]))
                    query2 = ' '.join(jieba.lcut(value['Q2'][1]))
                    this_bm25_score = each_result[1]
                    this_jaccard_score = Jaccard(query1, query2)
                    X_train.append([this_bm25_score, this_jaccard_score])
                    y_train.append(1)
                    # 构建负样本
                    neg_index = np.random.randint(len(bm25_result))
                    negsample_key = bm25_result[neg_index][0]
                    if negsample_key == key: negsample_key = bm25_result[(neg_index + 1) % len(bm25_result)][0] # 若随机到正确样本则选下一条
                    target_query = train_data[negsample_key]['Q2'][0]
                    query1 = ' '.join(jieba.lcut(target_query))
                    query2 = ' '.join(jieba.lcut(this_query))
                    this_bm25_score = bm25_result[neg_index][1]
                    this_jaccard_score = Jaccard(query1, query2)
                    X_train.append([this_bm25_score, this_jaccard_score])
                    y_train.append(0)
                    break
        print("******特征构建完成******")

        print("******Adaboost开始训练******")
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.adaboost.fit(X_train, y_train)
        print("******Adaboost训练完成*******")

        print("******正在保存Adaboost模型******") 
        # 保存模型
        with open(self.indexpath+'adaboost.pickle', 'wb') as f:
            pickle.dump(self.adaboost, f)
        print("******Adaboost模型已保存******")
    def eval(self, test_data):
        """
        测试时使用Q2[0]作为输入的query,Q1作为待匹配候选集
        """
        #self.test_data = test_data # dict{'id'(int):}
        inverted_retriever_for_eval = InvertedRetrieve(test_data, mode='eval')
        bm25_for_eval = BM25(inverted_retriever_for_eval)
        right_num = 0
        print('******正在测试******')
        for key, value in tqdm(test_data.items()):
            cnt = 0
            X_eval = []
            bm25_res = bm25_for_eval.search(value['Q2'][0])
            for i in bm25_res:
                this_index = i[0]
                this_bm25_score = i[1]
                if key == this_index: target = cnt # 记录正确匹配对位于结果中的位置
                this_pair = test_data[this_index]['Q1']
                query1 = ' '.join(jieba.lcut(this_pair))
                query2 = ' '.join(jieba.lcut(value['Q2'][0]))
                this_jaccard_score = Jaccard(query1, query2)
                X_eval.append([this_bm25_score, this_jaccard_score])
                cnt += 1
            X_eval = np.array(X_eval)
            y_pred = self.adaboost.predict_proba(X_eval)[:,1]
            y_pred = y_pred.argsort()[-self.topk:][::-1] # 取topk
            if target == y_pred[0]: right_num += 1
        acc = right_num / len(test_data) * 100
        print('******测试完成******')
        print('top1准确率:'+str(acc))
        
    def search(self, query):
        X_search = []
        bm25_res = self.bm25.search(query)
        for i in bm25_res:
            this_index = i[0]
            this_bm25_score = i[1]
            this_pair = self.data[this_index]['Q1']
            query1 = ' '.join(jieba.lcut(this_pair))
            query2 = ' '.join(jieba.lcut(query))
            this_jaccard_score = Jaccard(query1, query2)
            X_search.append([this_bm25_score, this_jaccard_score])
        X_search = np.array(X_search)
        y_pred_score = self.adaboost.predict_proba(X_search)[:,1] # 预测结果分数
        y_pred_topk = y_pred_score.argsort()[-self.topk:][::-1] # topk在预测结果中的位置
        y_pred_index = [bm25_res[i][0] for i in y_pred_topk] # topk在语料库位置
        show_result = []
        for i in zip(y_pred_index, y_pred_topk):
            show_result.append((self.data[i[0]]['Q1'], self.data[i[0]]['A'],y_pred_score[i[1]] ))
        print(show_result)
        sensitive_res = self.sensitive_detector.detect(query)
        print(sensitive_res)
        return([show_result, sensitive_res, query])
        

if __name__ == "__main__":
    data_path = './Dataset/xxwx/FAQ_parent_education1-297.xls'
    # faq = FAQ()
    # faq.init(datapath=data_path, indexpath='./config/')
    # faq.init_all()
    # faq.train(train_data=faq.data)
    # faq.eval(test_data=faq.data)
    # faq.search('我家孩子没有上进心')
    # faq.search('我的孩子没有主见')
    # faq.search('孩子青春期叛逆')
    faq = FAQ()
    faq.init(datapath=data_path, indexpath='./config/')
    faq.load_all()
    faq.search('孩子青春期叛逆')

