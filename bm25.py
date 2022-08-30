import collections
import numpy as np
import jieba
import argparse
import pickle
# TO DO MEMO
# BM25重复创建了inverted_index,可以直接用invertedIndex的保存文件

class BM25(object):
    def __init__(self, inverted_retrieve=None, index_path='./config/', k1=1.5, b=0.75):
        """
        Args:
            inverted_retrieve(object): 一个倒排索引对象，需要先构建倒排索引再使用bm25 
        """
        self.doc_len = None
        self.inverted_index = None
        self.avg_len = None
        self.doc_num = None
        self.k1 = None
        self.b = None
        if inverted_retrieve != None:
            self.doc_len = inverted_retrieve.doc_len
            self.inverted_index = inverted_retrieve.inverted_index
            self.avg_len = self._compute_avg_len(self.doc_len)
            self.doc_num = len(self.doc_len)
            self.k1 = k1
            self.b = b
            print('1')
        else: # 读取已有配置
            print('******正在读取bm25配置******')
            with open(index_path+"bm25.pkl", "rb") as tf:
                self.config = pickle.load(tf)
                self.doc_len = self.config['doc_len']
                self.inverted_index = self.config['inverted_index']
                self.avg_len = self.config['avg_len']
                self.doc_num = self.config['doc_num']
                self.k1 = self.config['k1']
                self.b = self.config['b']
            print("******bm25配置读取完成******")

    
    def _compute_avg_len(self, doc_len):
        len_sum = 0
        for _,value in doc_len.items():
            len_sum += value
        avg_len = len_sum / len(doc_len)
        return avg_len
    
    def _compute_idf_score(self, word, inverted_index):
        doc_frequency = len(inverted_index[word]) # 出现这个词的文档个数
        score = np.log((self.doc_num+1) / (doc_frequency+1))+1 # 这个词的idf
        return score
    
    def _compute_bm25_score(self, word, inverted_index, doc_len): # 计算一个word与所有文档的bm25值
        score_dict = {} # 返回字典 docid, score
        if word not in inverted_index: # 如果query中的token在corpus从未出现过，则返回空字典
            return score_dict
        inverted_dict = inverted_index[word] # 取出dict，记录了这个单词出现的次数docid-term frequency
        idf_score = self._compute_idf_score(word, inverted_index)

        for docid, term_frequency in inverted_dict.items(): # 词汇-出现过该词汇的文章的BM25score
            docid_len = doc_len[docid]
            numerator = term_frequency * (self.k1+1)
            denominator = term_frequency + self.k1 * (1-self.b+self.b*(docid_len / self.avg_len))
            score_dict[docid] = idf_score * (numerator / denominator)
        return score_dict

    def search(self, query):
        """
        Args:
            query(str): 待查询字符串 
                        example: "我喜欢你"
        
        Return:
            list of tuple:  列表,每个element是一个tuple,已经按照避免bm25分数排序,[(docid, bm25 score),....]
                            example: [(1: 7.25),(9, 4.56)]
        """
        query_split = jieba.lcut(query, cut_all=True)
        score_list = []
        for token in query_split:
            score_dict = self._compute_bm25_score(token, self.inverted_index, self.doc_len)
            score_list.append(score_dict)
        counter = collections.Counter()
        for score_dict in score_list: 
            counter.update(score_dict)
        return counter.most_common()

    def create_index(self, config_folder='./config/'):
        """
        保存建立好的倒排索引到指定目录
        Args:
            path: 路径, 用来保存索引
        """
        self.config = {'doc_len':self.doc_len, 'inverted_index':self.inverted_index,
                        'avg_len':self.avg_len, 'doc_num':self.doc_num,
                        'k1':self.k1, 'b':self.b}
        with open(config_folder+'bm25.pkl', 'wb') as f:
            pickle.dump(self.config, f)
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='input your dataset path')
    # parser.add_argument('--data_path', type=str, default='./Dataset', help='Your dataset folder path')
    # args = parser.parse_args()
    # path = args.data_path
    path = './Dataset/xxwx/FAQ_parent_education1-297.xls'
    from DataLoader import read_data_xxwx
    from inverted import InvertedRetrieve
    #corpus = read_data_xxwx(path)
    #inverted_retriever = InvertedRetrieve(corpus)
    #bm25_searcher = BM25(inverted_retriever)
    bm25_searcher = BM25()
    #bm25_searcher.create_index()
    ans = bm25_searcher.search('老师对待孩子不公平')
    print(ans)
