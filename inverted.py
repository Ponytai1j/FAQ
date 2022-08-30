from operator import index
import numpy as np
import jieba
import argparse
import pickle
# TO DO MEMO
# 1.倒排索引初始化两个可能都为空，要写下异常处理
class InvertedRetrieve(object):
    """
    建立倒排索引器
    Args:
    corpus_dict(dict): 字典key为docid, value为分此后的字符串 {'docid':'tokenized sentence'} 
                        example: {1:'我 喜欢 你', 2:'你 喜欢 我', ...}
    """
    def __init__(self, corpus_dict=None, index_path='./config/', mode='eval'):
        """
        若传入corpus_dict则初始化新的index并保存,
        若传入index_path则按照路径里的文件初始化index
        Args:
            mode: 'eval'使用Q1建立待匹配语料, 'train'使用Q2[0]建立待匹配语料
        """
        self.mode = mode
        self.inverted_index = None
        if corpus_dict != None:
            self.inverted_index = {} # 倒排索引字典:{'term_name': Doc_id}
            self.doc_len = {} # 记录每个doc长度，以便后续BM25等使用
            self._construct_inverted_index(corpus_dict)
            
        else: # 读入已配置好的index
            print('******读取倒排配置******')
            with open(index_path+"inverted_retrieve.pkl", "rb") as tf:
                self.inverted_index = pickle.load(tf)
            print('******倒排读取配置完成******')
                

            
    
    def _construct_inverted_index(self, corpus_dict): # corpus:一个dict key是docid, value是分此后的文章
        # for docid, doc in corpus_dict.items():
        #     doc_split = doc.split(' ')
        #     self.doc_len[docid] = len(doc_split)
        #     for term in doc_split:
        #         if term in self.inverted_index: # 如果倒排表存在这个term
        #             term_dict = self.inverted_index[term] # 取出这个term_dict
        #             if docid in term_dict: term_dict[docid] += 1 # 如果之前term存在过这个doc
        #             else: term_dict[docid] = 1 # 如果之前term不存在这个doc
        #         else:
        #             self.inverted_index[term] = {docid: 1} # 如果之前不存在这个term

        # **********将字典中的语料预处理如分词，去停用词等************
        print('预处理语料分词，去停用词等')
        corpus_afterprocess = {}
        for key, value in corpus_dict.items():
            if self.mode =='eval':
                doc = value['Q1']
            elif self.mode =='train':
                doc = value['Q2'][0]
            docid = key
            doc = ' '.join(jieba.lcut(doc, cut_all=True))
            corpus_afterprocess[docid] = doc
        print('预处理完毕')
        # ******************构造倒排***************
        for docid, doc in corpus_afterprocess.items():
            doc_split = doc.split(' ')
            self.doc_len[docid] = len(doc_split)
            for term in doc_split:
                if term in self.inverted_index: # 如果倒排表存在这个term
                    term_dict = self.inverted_index[term] # 取出这个term_dict
                    if docid in term_dict: term_dict[docid] += 1 # 如果之前term存在过这个doc
                    else: term_dict[docid] = 1 # 如果之前term不存在这个doc
                else:
                    self.inverted_index[term] = {docid: 1} # 如果之前不存在这个term
    
    def search(self, query):
        """
        倒排查找出含有query中词的docid
        return: (list) 含有查找到的docid 列表
        Args:
        query(string): 待查询的字符串 example: '我喜欢你'
        """
        query_split = jieba.lcut(query)
        doc_searched = [] # 搜寻到的docid
        for term in query_split:
            if term in self.inverted_index:
                term_dict = self.inverted_index[term] # 包含该词汇的dict {docid(int): 在这个doc中出现次数(int)}
                doc_searched += list(term_dict.keys())
        doc_searched = np.array(doc_searched)
        doc_unique = np.unique(doc_searched)
#         res = []
#         for i in doc_unique:
#             if np.sum(doc_searched == i) >= 2:
#                 res.append(i)
        
        return doc_unique
    
    def create_index(self, config_fold='./config/'):
        """
        保存建立好的倒排索引到指定目录
        Args:
            path: 路径, 用来保存索引
        """
        with open(config_fold+'inverted_retrieve.pkl', 'wb') as f:
            pickle.dump(self.inverted_index, f)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='input your dataset path')
    # parser.add_argument('--data_path', type=str, default='./Dataset', help='Your dataset folder path')
    # args = parser.parse_args()
    # path = args.data_path
    # if path == None:
    #     print(' 1')
    # if path != None:
    #     print(' 2')
    path = './Dataset/xxwx/FAQ_parent_education1-297.xls'
    from DataLoader import read_data_xxwx
    #corpus = read_data_xxwx(path) 读取数据
    inverted_retrieve = InvertedRetrieve(index_path='./config/') # 读取已配置index
    #inverted_retrieve.create_index() # 保存index
    ans = inverted_retrieve.search('老师对待孩子不公平')
    print(ans) # 返回list，是搜索到的qid