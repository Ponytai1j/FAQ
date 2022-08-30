
import pandas as pd
from utilities import preprocess_text

def read_data(Q1_raw):
    """
    Args:
    Q1_raw(list): 原始语料，一个列表每个元素是一个字符串 example: ['我喜欢你。','你喜欢~我吗？'...]
                  
    Return:
    corpus_dict(dict): 处理后的数据，key为docid，value为分词处理后的string example:
                       
    """
    print('processing')
    tokenized_Q1 = preprocess_text(Q1_raw, seg='jieba')
    
    corpus_dict = {}
    for i in range(len(tokenized_Q1)):
        corpus_dict[i] = tokenized_Q1[i]

    # query_dict = {}
    # for i in range(len(tokenized_Q2)):
    #     query_dict[i] = tokenized_Q2[i]
    return corpus_dict
def read_data_xxwx(path):
    """
    公司数据集读入
    Args:
        path: 文件路径
    Return:
        corpus_dict(dict): 处理后的数据，key为qid(int): id,topic(string):主题,Q1(string):q1问题,Q2(list):相似问题集合,A(string):对应回答,keyword(string):关键词,addition(string):附加信息
    """
    # 读入数据
    df = pd.read_excel(path, header=None)
    corpus_dict = {}
    for i in range(3, 300):
        per_sample = df.iloc[i]
        if per_sample.isna()[1] == False: # 如果存在qid则更新Qid
            qid = df.iloc[i][1]
        if per_sample.isna()[0] == False: # 如果主题存在
            topic = df.iloc[i][0]
        temp_q2 = per_sample[3].replace('\\n','')
        temp_q2 = temp_q2.replace('?', '？') # 字符串有两种问号，设为一种来分割
        temp_q2 = temp_q2.replace(' ', '')
        temp_q2 = temp_q2.split('？')
        if len(temp_q2[-1]) == 0: temp_q2 = temp_q2[:-1] # 切掉空白字符串
        temp_q2 = [j+'？' for j in temp_q2]
        corpus_dict[i] = {'qid':qid, 'topic':topic, 'Q1': per_sample[2], 'Q2': temp_q2, 'A':per_sample[4], 'keyword':per_sample[5], "addition":per_sample[6]}
    return corpus_dict
    
if __name__ == '__main__':
    # 测试是否正确将原始数据读入成字典
    path = './Dataset/xxwx/FAQ_parent_education1-297.xls'
    corpus = read_data_xxwx(path)
    print(corpus)