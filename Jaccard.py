# class Jaccard(object):
#     """
#     杰卡德距离,实例化将会对输入的语料每一句去除重复的token
#     """
#     def __init__(corpus):



def Jaccard(str1, str2):
    """
    Args:
        str1,str2:分词后的两个字符串,example:'我 喜欢 你"
    """
    Q1_list = str1.split(' ')
    Q2_list = str2.split(' ')
    union = set(Q1_list + Q2_list)
    intersection = set()
    for token in Q1_list:
        if token in Q2_list:
            intersection.add(token)
    jaccard_score = len(intersection) / len(union)

    return 1-jaccard_score

if __name__ == "__main__":
    import jieba
    str1 = ' '.join(jieba.lcut('我喜欢学习, 因为能让我成绩变好'))
    str2 = ' '.join(jieba.lcut('只有努利学习，你的成绩才会向上'))
    str3 = ' '.join(jieba.lcut('吃麦当劳'))

    dist1 = Jaccard(str1, str2)
    dist2 = Jaccard(str1, str3)
    print(str1)
    print(str2)
    print(str3)
    print(dist1)
    print(dist2)