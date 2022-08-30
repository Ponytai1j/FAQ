import ahocorasick
class Sensitive_detector(object):
    def __init__(self, wordlist_path):
        """
        wordlist_path: 屏蔽词路径
        """
        res = [] # 接受屏蔽词
        path1 = wordlist_path+'sensitive_word/emergency_words.txt'
        path2 = wordlist_path+'sensitive_word/illegal_words.txt'
        with open(path1) as f:
                for line in f.readlines():##readlines(),函数把所有的行都读取进来；
                    line = line.strip()##删除行后的换行符，img_file 就是每行的内容啦
                    res.append(line)
        with open(path2) as f:
                for line in f.readlines():##readlines(),函数把所有的行都读取进来；
                    line = line.strip()##删除行后的换行符，img_file 就是每行的内容啦
                    res.append(line)

        self.actree = self._build_actree(res)

    def _build_actree(self,wordlist):
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
        actree.make_automaton()
        return actree
    def detect(self, sentence):
        sentence_cp = sentence
        for i in self.actree.iter(sentence):
            sentence_cp = sentence_cp.replace(i[1][1], "**")
            print("屏蔽词：",i[1][1])
        return sentence_cp


 
        
if __name__ == '__main__':
    sent = '我草你妈的， 你是傻逼吗？狗东西是个什么玩意儿'
    path = './Dataset/sensitive_word/'
    
    sensitive_detector = Sensitive_detector(path)
    print(sensitive_detector.detect(sent)) 