from operator import mod
from os import access
from pmdarima.arima import auto_arima
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd 


# m01滑动arima模型 目前只支持每次向后预测1个时间单位
def rollWindowArima(sequence,window=15,pred_n=1):
    
    """
    在使用之前定义 forecasts=[] 作为全局变量 
    @sequence:Dataframe,1列
    @window:窗口大小
    @pred_n:每个窗口向后预测的步长

    """
    
    def calculate(train):
        model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(train)
        forecast = model.predict(n_periods =pred_n)
        global forecasts
        forecasts = np.append(forecasts,forecast)
        return forecast
    sequence.rolling(window,min_periods=window).apply(calculate)

    #作图
    forecasts=pd.DataFrame(forecasts[:-1],index = sequence.index[window:],columns=['Prediction'])
    plt.figure(figsize=(14,8), dpi=800)
    plt.plot(sequence[window:],label='Valid')
    plt.plot(forecasts, label='Prediction')
    plt.legend()
    rmse = mean_squared_error(sequence[window:], forecasts[:-1], squared=False)
    plt.title('RMSE : %.4f' % rmse)
    plt.show()

    return forecasts
# -----------------------------------------------------------------------------------
# m02 Apriori算法，用于关联分析。
class Apriori:
    """使用时候调用find_rules()即可"""
    def __init__(self, min_support, min_confidence):
        self.min_support = min_support # 最小支持度
        self.min_confidence = min_confidence # 最小置信度

    def count(self, filename='apriori.txt'):
        self.total = 0 # 数据总行数
        items = {} # 物品清单

        # 统计得到物品清单
        with open(filename) as f:
            for l in f:
                self.total += 1
                for i in l.strip().split(','): # 以逗号隔开
                    if i in items:
                        items[i] += 1.
                    else:
                        items[i] = 1.

        # 物品清单去重，并映射到ID
        # items:{实体：支持度}
        #item2id:{实体：index}
        self.items = {i:j/self.total for i,j in items.items() if j/self.total > self.min_support}
        self.item2id = {j:i for i,j in enumerate(self.items)}

        # 物品清单的0-1矩阵
        self.D = np.zeros((self.total, len(items)), dtype=bool)

        # 重新遍历文件，得到物品清单的0-1矩阵 形状（总行数，items个数）
    
        with open(filename) as f:
            for n,l in enumerate(f):
                for i in l.strip().split(','):
                    if i in self.items:
                        self.D[n, self.item2id[i]] = True

    def find_rules(self, filename='./model_text/apriori.txt'):
        self.count(filename)
        rules = [{(i,):j for i,j in self.items.items()}] # 记录每一步的频繁项集
        l = 0 # 当前步的频繁项的物品数

        while rules[-1]: # 包含了从k频繁项到k+1频繁项的构建过程
            rules.append({})
            keys = sorted(rules[-2].keys()) # 对每个k频繁项按字典序排序（核心）
            num = len(rules[-2])
            l += 1
            for i in range(num): # 遍历每个k频繁项对
                for j in range(i+1,num):
                    # 如果前面k-1个重叠，那么这两个k频繁项就可以组合成一个k+1频繁项
                    if keys[i][:l-1] == keys[j][:l-1]:
                        _ = keys[i] + (keys[j][l-1],)
                        _id = [self.item2id[k] for k in _]
                        support = 1. * sum(np.prod(self.D[:, _id], 1)) / self.total # 通过连乘获取共现次数，并计算支持度
                        if support > self.min_support: # 确认是否足够频繁
                            rules[-1][_] = support

        # 遍历每一个频繁项，计算置信度
        result = {}
        self.rule = rules
        for n,relu in enumerate(rules[1:]): # 对于所有的k，遍历k频繁项
            for r,v in relu.items(): # 遍历所有的k频繁项
                for i,_ in enumerate(r): # 遍历所有的排列，即(A,B,C)究竟是 A,B -> C 还是 A,B -> C 还是 A,B -> C ？
                    x = r[:i] + r[i+1:]
                    confidence = v / rules[n][x] # 不同排列的置信度
                    if confidence > self.min_confidence: # 如果某种排列的置信度足够大，那么就加入到结果
                        result[x+(r[i],)] = (confidence, v)
        self.result = sorted(result.items(), key=lambda x: -x[1][0]) # 按置信度降序排列
        return self.result


# ---------------------------------------------------------------------------------------------
#m03寻找新词算法  算法地址http://www.matrix67.com/blog/archives/5044
import re
from numpy import log,min
txtpath = './model_text/m3.txt'
def find_newword(txtpath):
    """
@ txtpath: txt文件路径
"""
    f = open(txtpath, 'r',encoding='utf-8') #读取文章
    s = f.read() #读取为一个字符串

    #定义要去掉的标点字
    drop_dict = [u'，', u'\n', u'。', u'、', u'：', u'(', u')', u'[', u']', u'.', u',', u' ', u'\u3000', u'”', u'“', u'？', u'?', u'！', u'‘', u'’', u'…',u'@']
    for i in drop_dict: #去掉标点字
        s = s.replace(i, '')

    #为了方便调用，自定义了一个正则表达式的词典
    myre = {2:'(..)', 3:'(...)', 4:'(....)', 5:'(.....)', 6:'(......)', 7:'(.......)'}

    min_count = 10 #录取词语最小出现次数
    min_support = 30 #录取词语最低支持度，1代表着随机组合
    min_s = 3 #录取词语最低信息熵，越大说明越有可能独立成词
    max_sep = 4 #候选词语的最大字数
    t=[] #保存结果用。

    t.append(pd.Series(list(s)).value_counts()) #逐字统计
    tsum = t[0].sum() #统计总字数
    rt = [] #保存结果用

    for m in range(2, max_sep+1):
        print(u'正在生成长度为%s的字词...'%m)
        t.append([])
        for i in range(m): #生成所有可能的m字词
            t[m-1] = t[m-1] + re.findall(myre[m], s[i:])
        
        t[m-1] = pd.Series(t[m-1]).value_counts() #逐词统计
        t[m-1] = t[m-1][t[m-1] > min_count] #最小次数筛选
        tt = t[m-1][:]
        for k in range(m-1):
            qq = np.array(list(map(lambda ms: tsum*t[m-1][ms]/t[m-2-k][ms[:m-1-k]]/t[k][ms[m-1-k:]], tt.index))) > min_support #最小支持度筛选。
            tt = tt[qq]
        rt.append(tt.index)

    def cal_S(sl): #信息熵计算函数
        return -((sl/sl.sum()).apply(log)*sl/sl.sum()).sum()

    for i in range(2, max_sep+1):
        print(u'正在进行%s字词的最大熵筛选(%s)...'%(i, len(rt[i-2])))
        pp = [] #保存所有的左右邻结果
        for j in range(i+2):
            pp = pp + re.findall('(.)%s(.)'%myre[i], s[j:])
        pp = pd.DataFrame(pp).set_index(1).sort_index() #先排序，这个很重要，可以加快检索速度
        index = np.sort(np.intersect1d(rt[i-2], pp.index)) #作交集
        #下面两句分别是左邻和右邻信息熵筛选
        index = index[np.array(list(map(lambda s: cal_S(pd.Series(pp[0][s]).value_counts()), index))) > min_s]
        rt[i-2] = index[np.array(list(map(lambda s: cal_S(pd.Series(pp[2][s]).value_counts()), index))) > min_s]

    #下面都是输出前处理
    for i in range(len(rt)):
        t[i+1] = t[i+1][rt[i]]
        t[i+1].sort_values(ascending = False)

    #保存结果并输出
    pd.DataFrame(pd.concat(t[1:])).to_csv('result.txt', header = False)


# find_newword('./model_text/m3.txt')

#04 寻找新词方法2
from collections import defaultdict #defaultdict是经过封装的dict，它能够让我们设定默认值
from tqdm import tqdm #tqdm是一个非常易用的用来显示进度的库
from math import log
import re

class Find_Words:
    def __init__(self, min_count=10, min_pmi=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.chars, self.pairs = defaultdict(int), defaultdict(int) #如果键不存在，那么就用int函数
                                                                  #初始化一个值，int()的默认结果为0
        self.total = 0.
    def text_filter(self, texts): #预切断句子，以免得到太多无意义（不是中文、英文、数字）的字符串
        for a in tqdm(texts):
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', a): #这个正则表达式匹配的是任意非中文、
                                                              #非英文、非数字，因此它的意思就是用任
                                                              #意非中文、非英文、非数字的字符断开句子
                if t:
                    yield t
    def count(self, texts): #计数函数，计算单字出现频数、相邻两字出现的频数
        for text in self.text_filter(texts):
            self.chars[text[0]] += 1
            for i in range(len(text)-1):
                self.chars[text[i+1]] += 1
                self.pairs[text[i:i+2]] += 1
                self.total += 1
        self.chars = {i:j for i,j in self.chars.items() if j >= self.min_count} #最少频数过滤
        self.pairs = {i:j for i,j in self.pairs.items() if j >= self.min_count} #最少频数过滤
        self.strong_segments = set()
        for i,j in self.pairs.items(): #根据互信息找出比较“密切”的邻字
            _ = log(self.total*j/(self.chars[i[0]]*self.chars[i[1]]))
            if _ >= self.min_pmi:
                self.strong_segments.add(i)
    def find_words(self, texts): #根据前述结果来找词语
        self.words = defaultdict(int)
        for text in self.text_filter(texts):
            s = text[0]
            for i in range(len(text)-1):
                if text[i:i+2] in self.strong_segments: #如果比较“密切”则不断开
                    s += text[i+1]
                else:
                    self.words[s] += 1 #否则断开，前述片段作为一个词来统计
                    s = text[i+1]
            self.words[s] += 1 #最后一个“词”
        self.words = {i:j for i,j in self.words.items() if j >= self.min_count} #最后再次根据频数过滤

# f = open('D:\Model\model_text\m3.txt',encoding='utf-8')
# text = f.read()
# fw = Find_Words(16, 1)
# fw.count(text)
# fw.find_words(text)

# --------------------------------------------------------------------------------------------
#05 基于AC自动机的快速分词
import ahocorasick
class AC:
    def __init__(self) -> None:
        dic = None

    def load_dic(self,dicfile):
        from math import log
        self.dic = ahocorasick.Automaton()
        total = 0
        with open(dicfile,encoding='utf-8') as dicfile:
            words = []
            for line in dicfile:
                line = line.split(' ')
                words.append((line[0],int(line[1])))
                total+=int(line[1])
        for i,j in words:
            self.dic.add_word(i,(i,log(j/total)))
        self.dic.make_automaton()
        return self.dic 
# 全模式分词
    def all_cut(self,sentence):
        words = []
        for i,j in self.dic.iter(sentence):
            words.append(j[0])
        return words
# 最大匹配法分词
    def max_match_cut(self,sentence):
        words = ['']
        for i in sentence:
            if self.dic.match(words[-1]+i):
                words[-1]+=i
            else:
                words.append(i)
        return words
    
#最大概率组合分词
    def max_proba_cut(self,sentence):
        paths={0:([],0)}
        end = 0
        for i,j in self.dic.iter(sentence):#i是结束的下标，
            start,end = 1+i-len(j[0]),i+1
            if start not in paths:#如果出现为登入词，就是说你上一个词的结尾不是这个词的开头
                last = max([i for i in paths if i <start])
                paths[start] = (paths[last][0]+[sentence[last:start]],paths[last][1]-10)
            proba = paths[start][1]+j[1]
            if end not in paths or proba >paths[end][1]:
                paths[end] =(paths[start][0]+[j[0]],proba)
        if end<len(sentence):
            return paths[end][0]+[sentence[end:]]
        else:
            return paths[end][0]


    def map_cut(self,sentence):#根据标点把句子分词很多部分，然后分词。
        to_break = ahocorasick.Automaton()
        for i in ['，', '。', '！', '、', '？', ' ', '\n']:
            to_break.add_word(i, i)
            to_break.make_automaton()
        start = 0
        words = []
        for i in to_break.iter(sentence):
            words.extend(self.max_proba_cut(sentence[start:i[0]+1]))
            start = i[0]+1
        words.extend(self.max_proba_cut(sentence[start:]))
        return words
#最少词数分词：用罚分——有多少个词罚多少分，未登录词再罚一分，最后罚分最少的胜出。
    def min_words_cut(self,sentence):
        paths = {0:([], 0)}
        end = 0
        for i,j in self.dic.iter(sentence):
            start,end = 1+i-len(j[0]), i+1
            if start not in paths:
                last = max([i for i in paths if i < start])
                paths[start] = (paths[last][0]+[sentence[last:start]], paths[last][1]+1)
            num = paths[start][1]+1
            if end not in paths or num < paths[end][1]:
                paths[end] = (paths[start][0]+[j[0]], num)
        if end < len(sentence):
            return paths[end][0] + [sentence[end:]]
        else:
            return paths[end][0]

