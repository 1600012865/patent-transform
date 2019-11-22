import pandas as pd
import numpy as np
import os
import torch


def word_check(word):

    w = word.lower()
    flag = True if len(w) > 0 else False
    for c in w:
        if not (c >= 'a' and c <= 'z') and c != "'" and c!= "-":
            flag = False
            break
    return flag

def get_function_words(name='tmp.txt'):

    if os.path.exists('function_words.pth'):
        print('==> Found saved function workds!')
        return torch.load('function_words.pth')
    
    print('==> Compute function words on the scratch!')
    function_words = []
    with open(name) as f:
        
        lines = f.readlines()
        for line in lines:
            words = line.split(' ')
            for w in words:
                w = w.lower()
                if word_check(w):
                    function_words.append(w)
    return function_words

def extract_patent(name, function_words):
    code = []
    des = []
    data = pd.read_csv(name)
    data = np.array(data[data.columns[0]])[:-1]
    for d in data:
        d = d.lower().split(';')
        code.append(d[0])
        tmp_des = d[-1]
        res = []
        for w in tmp_des.split(' '):
            w = w.lower().strip()
            if word_check(w) and not w in function_words:
                res.append(w)
        des.append(res)

    return code, des

def extract_classification_system(name, length=4):
    data = pd.read_excel(name)
    code = np.array(data['Commodity Code'])
    des = np.array(data['Commodity description'])
    code = code[:-1] #remove the last line (all commodity)
    des = des[:-1]

    code_length = np.array([len(i) for i in code])
    mask = code_length == length
    code = code[mask]
    des = des[mask]

    return code, des

def keywords_processing(line, function_words):
    res = []
    des_line = line.split(' ')
    for w in des_line:
        w = w.lower().strip(',')
        if word_check(w) and not w in function_words:
            res.append(w)
    return res
        
def extract_keywords(des, function_words):
    keywords = []
    for i in range(des.shape[0]):
        keywords.append(keywords_processing(des[i], function_words))
    return keywords

def cut_off(matrix, thresh=0.02):
    
    matrix[matrix <= thresh] = 0
    matrix = matrix / (1e-10 + matrix.sum(axis=1, keepdims=True))
    return matrix

if __name__ == '__main__':

    function_words = get_function_words()


    name1 = 'SITC Rev2.xls'
    sys_code, sys_des = extract_classification_system(name1)
    sys_des = extract_keywords(sys_des, function_words)

    name2 = 'tls902_ipc_nace2.csv'
    patent_code, patent_des = extract_patent(name2, function_words)

    sys_dict = set(sys_code)
    sys_num = len(sys_dict)
    sys_dict = dict(zip(sys_dict, range(sys_num)))

    patent_dict = set(patent_code)
    patent_num = len(patent_dict)
    patent_dict = dict(zip(patent_dict, range(patent_num)))

    freq = np.zeros([sys_num, patent_num])
    for i in range(sys_code.shape[0]):
        for j in range(len(patent_code)):

            tmp_sys_code = sys_code[i]
            tmp_patent_code = patent_code[j]

            tmp_sys_index = sys_dict[tmp_sys_code]
            tmp_patent_index = patent_dict[tmp_patent_code]

            tmp_sys_des = sys_des[i]
            tmp_patent_des = patent_des[j]

            if len(set(tmp_sys_des) & set(tmp_patent_des)) > 0:
                freq[tmp_sys_index, tmp_patent_index] += 1

    raw_freq = freq.copy()
    specify_freq = freq.copy()
    hybrid_freq = freq.copy()

    A = freq.sum(axis=0)
    A_weight = A / (1e-10 + A.sum())

    raw_freq = raw_freq / (1e-10 + raw_freq.sum(axis=0, keepdims=True))
    raw_freq = raw_freq * A_weight.reshape(1, -1)
    raw_freq = raw_freq / (1e-10 + raw_freq.sum(axis=1, keepdims=True))

    specify_freq = specify_freq / (1e-10 + specify_freq.sum(axis=0, keepdims=True))
    specify_freq = specify_freq / (1e-10 + specify_freq.sum(axis=1, keepdims=True))

    hybrid_freq = hybrid_freq / (1e-10 + hybrid_freq.sum(axis=0, keepdims=True))
    hybrid_freq = hybrid_freq * raw_freq
    hybrid_freq = hybrid_freq / (1e-10 + hybrid_freq.sum(axis=1, keepdims=True))

    raw_freq = cut_off(raw_freq)
    specify_freq = cut_off(specify_freq)
    hybrid_freq = cut_off(hybrid_freq)
    import ipdb; ipdb.set_trace()