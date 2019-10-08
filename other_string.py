import distance
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import Levenshtein

# Levenshtein距离
def edit_distance(s1, s2):
    return distance.levenshtein(s1, s2)

def Levenshtein_test():
    filename = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/dataset/test.txt'
    output_file = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/other/Levenshtein_test.csv'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output_file, 'w', encoding='utf-8') as filehandler2:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                lable = line[2]
                d = edit_distance(s1, s2)
                filehandler2.write(str(lable) + ',' + str(d) + '\n')

def Levenshtein_train():
    filename = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/dataset/train.txt'
    output_file = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/other/Levenshtein_train.csv'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output_file, 'w', encoding='utf-8') as filehandler2:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                lable = line[2]
                d = edit_distance(s1, s2)
                filehandler2.write(lable + ',' + str(d) + '\n')

# Jaccard相似度
def Jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator

def Jaccard_train():
    filename = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/dataset/train.txt'
    output_file = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/other/Jaccard_train.csv'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output_file, 'w', encoding='utf-8') as filehandler2:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                lable = line[2]
                s = Jaccard_similarity(s1, s2)
                filehandler2.write(lable + ',' + str(s) + '\n')

def Jaccard_test():
    filename = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/dataset/test.txt'
    output_file = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/other/Jaccard_test.csv'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output_file, 'w', encoding='utf-8') as filehandler2:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                lable = line[2]
                s = Jaccard_similarity(s1, s2)
                filehandler2.write(lable + ',' + str(s) + '\n')

# Jaro距离
def Jaro_distance(s1, s2):
    return Levenshtein.jaro(s1, s2)

def Jaro_train():
    filename = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/dataset/train.txt'
    output_file = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/other/Jaro_train.csv'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output_file, 'w', encoding='utf-8') as filehandler2:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                lable = line[2]
                s = Jaro_distance(s1, s2)
                filehandler2.write(lable + ',' + str(s) + '\n')

def Jaro_test():
    filename = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/dataset/test.txt'
    output_file = 'D:/Lydia/PycharmProjects/Deep learning for geocoding/data/other/Jaro_test.csv'
    with open(filename, 'r', encoding='UTF-8') as f:
        with open(output_file, 'w', encoding='utf-8') as filehandler2:
            for line in f.readlines():
                line = line.strip().split("\t")
                s1 = line[0]
                print(s1)
                s2 = line[1]
                print(s2)
                lable = line[2]
                s = Jaro_distance(s1, s2)
                filehandler2.write(lable + ',' + str(s) + '\n')


Jaro_test()