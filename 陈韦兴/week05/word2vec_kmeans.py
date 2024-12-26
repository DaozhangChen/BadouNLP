#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import torch
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip() #去除首尾空格
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r".\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    sentence_index = 0
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        cluster_centroid_list = np.array(kmeans.cluster_centers_)

        min_distance_list = []
        for i in range(len(cluster_centroid_list)):
            sentence_vector = vectors[sentence_index]
            min_distance_list.append(np.linalg.norm(sentence_vector - cluster_centroid_list[i])) # 使用欧几里得距离计算最小距离
        min_distance = min(min_distance_list) # 取出与中心点的最小距离
        sentence_label_dict[label].append({"sentence":sentence,"min_distance":min_distance})         #同标签的放到一起，放入句子与句子与中心点的最小距离
        sentence_index += 1
    for label, sentences_and_distance in sentence_label_dict.items():
        sentences_and_distance.sort(key=lambda x: x["min_distance"]) # 按最小距离排序
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences_and_distance))):  #随便打印几个，太多了看不过来
            print(sentences_and_distance[i]["sentence"].replace(" ", ""),sentences_and_distance[i]["min_distance"])
        print("---------")

if __name__ == "__main__":
    main()

