import pickle

import numpy as np

"""
构建词典,实现方法把句子转换为序列,和其翻转
"""


class Word2Sequence(object):
    # 2个特殊类属性，标记特殊字符和填充标记
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            # 保存词语和对应的数字
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}  # 统计词频

    def fit(self, sentence):
        """
        把单个句子保存到dict中
        :param sentence: [word1, word2 , ... , ]
        :return:
        """
        for word in sentence:
            # 对word出现的频率进行统计，当word不在sentence时，返回值是0，当word在sentence中时，返回+1，以此进行累计计数
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_features=None):
        """
        生成词典
        :param min:最小词频数
        :param max:最大词频数
        :param max_feature:一共保留多少词语
        :return:
        """
        # 删除count < min 的词语,即保留count > min 的词语
        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value > min}
        # 删除count > min 的词语,即保留count < max 的词语
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value < max}
        # 限制保留的词语数
        if max_features is not None:
            # sorted 返回一个列表[(key1, value1), (key2, value2),...]，True为升序
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features]
            self.count = dict(temp)
            for word in self.count:
                self.dict[word] = len(self.dict)

        # 得到一个翻转的dict字典
        # zip方法要比{value: word for word, value in self.dict.items()}快
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        把句子转换为序列
        :param sentence: [word1, word2...]
        :param max_len: 对句子进行填充或者裁剪
        :return:
        """
        if max_len is not None:
            # 句子长度小于最大长度，进行填充
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            # 句子长度大于最大长度，进行裁剪
            if max_len < len(sentence):
                sentence = sentence[:max_len]
        # for word in sentence:
        #     self.dict.get(word, self.UNK)
        # 字典的get(key, default=None) 如果指定键不存在，则返回默认值None。
        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        """
        把序列转换为句子
        :param indices: [1, 2, 3, ...]
        :return:
        """
        return [self.inverse_dict.get(idx) for idx in indices]

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    ws = Word2Sequence()
    ws.fit(["我", "是", "我"])
    ws.fit(["我", "是", "你"])
    ws.build_vocab(min=1, max_features=156)
    pickle.dump(ws, open('./models/ws.pkl', 'wb'))
    print(ws.dict)
    ret = ws.transform(['我', '爱', '北京'], max_len=10)
    print(ret)
    print(ws.inverse_transform(ret))