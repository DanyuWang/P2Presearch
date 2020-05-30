from P2PResearch.TextCNNModel.Model_Config import *
import tensorflow as tf
import tensorflow.keras as kr
import os
import pandas as pd
import numpy as np
import jieba
import re
import heapq
import codecs


def predict(sentences):
    config = TextConfig()
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)
    model = TextCNN(config)
    save_dir = './checkpoints'
    save_path = os.path.join(save_dir, 'best_validation')

    _, word_to_id = read_vocab(config.vocab_filename)
    input_x = process_file(sentences, word_to_id, max_length=config.seq_length)
    labels = {0: 0,
              1: 1}

    feed_dict = {
        model.input_x: input_x,
        model.keep_prob: 1,
    }
    session = tf.compat.v1.Session()
    session.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess=session, save_path=save_path)
    y_prob = session.run(model.prob, feed_dict=feed_dict)
    y_prob = y_prob.tolist()
    cat = []
    for prob in y_prob:
        top2 = list(map(prob.index, heapq.nlargest(1, prob)))
        cat.append(labels[top2[0]])
    tf.compat.v1.reset_default_graph()
    return cat


def sentence_cut(sentences):
    """
    Args:
        sentence: a list of text need to segment
    Returns:
        seglist:  a list of sentence cut by jieba

    """
    jieba.add_word('跑路')
    jieba.add_word('失联')
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    seglist = []
    for sentence in sentences:
        words = []
        blocks = re_han.split(sentence)
        for blk in blocks:
            if re_han.match(blk):
                words.extend(jieba.lcut(blk))
        seglist.append(words)
    return seglist


def process_file(sentences, word_to_id, max_length=600):
    """
    Args:
        sentence: a text need to predict
        word_to_id:get from def read_vocab()
        max_length:allow max length of sentence
    Returns:
        x_pad: sequence data from  pre-processing sentence

    """
    data_id = []
    seglist = sentence_cut(sentences)
    for i in range(len(seglist)):
        data_id.append([word_to_id[x] for x in seglist[i] if x in word_to_id])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    return x_pad


def read_vocab(vocab_dir):
    """
    Args:
        filename:path of vocab_filename
    Returns:
        words: a list of vocab
        word_to_id: a dict of word to id

    """
    words = codecs.open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]


if __name__ == '__main__':
    print('Predict data.... ')
    sentences = []
    labels = []
    """with codecs.open('./data/cnews.test.txt', 'r', encoding='utf-8') as f:
        sample = random.sample(f.readlines(), 20)
        for line in sample:
            try:
                line = line.rstrip().split('\t')
                assert len(line) == 2
                sentences.append(line[1])
                labels.append(line[0])
            except:
                pass"""
    df_temp = pd.read_csv('/Users/holly/PycharmProjects/untitled/venv/P2PResearch/TextCNNModel/data/Split_Predict/4.csv')
    for index, text in df_temp['text'].iteritems():
        sentences.append(text)
    # print(sentences[1])
    cat = predict(sentences)

    """for i, sentence in enumerate(sentences, 0):
        print('----------------------the text-------------------------')
        print(sentence[:50] + '....')
        print('the orginal label:%s' % labels[i])
        print('the predict label:%s' % cat[i])"""
    df_temp['panic_flag'] = cat
    df_temp.to_csv('/Users/holly/PycharmProjects/untitled/venv/P2PResearch/TextCNNModel/data/Split_Predict/4.csv', index=False)
