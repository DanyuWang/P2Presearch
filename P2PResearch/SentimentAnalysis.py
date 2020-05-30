import os
import glob
import gensim
import pandas as pd
import numpy as np
from P2PResearch.P2PEyeScraping import P2PEyeCrawler


class SentimentAnalysis(P2PEyeCrawler):
    def comments_major_tag(self):
        filenames = glob.glob(self._comments_dir + '/*.csv')
        result_set = {np.NAN}

        for file in filenames:
            temp_df = pd.read_csv(file)
            result_set = result_set | set(temp_df['major_tag'])

        self.writing_set2txt('Major Tags', result_set)
        return 0

    def comments_minor_tag(self):
        filenames = glob.glob(self._comments_dir + '/*.csv')
        result_set = {np.NAN}

        for file in filenames:
            temp_df = pd.read_csv(file)
            minor_list = temp_df['minor_tags']
            for index, value in minor_list.items():
                if type(value) is str:
                    sep_tags = value.split(', ')
                    result_set = result_set | set(sep_tags)

        self.writing_set2txt('Minor Tags', result_set)
        return 0

    @staticmethod
    def writing_set2txt(filename, input_set):
        from P2PResearch.P2PEyeScraping import DATA_DIR
        txts_name = os.path.join(DATA_DIR, 'NLP_model', filename)

        input_list = list(input_set)
        with open(txts_name, 'w') as f:
            for word in input_list:
                f.write("%s\n" % word)
        f.close()

    @staticmethod
    def test_knn_result():
        model = gensim.models.Word2Vec.load('/Users/holly/Desktop/毕设/Data/(旧)PlatformsComments/p2p.word2vec.model')
        X = model[model.wv.vocab]
        from nltk.cluster import KMeansClusterer
        import nltk
        NUM_CLUSTERS = 5
        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

        words = list(model.wv.vocab)
        cluster_dict = {0: [], 1: [], 2: [], 3: [], 4: []}
        for i, word in enumerate(words):
            cluster_dict[assigned_clusters[i]].append(word)

        for j in range(5):
            with open(os.path.join('/Users/holly/Desktop/毕设/Data/(旧)PlatformsComments/result', str(j)+'.txt'), 'w') as f:
                for word in cluster_dict[j]:
                    f.write("%s\n" % word)
                f.close()


if __name__ == '__main__':
    s = SentimentAnalysis()
    s.comments_major_tag()
