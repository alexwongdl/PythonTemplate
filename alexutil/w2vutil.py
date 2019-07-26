"""
Created by Alex Wang on 2019-07-29
"""
import os
import pickle
import numpy as np
import scipy
import traceback
import queue

from scipy.spatial import distance
from sklearn.cluster import KMeans

from alexutil.heap_sort import MinHeapSort


class SimilaryWords():
    def __init__(self):
        """
        word2vec_info = {'word_list': word_list, 'word_vec_map': word_vector_map,
                         'word_labels': word_labels, 'cluster_words': cluster_words,
                         'cluster_centers': cluster_centers}
        """
        print('load word2vec info...')
        data_dir = "/Users/alexwang/data/video_label"
        cluster_file = os.path.join(data_dir, 'waou_word2vec_clusters.pkl')

        with open(cluster_file, 'rb') as reader:
            word2vec_info = pickle.load(reader)

        self.cluster_centers = word2vec_info['cluster_centers']  # [n_clusters, n_features]
        self.word_vec_map = word2vec_info['word_vec_map']
        self.cluster_words = word2vec_info['cluster_words']  # dict(cluster_id, set(word))

    def cos_distance(self, vector_one, vector_two):
        """
        Calculating pairwise cosine distance using a common for loop with manually calculated cosine value.
        """
        return 1 - distance.cosine(vector_one, vector_two)

    def find_similar_words(self, query_word, cluster_num=10, words_num=20):
        """
        :param cluster_num:
        :param words_num:
        :return:
        """
        query_word = query_word.strip()

        if query_word not in self.word_vec_map:
            print('ERROR: word not find')
            return None

        print('get nearest cluster centers...')
        word_vec = self.word_vec_map[query_word]
        cluster_min_heap = MinHeapSort(cluster_num)
        for row in range(self.cluster_centers.shape[0]):
            cluster_min_heap.try_add_item(row, self.cos_distance(word_vec, self.cluster_centers[row, :]))
        nearest_clusters = cluster_min_heap.sort()

        print('get nearest words...')
        candidate_word_set = set()
        for (cluster_id, score) in nearest_clusters:
            candidate_word_set = candidate_word_set.union(self.cluster_words[cluster_id])

        words_min_heap = MinHeapSort(words_num)
        for word in candidate_word_set:
            if word == query_word:
                continue
            try:
                words_min_heap.try_add_item(word, self.cos_distance(word_vec, self.word_vec_map[word]))
            except Exception as e:
                traceback.print_exc()
                print(word)
        nearest_words = words_min_heap.sort()
        return nearest_words


def read_word2vec(file_path):
    """
    ./word2vec/bin/word2vec  -train waou_video_token_all_filted.txt -output waou_word2vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 0 -iter 15      3 hours
    :param file_path:
    :return:
    """
    word_list = []
    word_vector_map = dict()
    for line in open(file_path, 'r'):
        try:
            elems = line.strip().split(' ')
            if len(elems) < 200:
                continue
            word = elems[0]
            vec = elems[1:200]
            vec = [float(elem) for elem in vec]
            word_list.append(word)
            word_vector_map[word] = vec
        except Exception as e:
            traceback.print_exc()
    return word_list, word_vector_map


def word_vec_cluster():
    """
    total 196161 words
    create 200 cluters
    :return:
    """
    data_dir = '/alexwang'
    word2vec_file = os.path.join(data_dir, 'waou_word2vec.txt')
    cluster_file = os.path.join(data_dir, 'waou_word2vec_clusters.pkl')

    print('read word2vec file...')
    word_list, word_vector_map = read_word2vec(word2vec_file)
    word_vec_arr = np.array([word_vector_map[word] for word in word_list])

    print('training kmeans...')
    kmeans = KMeans(n_clusters=1000, init='k-means++', n_init=10, max_iter=300, verbose=1, n_jobs=5).fit(word_vec_arr)

    print('process kmeans result...')
    word_labels = kmeans.labels_
    cluster_words = dict()
    cluster_centers = kmeans.cluster_centers_
    for i in range(len(word_labels)):  # assert len(word_list) == len(word_labels)
        label = word_labels[i]
        word = word_list[i]
        if label in cluster_words:
            cluster_words[label].add(word)
        else:
            cluster_words[label] = set(word)

    word2vec_info = {'word_list': word_list, 'word_vec_map': word_vector_map,
                     'word_labels': word_labels, 'cluster_words': cluster_words,
                     'cluster_centers': cluster_centers}

    print('save model...')
    with open(cluster_file, 'wb') as writer:
        pickle.dump(word2vec_info, writer)


def tags_similarity():
    """

    :return:
    """
    data_dir = '/alexwang'
    tags_file = os.path.join(data_dir, 'tags_v0.1.txt')
    similar_file = os.path.join(data_dir, 'tags_v0.1_similarity.txt')

    fifo_queue = queue.Queue(maxsize=-1)
    processed_word_list = set()

    similar_words = SimilaryWords()
    tag_list = []

    with open(tags_file, 'r') as reader, open(similar_file, 'w') as writer:
        for line in reader:
            elems = line.strip().split(' ')
            tag_list.append(elems[0])

        tag_set = set(tag_list)
        print('size of tag_list:{}, size of tag_set:{}'.format(len(tag_list), len(tag_set)))

        for word in tag_list:
            if word not in processed_word_list:
                fifo_queue.put(word)
                processed_word_list.add(word)

            while not fifo_queue.empty():
                current_word = fifo_queue.get()
                nearest_words = similar_words.find_similar_words(current_word, cluster_num=10, words_num=20)
                if not nearest_words:
                    print('nearest_words is none:{}'.format(current_word))
                    continue

                similar_list = []
                similar_tag_list = []
                for (temp_word, score) in nearest_words:
                    similar_list.append(temp_word)
                    if temp_word in tag_set:
                        similar_tag_list.append(temp_word)
                        if temp_word not in processed_word_list:
                            fifo_queue.put(temp_word)
                            processed_word_list.add(temp_word)
                writer.write('{}\t{}\t{}\n'.format(current_word,
                                                   ','.join(similar_tag_list),
                                                   ','.join(similar_list)))


def test_word_similarity():
    similar_words = SimilaryWords()
    nearest_words = similar_words.find_similar_words("苹果")
    for (word, score) in nearest_words:
        print("{}\t{}".format(word, score))

    nearest_words = similar_words.find_similar_words("摄像头")
    for (word, score) in nearest_words:
        print("{}\t{}".format(word, score))

    nearest_words = similar_words.find_similar_words("测评")
    for (word, score) in nearest_words:
        print("{}\t{}".format(word, score))


if __name__ == '__main__':
    print('test')
    # test_word_similarity()
    # tags_similarity()
