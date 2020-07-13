"""
Created by Alex Wang on 2020-07-13
"""
import traceback
import pickle

import numpy as np
from sklearn import decomposition

def ipca_dim_reduce(pca_dim=128):
    data_dir = '/alexwang'
    input_file = os.path.join(data_dir, 'fea.txt')
    model_file = os.path.join(data_dir, 'fea_ipca_{}.pkl'.format(pca_dim))

    ipca = decomposition.IncrementalPCA(n_components=pca_dim)
    data_list = []
    processed_num = 0

    with open(input_file, 'r') as reader, open(model_file, 'wb') as writer:
        for line in reader:
            try:
                elems = line.split(',')
                arr = [float(elem) for elem in elems]
                data_list.append(np.array(arr))

                if len(data_list) >= 10000:
                    data_mat = np.array(data_list)
                    processed_num += data_mat.shape[0]
                    print('shape of data_mat:{}, {} record processed'.format(data_mat.shape, processed_num))
                    ipca.partial_fit(data_mat)

                    data_list = []

            except Exception as e:
                traceback.print_exc()

        if len(data_list) > 0:
            data_mat = np.array(data_list)
            processed_num += data_mat.shape[0]
            print('shape of data_mat:{}, {} record processed'.format(data_mat.shape, processed_num))
            ipca.partial_fit(data_mat)

        print('save model...')
        ipca_model_dict = {'mean': ipca.mean_, 'components': ipca.components_}
        pickle.dump(ipca_model_dict, writer)

if __name__ == '__main__':
    ipca_dim_reduce(pca_dim=256)