"""
Created by Alex Wang on 2019-09-09
"""
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_curve
from eval_util import EvaluationMetrics


def cal_accuracy(predict_all, tags_all):
    predict_all = np.concatenate(predict_all, axis=0)
    tags_all = np.concatenate(tags_all, axis=0)
    predict_argmax = np.argmax(predict_all, axis=1)

    equal_arr = np.equal(tags_all, predict_argmax)
    acc_num = sum([1 for item in equal_arr if item == True])
    accuracy = acc_num * 1. / len(equal_arr)
    return tags_all, predict_argmax, accuracy


def cal_weight_acc_and_recall(predict_all, tags_all, label_weight_map):
    label_weight_map[0] = 0
    predict_all = np.concatenate(predict_all, axis=0)
    tags_all = np.concatenate(tags_all, axis=0)
    predict_argmax = np.argmax(predict_all, axis=1)

    equal_arr = np.equal(tags_all, predict_argmax)

    acc_sum = sum([label_weight_map[predict_argmax[i]] for i in range(len(equal_arr)) if equal_arr[i] == True])
    total_acc_sum = sum([label_weight_map[predict_argmax[i]] for i in range(len(equal_arr))])

    recal_sum = sum([label_weight_map[tags_all[i]] for i in range(len(equal_arr)) if equal_arr[i] == True])
    total_recall_sum = sum([label_weight_map[tags_all[i]] for i in range(len(equal_arr))])

    true_num = sum([1 for item in equal_arr if item == True])
    print('length of equal_arr:{}, true_num:{}, acc_sum:{:.4f}, total_acc_sum:{:.4f},'
          ' recal_sum:{:.4f}, total_recall_sum:{:.4f}'.
          format(len(equal_arr), true_num, acc_sum, total_acc_sum, recal_sum, total_recall_sum))
    if total_acc_sum == 0 or total_recall_sum == 0:
        return 0, 0
    return acc_sum * 1. / total_acc_sum, recal_sum * 1. / total_recall_sum


def cal_accuracy_weight(predict_all, tags_all, sample_weight_map):
    predict_all = np.concatenate(predict_all, axis=0)
    tags_all = np.concatenate(tags_all, axis=0)
    predict_argmax = np.argmax(predict_all, axis=1)

    equal_arr = np.equal(tags_all, predict_argmax)
    acc_num = sum([1 for item in equal_arr if item == True])
    accuracy = acc_num * 1. / len(equal_arr)

    weight_devider = sum([max(sample_weight_map[predict_argmax[i]], sample_weight_map[tags_all[i]])
                          for i in range(len(tags_all))])
    weight_correct = sum([sample_weight_map[predict_argmax[i]] for i in range(len(predict_argmax)) if
                          predict_argmax[i] == tags_all[i]])

    return tags_all, predict_argmax, accuracy, weight_correct * 1. / weight_devider


def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]


def top_n_accuracy(preds, truths, k):
    """
    :param preds: n * num_class
    :param truths: n * 1
    :param k:
    :return:
    """
    best_n = np.argsort(preds, axis=1)[:, -k:]
    # ts = np.argmax(truths, axis=1)
    successes = 0
    for i in range(truths.shape[0]):
        if truths[i] in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def test_map():
    print(sklearn.__version__)

    y_true = np.array([[1, 0, 1], [0, 1, 1]])
    y_scores = np.array([[0.9, 0.2, 0.6], [0.8, 0.4, 0.7]])
    print(average_precision_score(y_true, y_scores))  # 1.0

    y_true = np.array([[1, 0, 1], [0, 0, 1]])
    y_scores = np.array([[0.9, 0.2, 0.6], [0.8, 0.4, 0.7]])
    print(average_precision_score(y_true, y_scores))  # nan

    y_true_1 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y_scores_1 = np.array([[0.008933052, 0.008828126, 0.008171955, 0.008292454, 0.008541321, 0.008941861, 0.008382217,
                            0.0093171345, 0.00887154, 0.00924207, 0.008431922, 0.010134701, 0.009067293, 0.008715886,
                            0.009046094, 0.008905834, 0.009304215, 0.0084734885, 0.009113058, 0.008725466, 0.00891771,
                            0.009029874, 0.009075413, 0.008714477, 0.008642749, 0.009606169, 0.009153075, 0.008541235,
                            0.008757088, 0.008895436, 0.008685254, 0.009073456, 0.009021303, 0.0091500655, 0.008602242,
                            0.008729225, 0.008651532, 0.00889056, 0.008539058, 0.008853196, 0.008528904, 0.009081975,
                            0.008880149, 0.009011798, 0.008814402, 0.008310771, 0.0086474195, 0.0088569755, 0.008831825,
                            0.008448529, 0.009218237, 0.008776604, 0.008426106, 0.009735886, 0.008201035, 0.0084616,
                            0.009321873, 0.008009138, 0.008612209, 0.008708076, 0.009205309, 0.009468417, 0.008822096,
                            0.009634672, 0.009128048, 0.009276119, 0.008234107, 0.008126152, 0.008283921, 0.008857169,
                            0.00962041, 0.009142644, 0.008910086, 0.008707047, 0.009858493, 0.008486905, 0.008855863,
                            0.0085881585, 0.009101725, 0.008991773, 0.008670437, 0.009238689, 0.008999442, 0.0087768845,
                            0.008720844, 0.008967684, 0.008977434, 0.008293324, 0.009205814, 0.00939858, 0.008000079,
                            0.009096718, 0.009747102, 0.008235297, 0.008520133, 0.008829131, 0.008627394, 0.008885311,
                            0.008586436, 0.008513904, 0.009096116, 0.00846825, 0.008941086, 0.009500336, 0.008395509,
                            0.0090844305, 0.008948396, 0.009351325, 0.008645779, 0.008186368, 0.009221275, 0.0087996945,
                            0.007919371],
                           [0.008863348, 0.008990947, 0.008665387, 0.008738659, 0.008795575, 0.0088580595, 0.008690404,
                            0.00893555, 0.008795486, 0.009135452, 0.00867635, 0.009216128, 0.008762823, 0.008837637,
                            0.008773661, 0.008889659, 0.00897715, 0.008758745, 0.008812604, 0.008703191, 0.009076109,
                            0.009168529, 0.008894875, 0.008803903, 0.008823888, 0.0090954555, 0.008903092, 0.008849877,
                            0.008811562, 0.008768541, 0.008788469, 0.008823974, 0.008926489, 0.009004612, 0.008925563,
                            0.008869472, 0.008670877, 0.0089344, 0.008665922, 0.008771081, 0.008865634, 0.008838625,
                            0.008891488, 0.008868792, 0.008716473, 0.008714524, 0.008965747, 0.008925145, 0.008503174,
                            0.008657943, 0.008897216, 0.008951141, 0.008816177, 0.009063996, 0.0084034335, 0.0088644065,
                            0.008865652, 0.008744331, 0.008692555, 0.0088317795, 0.0089199385, 0.008973046, 0.008862132,
                            0.00932832, 0.00900139, 0.008982577, 0.008714671, 0.00865364, 0.008616712, 0.008831579,
                            0.008901949, 0.009169578, 0.009096002, 0.00880188, 0.009072054, 0.008692439, 0.0089836605,
                            0.008842373, 0.008753372, 0.008847853, 0.008797723, 0.009024471, 0.008734624, 0.008726025,
                            0.008803052, 0.008897076, 0.009003864, 0.008712522, 0.008844816, 0.008648713, 0.008749454,
                            0.008825217, 0.009085626, 0.00866521, 0.00862834, 0.008855059, 0.008822056, 0.008927014,
                            0.0090179695, 0.008871902, 0.008933072, 0.008876315, 0.0090619, 0.008783536, 0.008819637,
                            0.008912323, 0.008794113, 0.00904266, 0.0086641675, 0.008623698, 0.008907507, 0.008743409,
                            0.008620076]])

    print(top_n_accuracy(y_scores_1, y_true_1, 3))
    # print(average_precision_score(y_true_1, y_scores_1))

    eval_metrics = EvaluationMetrics(113, 113)
    eval_metrics.accumulate(y_scores_1, y_true_1, [0, 0])
    print(eval_metrics.get())

    eval_metrics = EvaluationMetrics(113, 20)
    eval_metrics.accumulate(y_scores_1, y_true_1, [0, 0])
    print(eval_metrics.get())


def load_id_name_map(file_path, level_num=2):
    previous_id = -1
    id_name_map = {}
    for i in range(level_num):
        id_name_map[i] = {}
    current_level = 0

    for line in open(file_path, 'r'):
        if line.strip() == '':
            continue
        elems = line.strip().split("\t")
        if len(elems) < 2:
            print('load_id_name_map error:{}'.format(line))

        id, name = elems
        id = int(id)
        if id < previous_id:
            current_level += 1
        id_name_map[current_level][id] = name
        previous_id = id
    return id_name_map


if __name__ == '__main__':
    test_map()
    print(one_hot_encode([1, 3, 4, 6, 9, 14, 1], 20))
    id_name_map = load_id_name_map('../new_struct/train_data_format/video_label_id_name_map_20200609.txt', level_num=3)

    for i in range(2):
        id_name_map_tmp = id_name_map[i]
        print('level {}'.format(i))
        for j in range(len(id_name_map_tmp)):
            print('{}\t{}'.format(j, id_name_map_tmp[j]))

