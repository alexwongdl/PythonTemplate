"""
Created by Alex Wang on 20200513
"""


def badcase_set_get(groundtruth_list, predict_list, video_id_list, id_name_map):
    badcase_set_map = {}

    for i in range(len(groundtruth_list)):
        groundtruth = groundtruth_list[i]
        predict_label = predict_list[i]

        if groundtruth != predict_label:
            pair = "{}_{}".format(id_name_map[groundtruth], id_name_map[predict_label])
            if pair not in badcase_set_map:
                badcase_set_map[pair] = set()
            badcase_set_map[pair].add(video_id_list[i])
    return badcase_set_map


def get_domain_split_dict():
    domain_dict = dict()
    for i in range(0, 17):
        domain_dict[i] = 0
    for i in range(17, 38):
        domain_dict[i] = 1
    for i in range(38, 58):
        domain_dict[i] = 2
    for i in range(58, 72):
        domain_dict[i] = 3
    for i in range(72, 89):
        domain_dict[i] = 4
    return domain_dict

def badcase_set_get_yw(groundtruth_list, predict_list, confidence_list, video_id_list, ):
    badcase_set_map = {}

    for i in range(len(groundtruth_list)):
        groundtruth = groundtruth_list[i]
        predict_label = predict_list[i]
        confidence = confidence_list[i]

        if groundtruth != predict_label:
            pair = "{}_{}".format(groundtruth, predict_label)
            gt_score = confidence[groundtruth]
            pred_score = confidence[predict_label]

            if pair not in badcase_set_map:
                badcase_set_map[pair] = list()
            badcase_set_map[pair].append((video_id_list[i], gt_score, pred_score))
    return badcase_set_map

if __name__ == '__main__':
    pass
