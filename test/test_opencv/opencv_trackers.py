"""
Created by Alex Wang
On 2018-07-20

reference:https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
            https://github.com/abewley/sort
Accuracy and speed of KCF are both better than MIL and it reports tracking failure better than BOOSTING and MIL. If you are using OpenCV 3.1 and above, I recommend using this for most applications.

class TrackerKCF(Tracker)
 |  Method resolution order:
 |      TrackerKCF
 |      Tracker
 |      Algorithm
 |      __builtin__.object
 |
 |  Methods defined here:
 |
 |  __repr__(...)
 |      x.__repr__() <==> repr(x)
 |
 |  create(...)
 |      create() -> retval
 |      .   @brief Constructor
 |      .   @param parameters KCF parameters TrackerKCF::Params
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __new__ = <built-in method __new__ of type object>
 |      T.__new__(S, ...) -> a new object with type S, a subtype of T
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from Tracker:
 |
 |  init(...)
 |      init(image, boundingBox) -> retval
 |      .   @brief Initialize the tracker with a know bounding box that surrounding the target
 |      .   @param image The initial frame
 |      .   @param boundingBox The initial boundig box
 |      .
 |      .   @return True if initialization went succesfully, false otherwise
 |
 |  update(...)
 |      update(image) -> retval, boundingBox
 |      .   @brief Update the tracker, find the new most likely bounding box for the target
 |      .   @param image The current frame
 |      .   @param boundingBox The boundig box that represent the new target location, if true was returned, not modified otherwise
 |      .
 |      .   @return True means that target was located and false means that tracker cannot locate target in current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed missing from the frame (say, out of sight)
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from Algorithm:
 |
 |  clear(...)
 |      clear() -> None
 |      .   @brief Clears the algorithm state
 |
 |  empty(...)
 |      empty() -> retval
 |      .   @brief Returns true if the Algorithm is empty (e.g. in the very beginning or after unsuccessful read
 |
 |  getDefaultName(...)
 |      getDefaultName() -> retval
 |      .   Returns the algorithm string identifier.
 |      .   This string is used as top level xml/yml node tag when the object is saved to a file or string.
 |
 |  read(...)
 |      read(fn) -> None
 |      .   @brief Reads algorithm parameters from a file storage
 |
 |  save(...)
 |      save(filename) -> None
 |      .   Saves the algorithm to a file.
 |      .   In order to make this method work, the derived class must implement Algorithm::write(FileStorage& fs).
 |
 |  write(...)
 |      write(fs[, name]) -> None
 |      .   @brief simplified API for language bindings
 |      .   * @overload

"""
import numpy as np
import cv2
from sklearn.utils.linear_assignment_ import linear_assignment

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE']


def creat_tracker(tracker_type):
    """
    :param tracker_type:
    :return:
    """
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    else:
        print('tracker method {} not find, create KCF tracker instead.'.format(tracker_type))
        tracker = cv2.TrackerKCF_create()
    return tracker


def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou_value = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                      + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    # iou_a = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]))
    # iou_b = wh / ((bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]))
    return iou_value


class Trackers(object):
    def __init__(self, tracker_type='KCF'):
        self.tracker_type = tracker_type
        self.tracker_list = []  # {'tracker':xxx, 'count':xxx, 'box':xxx, 'face_box':xxx}
        self.count = 0

    def update_and_detect(self, frame, dets, iou_threashold=0.3):
        """
        :param frame:ok, frame = video.read()
        :param dets: boxes predicted by object detection algorithm
                    every elem must be [x_min, y_min, x_max, y_max, score]
        :return:
        """
        new_tracker_list = []

        for tracker_info in self.tracker_list:
            tracker = tracker_info['tracker']
            ok, bbox = tracker.update(frame)
            if ok:
                tracker_info['box'] = bbox
                new_tracker_list.append(tracker_info)

        unmatched_dets = []
        # calculate iou matrix
        iou_matrix = np.zeros((len(dets), len(new_tracker_list)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, tracker_info in enumerate(new_tracker_list):
                tracker_box = tracker_info['box']
                iou_matrix[d, t] = iou(det, tracker_box)

        matched_indices = linear_assignment(-iou_matrix)
        # match face and tracker
        for d, det in enumerate(dets):
            if (d not in matched_indices[:, 0]):
                unmatched_dets.append(det)
            else:
                for m in matched_indices:
                    if m[0] == d:
                        new_tracker_list[m[1]]['face_box'] = tuple(det[0:4])
                        if iou_matrix[d, m[1]] < 0.5:  # fix tracker box
                            print('fix tracker box, reinitialize')
                            new_tracker = creat_tracker(self.tracker_type)
                            new_tracker.init(frame, tuple(det[0:4]))
                            new_tracker_list[m[1]]['tracker'] = new_tracker
                            new_tracker_list[m[1]]['box'] = tuple(det[0:4])

        # create new trackers for unmatched box
        for det in unmatched_dets:
            new_tracker = creat_tracker(self.tracker_type)
            # init_det = (det[0], det[1], det[2], det[3])
            init_det = tuple(det[0:4])
            ok = new_tracker.init(frame, init_det)
            if ok:
                self.count += 1
                new_tracker_list.append({'tracker': new_tracker, 'count': self.count,
                                         'box': init_det, 'score': det[4],
                                         'face_box': init_det})
            else:
                print('initialize tracker error:{}'.format(det))

        self.tracker_list = new_tracker_list
        return self.tracker_list

    def update(self, frame, iou_threashold=0.3):
        """
        :param frame:ok, frame = video.read()
        :return:
        """
        new_tracker_list = []

        for tracker_info in self.tracker_list:
            tracker = tracker_info['tracker']
            ok, bbox = tracker.update(frame)
            if ok:
                tracker_info['box'] = bbox
                new_tracker_list.append(tracker_info)

        self.tracker_list = new_tracker_list
        return self.tracker_list
