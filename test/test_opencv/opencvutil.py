"""
Created by Alex Wang
On 2018-07-18
"""

def rectangle(image, x, y, w, h, color, thickness=2, label=None):
    """Draw a rectangle.

    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.
    """
    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), thickness)

def rectangle_bottom(image, x, y, w, h, color, thickness=2, label=None):
    """Draw a rectangle.

    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.

    """
    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

        # center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        # pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        # right-bottom cornor
        center = pt2[0] - 5 - text_size[0][0], pt2[1] + 5 - text_size[0][1]
        pt1 = pt2[0] - 10 - text_size[0][0], pt2[1] - 10 - text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), thickness)