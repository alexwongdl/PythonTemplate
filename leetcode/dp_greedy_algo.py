"""
 @author: AlexWang
 @date: 2021/8/29 4:00 PM

 动态规划和贪心算法练习
"""

"""
********** 俄罗斯套娃信封问题
涉及的问题:
1. 需要转化成最长递增子序列;
2. 最长递增子序列的解法:O(n2), O(nlogn)

给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。

当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

注意：不允许旋转信封。


示例 1：

输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出：3
解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
示例 2：

输入：envelopes = [[1,1],[1,1],[1,1]]
输出：1

提示：
1 <= envelopes.length <= 5000
envelopes[i].length == 2
1 <= wi, hi <= 104  可以知道最大长度是206

正确解法:动态规划
第一个维度按照递增序列排序
第二个维度上按照递减序列排序
然后在第二个维度上求最长递增序列

（1）解法1：只保存最大元素最小的递增子序列
以序列{6，7，8，9，10，1，2，3，4，5，6}为例，
程序开始时，最长递增序列长度为1（每个元素都是一个长度为1的递增序列），当处理第2个元素时发现7比最长递增序列6的最大元素还要大，所以将6，7结合生成长度为2的递增序列，说明已经发现了长度为2的递增序列，依次处理，到第5个元素(10)，这一过程中B数组的变化过程是
    6
    6，7
    6，7，8
    6，7，8，9
    6，7，8，9，10
开始处理第6个元素是1，查找比1大的最小元素，发现是长度为1的子序列的最大元素6，说明1是最大元素更小的长度为1的递增序列，用1替换6，形成新数组1，7，8，9，10。然后查找比第7个元素(2)大的最小元素，发现7，说明存在长度为2的序列，其末元素2，比7更小，用2替换7，依次执行，直到所有元素处理完毕，生成新的数组1，2，3，4，5，最后将6加入B数组，形成长度为6的最长递增子序列.
这一过程中，B数组的变化过程是

    1，7，8，9，10
    1，2，8，9，10
    1，2，3，9，10
    1，2，3，4，10
    1，2，3，4，5
    1，2，3，4，5，6
当处理第10个元素(5)时，传统算法需要查看9个元素(6,7,8,9,10,1,2,3,4)，而改进算法只需要用二分查找数组B中的两个元素(3, 4)，可见改进算法还是很阴霸的。
"""
import numpy as np
import queue
import time

np.set_printoptions(precision=3, suppress=True)


def numpy_print_format(arr):
    height, width = arr.shape
    for i in range(height):
        print(", ".join(map(str, arr[i, :].tolist())))


def envelopes(input_list):
    """
    只要list长度大于1,最小结果就是1

    X 1. 计算item之间的大小关系;
    X 2. 加入第k项之前先计算k-1项中的关系;
    转化成LCS问题
    :param input_list:
    :return:
    """
    import bisect
    import time
    time1 = time.time()
    item_num = len(input_list)
    if item_num <= 0:
        return 0

    if item_num == 1:
        return 1

    sorted_input = sorted(input_list, key=lambda x: (x[0], -x[1]))
    for item in sorted_input:
        print(item)
    sorted_input = [item[-1] for item in sorted_input]
    print(sorted_input)
    time2 = time.time()

    inc_arr = [sorted_input[0]]
    for i in range(1, item_num):
        item_i = sorted_input[i]
        index = bisect.bisect(inc_arr, item_i)
        if index >= len(inc_arr):  # 最后面
            if item_i > inc_arr[-1]:
                inc_arr.append(item_i)
        elif index == 0:  # 替换第一个
            inc_arr[index] = item_i
        elif inc_arr[index] > item_i > inc_arr[index - 1]:
            inc_arr[index] = item_i

    print("result:{}".format(inc_arr))
    time3 = time.time()
    print("time2-time1:{}, time_3 - time_2:{}".format(time2 - time1, time3 - time2))

    return len(inc_arr)
    # 错误方法:用动态规划方法计算,会很慢
    # dp_arr = np.eye(item_num, item_num, dtype=np.int32)
    # print(dp_arr)
    # for i in range(1, item_num):
    #     item_i = sorted_input[i]
    #     for j in range(i):  # 遍历在他前面的数据, j 在 i之前
    #         item_j = sorted_input[j]
    #         if item_i > item_j:
    #             dp_arr[i][j] = np.max(dp_arr[j, :]) + 1


def test_envelopes():
    result = envelopes(
        [[15, 8], [2, 20], [2, 14], [4, 17], [8, 19], [8, 9], [5, 7], [11, 19], [8, 11], [13, 11], [2, 13], [11, 19],
         [8, 11], [13, 11], [2, 13], [11, 19], [16, 1], [18, 13], [14, 17], [18, 19]])
    print("result:{}".format(result))

    result = envelopes([[5, 4], [6, 4], [6, 7], [2, 3]])
    print("result:{}".format(result))  # 3

    result = envelopes([[4, 5], [4, 6], [6, 7], [2, 3], [1, 1]])
    print("result:{}".format(result))  # 4

    result = envelopes(
        [[33, 23], [43, 3], [10, 43], [42, 29], [5, 34], [41, 14], [40, 14], [5, 37], [25, 6], [7, 2], [34, 47],
         [46, 40], [7, 6], [41, 40], [16, 36], [41, 30], [18, 31], [21, 42], [10, 5], [40, 29], [8, 12], [36, 13],
         [47, 8], [3, 8], [38, 18], [2, 48], [15, 29], [17, 4], [30, 47], [32, 36], [8, 49], [11, 41], [34, 22],
         [1, 48], [4, 1], [42, 35], [33, 9], [3, 16], [29, 30], [18, 13], [30, 11], [6, 43], [4, 16], [32, 15],
         [11, 50], [13, 21], [40, 28], [36, 21], [39, 26], [32, 31], [25, 8], [40, 28], [30, 22], [20, 42], [43, 18],
         [19, 40], [45, 9], [50, 12], [50, 38], [41, 27], [47, 14], [8, 39], [40, 45], [38, 34], [33, 5], [14, 37],
         [35, 15], [7, 6], [38, 47], [43, 46], [30, 29], [36, 49], [4, 18], [28, 47], [50, 31], [10, 34], [40, 31]])
    print("result:{}".format(result))  # 10

    result = envelopes([[1, 1], [1, 1], [1, 1]])
    print("result:{}".format(result))  # 10

    result = envelopes(
        [[856, 533], [583, 772], [980, 524], [203, 666], [987, 151], [274, 802], [982, 85], [359, 160], [58, 823],
         [512, 381], [796, 655], [341, 427], [145, 114], [76, 306], [760, 929], [836, 751], [922, 678], [128, 317],
         [185, 953], [115, 845], [829, 991], [93, 694], [317, 434], [818, 571], [352, 638], [926, 780], [819, 995],
         [54, 69], [191, 392], [377, 180], [669, 952], [588, 920], [335, 316], [48, 769], [188, 661], [916, 933],
         [674, 308], [356, 556], [350, 249], [686, 851], [600, 178], [849, 439], [597, 181], [80, 382], [647, 105],
         [4, 836], [901, 907], [595, 347], [214, 335], [956, 382], [77, 979], [489, 365], [80, 220], [859, 270],
         [676, 665], [636, 46], [906, 457], [522, 769], [2, 758], [206, 586], [444, 904], [912, 370], [64, 871],
         [59, 409], [599, 238], [437, 58], [309, 767], [258, 440], [922, 369], [848, 650], [478, 76], [84, 704],
         [314, 207], [138, 823], [994, 764], [604, 595], [537, 876], [877, 253], [945, 185], [623, 497], [968, 633],
         [172, 705], [577, 388], [819, 763], [409, 905], [275, 532], [729, 593], [547, 226], [445, 495], [398, 544],
         [243, 500], [308, 24], [652, 452], [93, 885], [75, 884], [243, 113], [600, 555], [756, 596], [892, 762],
         [402, 653], [916, 975], [770, 220], [455, 579], [889, 68], [306, 899], [567, 290], [809, 653], [92, 329],
         [370, 861], [632, 754], [321, 689], [190, 812], [88, 701], [79, 310], [917, 91], [751, 480], [750, 39],
         [781, 978], [778, 912], [946, 559], [529, 621], [55, 295], [473, 748], [646, 854], [930, 913], [116, 734],
         [647, 812], [426, 172], [122, 14], [522, 843], [88, 308], [719, 602], [712, 928], [303, 890], [973, 886],
         [276, 354], [660, 720], [708, 387], [776, 605], [653, 815], [448, 285], [549, 959], [139, 365], [74, 952],
         [372, 424], [642, 504], [361, 901], [620, 612], [313, 301], [397, 225], [446, 716], [17, 361], [160, 812],
         [171, 529], [180, 482], [454, 600], [228, 872], [204, 492], [607, 889], [86, 79], [494, 78], [442, 404],
         [462, 127], [935, 402], [509, 649], [458, 941], [219, 444], [306, 57], [674, 617], [79, 652], [73, 735],
         [900, 756], [649, 294], [982, 754], [521, 439], [356, 265], [240, 533], [865, 44], [744, 379], [97, 454],
         [65, 480], [544, 191], [18, 191], [503, 38], [696, 658], [61, 884], [793, 984], [383, 364], [280, 467],
         [888, 662], [133, 643], [365, 512], [610, 975], [98, 584], [40, 177], [548, 102], [80, 98], [986, 951],
         [264, 258], [583, 734], [353, 322], [427, 551], [80, 660], [273, 609], [980, 871], [739, 802], [366, 836],
         [55, 509], [889, 720], [857, 661], [48, 489], [119, 26], [31, 180], [472, 673], [960, 951], [383, 500],
         [928, 351], [848, 705], [969, 766], [311, 714], [861, 230], [34, 596], [38, 642], [1, 955], [698, 846],
         [784, 791], [760, 344], [677, 239], [969, 191], [539, 644], [470, 418], [289, 357], [269, 446], [668, 245],
         [293, 719], [937, 103], [575, 297], [874, 656], [714, 257], [934, 396], [109, 904], [89, 635], [374, 545],
         [316, 587], [158, 121], [901, 969], [284, 564], [666, 568], [993, 409], [370, 637], [443, 694], [576, 160],
         [262, 357], [590, 729], [194, 976], [743, 376], [348, 80], [669, 527], [338, 953], [236, 785], [144, 460],
         [438, 457], [517, 951], [545, 647], [158, 556], [905, 591], [793, 609], [571, 643], [9, 850], [581, 490],
         [804, 394], [635, 483], [457, 30], [42, 621], [65, 137], [424, 864], [536, 455], [59, 492], [645, 734],
         [892, 571], [762, 593], [608, 384], [558, 257], [692, 420], [973, 203], [531, 51], [349, 861], [804, 649],
         [3, 611], [6, 468], [298, 568], [651, 767], [251, 142], [173, 974], [117, 728], [326, 562], [894, 288],
         [814, 555], [420, 771], [20, 775], [445, 247], [243, 592], [186, 173], [101, 800], [590, 876], [515, 534],
         [73, 540], [333, 215], [902, 394], [640, 787], [596, 298], [984, 712], [307, 378], [540, 646], [473, 743],
         [340, 387], [756, 217], [139, 493], [9, 742], [195, 25], [763, 823], [451, 693], [24, 298], [645, 595],
         [224, 770], [976, 41], [832, 78], [599, 705], [487, 734], [818, 134], [225, 431], [380, 566], [395, 680],
         [294, 320], [915, 201], [553, 480], [318, 42], [627, 94], [164, 959], [92, 715], [588, 689], [734, 983],
         [976, 334], [846, 573], [676, 521], [449, 69], [745, 810], [961, 722], [416, 409], [135, 406], [234, 357],
         [873, 61], [20, 521], [525, 31], [659, 688], [424, 554], [203, 315], [16, 240], [288, 273], [281, 623],
         [651, 659], [939, 32], [732, 373], [778, 728], [340, 432], [335, 80], [33, 835], [835, 651], [317, 156],
         [284, 119], [543, 159], [719, 820], [961, 424], [88, 178], [621, 146], [594, 649], [659, 433], [527, 441],
         [118, 160], [92, 217], [489, 38], [18, 359], [833, 136], [470, 897], [106, 123], [831, 674], [181, 191],
         [892, 780], [377, 779], [608, 618], [618, 423], [180, 323], [390, 803], [562, 412], [107, 905], [902, 281],
         [718, 540], [16, 966], [678, 455], [597, 135], [840, 7], [886, 45], [719, 937], [890, 173]])
    print("result:{}".format(result))
    # print("result:{}".format(result))

    result = envelopes([[1, 1], [1, 1], [1, 1]])
    print("result:{}".format(result))


"""
***** 买卖股票的最佳时机
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

示例 1：
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
     
示例 2：
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
"""


def maxProfit(prices: list[int]) -> int:
    if len(prices) <= 1:
        return 0

    candidate_pairs = []
    min_out_of_pair = None
    smallest_a = None
    a_set = set()

    for i in range(len(prices)):

        small_than_all_a = True
        if len(candidate_pairs) <= 0:
            a = prices[i]
            if i + 1 <= len(prices) - 1:
                b = prices[i + 1]
                if a >= b:
                    continue
                candidate_pairs.append([a, b])
                smallest_a = a
                a_set.add(a)

        else:
            c = prices[i]
            if c in a_set:
                continue

            if c < smallest_a:
                if min_out_of_pair is None:
                    min_out_of_pair = c
                elif c < min_out_of_pair:
                    min_out_of_pair = c
            else:
                for j, item in enumerate(candidate_pairs):
                    a, b = item

                    # c a c b c 三种位置情况
                    if a < c < b:
                        continue
                    elif c > b and c > a:
                        candidate_pairs[j] = [a, c]
                        smallest_a = a

            if min_out_of_pair is not None and c > min_out_of_pair:
                candidate_pairs.append([min_out_of_pair, c])
                min_out_of_pair = None
            # print(len(candidate_pairs))

    print(candidate_pairs)
    max_bonus = 0
    for pair in candidate_pairs:
        a, b = pair
        bonus = b - a
        if bonus > max_bonus:
            max_bonus = bonus
    return max_bonus


def test_max_profit():
    prices = [7, 1, 5, 3, 6, 4]
    maxProfit(prices)

    prices = [7, 6, 4, 3, 1]
    maxProfit(prices)

    print("------------------------")
    prices = [2, 1, 2, 1, 0, 1, 2]
    maxProfit(prices)

    print("------------------------")
    prices = [5, 2, 5, 6, 8, 2, 3, 0, 1, 8, 5, 2, 1]
    prices = [4, 7, 2, 6, 4, 3, 8, 2, 7, 5]

    with open("dp_greedy.txt", 'r') as reader:
        line = reader.read()
        prices = list(map(int, line.split(",")))

    maxProfit(prices)


"""
买卖股票的最佳时机 II
给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。
在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。
返回 你能获得的 最大 利润 。
"""


def maxProfit2(prices: list[int]) -> int:
    bonus = 0
    candidate_pair = []

    for price in prices:
        if len(candidate_pair) == 0:
            candidate_pair.append(price)

        if len(candidate_pair) == 1:
            if price <= candidate_pair[0]:
                candidate_pair[0] = price
            else:
                candidate_pair.append(price)

        if len(candidate_pair) == 2:
            if price >= candidate_pair[1]:
                candidate_pair[1] = price
            else:
                # 变现
                bonus += candidate_pair[1] - candidate_pair[0]
                candidate_pair = [price]

    if len(candidate_pair) == 2:
        bonus += candidate_pair[1] - candidate_pair[0]

    print(bonus)
    return bonus


def test_max_profit2():
    prices = [7, 1, 5, 3, 6, 4]
    maxProfit2(prices)

    prices = [1, 2, 3, 4, 5]
    maxProfit2(prices)

    prices = [7, 6, 4, 3, 1]
    maxProfit2(prices)


"""
******** 最大正方形
在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：4

输入：matrix = [["0","1"],["1","0"]]
输出：1

输入：matrix = [["0"]]
输出：0

解题思路：动态规划
在正方形的右下角方格记录当前正方形的坐标[x1, x2, y1, y2]
每个位置都只在左上角的基础上扩展
"""


def maximalSquare(matrix: list[list[str]]) -> int:
    rows = len(matrix)

    if rows <= 0:
        return 0
    cols = len(matrix[0])
    if cols <= 0:
        return 0

    pos_matrix = []
    for i in range(rows):
        rows_value = []
        for j in range(cols):
            rows_value.append([])
        pos_matrix.append(rows_value)

    max_pos = None
    max_value = 0

    for i in range(rows):
        for j in range(cols):
            i_before = i - 1
            j_before = j - 1
            if matrix[i][j] == "0":
                continue

            if i_before < 0 or j_before < 0 or pos_matrix[i_before][j_before] == []:
                # 只看当前位置，不看左上角
                pos_matrix[i][j] = [i - 1, i, j - 1, j]
                if max_value < 1:
                    max_value = 1
                    max_pos = [i - 1, i, j - 1, j]

            else:
                [x1, x2, y1, y2] = pos_matrix[i_before][j_before]
                all_ones = True
                rows_max = 0
                cols_max = 0

                for i_tmp in range(x2, x1, -1):
                    if matrix[i_tmp][j] == "0":
                        break
                    else:
                        rows_max += 1

                for j_tmp in range(y2, y1, -1):
                    if matrix[i][j_tmp] == "0":
                        break
                    else:
                        cols_max += 1

                row_col_max = min(rows_max, cols_max)
                pos_matrix[i][j] = [i - 1 - row_col_max, i, j - 1 - row_col_max, j]

                area = (row_col_max + 1) ** 2
                if area > max_value:
                    max_value = area
                    max_pos = [i - 1 - row_col_max, i, j - 1 - row_col_max, j]

                # 错误case：matrix = [["0", "0", "0", "1"],
                #                    ["1", "1", "0", "1"],
                #                    ["1", "1", "1", "1"],
                #                    ["0", "1", "1", "1"],
                #                    ["0", "1", "1", "1"]]
                # for i_tmp in range(x1 + 1, x2 + 1):
                #     # [ x1+1 -- x2, j]
                #     if matrix[i_tmp][j] == "0":
                #         all_ones = False
                #         break
                # if all_ones:
                #     for j_tmp in range(y1 + 1, y2 + 1):
                #         # [i, y1+1 -- y2]
                #         if matrix[i][j_tmp] == "0":
                #             all_ones = False
                #             break
                #
                # if all_ones:
                #     area = (j - y1) ** 2
                #     if area > max_value:
                #         max_value = area
                #         max_pos = [x1, i, y1, j]
                #         pos_matrix[i][j] = [x1, i, y1, j]

    print(pos_matrix)
    print(max_value)
    return max_value


def test_maximal_square():
    matrix = [["1", "0", "1", "0", "0"], ["1", "0", "1", "1", "1"], ["1", "1", "1", "1", "1"],
              ["1", "0", "0", "1", "0"]]
    maximalSquare(matrix)

    matrix = [["0", "1"], ["1", "0"]]
    maximalSquare(matrix)

    matrix = [["0"]]
    maximalSquare(matrix)

    matrix = [["0", "0", "0", "1"],
              ["1", "1", "0", "1"],
              ["1", "1", "1", "1"],
              ["0", "1", "1", "1"],
              ["0", "1", "1", "1"]]
    maximalSquare(matrix)


"""
****** 最大子数组和
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
子数组 是数组中的一个连续部分。

示例 1：
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

示例 2：
输入：nums = [1]
输出：1

示例 3：
输入：nums = [5,4,-1,7,8]
输出：23

解题思路：
当前序列和为正数的话就可以一直往下加，否则就丢弃当前序列
"""


def maxSubArray(nums: list[int]) -> int:
    max_value = -10000000
    cur_sum = 0

    for num in nums:
        cur_sum += num
        if cur_sum > max_value:
            max_value = cur_sum

        if cur_sum <= 0:
            cur_sum = 0

    print(max_value)
    return max_value


def test_max_sub_array():
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    maxSubArray(nums)

    nums = [1]
    maxSubArray(nums)

    nums = [5, 4, -1, 7, 8]
    maxSubArray(nums)


"""
****** 三角形最小路径和
给定一个三角形 triangle ，找出自顶向下的最小路径和。
每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

示例 1：
输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
输出：11
解释：如下面简图所示：自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。-104 <= triangle[i][j] <= 104
   2
  3 4
 6 5 7
4 1 8 3

解题思路：动态规划
构造与三角形形状相同的memory数组
每一行都比上一行多一个数(i，i+1)

triangle[0].length == 1
triangle[i].length == triangle[i - 1].length + 1

解题思路：动态规划
构造与三角形形状相同的memory数组
每一行都比上一行多一个数(i，i+1)
"""


def minimumTotal(triangle: list[list[int]]) -> int:
    min_memory = []
    for i in range(len(triangle)):
        if i == 0:
            min_memory.append(triangle[i])

        else:
            cur_mem = []
            prev_mem = min_memory[i - 1]
            len_men = len(prev_mem)

            cur_arr = triangle[i]
            for k in range(len(cur_arr)):
                cur_item = cur_arr[k]
                # 看 k-1 和k 两个menory
                if k - 1 < 0:
                    cur_mem.append(prev_mem[k] + cur_item)
                elif k > len_men - 1:
                    cur_mem.append(prev_mem[k - 1] + cur_item)
                else:
                    cur_mem.append(min(prev_mem[k] + cur_item, prev_mem[k - 1] + cur_item))
            min_memory.append(cur_mem)

    print(min(min_memory[-1]))
    return min(min_memory[-1])


def test_minimum_total():
    triangle = [[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]
    minimumTotal(triangle)


"""
一次编辑
字符串有三种编辑操作:插入一个英文字符、删除一个英文字符或者替换一个英文字符。 给定两个字符串，编写一个函数判定它们是否只需要一次(或者零次)编辑。

思路：动态规划
编辑距离的简化版本，动态规划计算编辑距离  
    f(i, j) = min{
                   f(i-1, j-1) + I(i, j),
                   f(i-1, j) + 1,
                   f(i, j-1) +1
                  }
"""
import math
import numpy as np


def oneEditAway(first: str, second: str) -> bool:
    m = len(first)
    n = len(second)
    if math.fabs(m - n) >= 2:  # 字符串长度不能超过2
        return False
    if len(first) <= 1 and len(second) <= 1:
        return True

    memory = np.zeros(shape=(m, n), dtype=np.int32)
    for i in range(m):
        if first[i] == second[0]:
            memory[i, 0] = memory[i - 1, 0]  # 注意第一行第一列的初始化
        else:
            memory[i, 0] = memory[i - 1, 0] + 1

    for i in range(n):
        if first[0] == second[i]:
            memory[0, i] = memory[0, i - 1]
        else:
            memory[0, i] = memory[0, i - 1] + 1

    for i in range(1, m):
        for j in range(1, n):
            if first[i] == second[j]:
                memory[i, j] = min(memory[i - 1, j - 1], memory[i - 1, j] + 1, memory[i, j - 1] + 1)
            else:
                memory[i, j] = min(memory[i - 1, j - 1] + 1, memory[i - 1, j] + 1, memory[i, j - 1] + 1)
    print(memory)
    if memory[-1, -1] <= 1:
        return True
    else:
        return False


def test_one_edit_away():
    first = "pale"
    second = "ple"
    oneEditAway(first, second)

    first = "ab"
    second = "bc"
    oneEditAway(first, second)

    first = "a"
    second = "ab"
    oneEditAway(first, second)


"""
***** 分割等和子集
给你一个 只包含正整数 的 非空 数组 nums 。判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
示例 1：
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。

示例 2：
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
1 <= nums.length <= 200
1 <= nums[i] <= 100

解题思路：
      暴力算法：O(2^n)复杂度，每个数据判断是在哪个子集
      算法优化：每个子集的和是sum(nums) / 2，从最大的数开始，
            如果和的大小超过sum(nums) / 2，则这个数应该放到另一个子集
            转化为0-1背包问题，用动态规划方法求解
"""


def canPartition(nums: list[int]) -> bool:
    if len(nums) <= 1:
        return False

    sum_nums = sum(nums)
    if sum_nums % 2 == 1:  # 必须能被2整除
        return False

    nums = sorted(nums)
    target_sum = int(sum_nums / 2)
    # 从数组中找出若干个数，且大小等于target_sum，递归求解，
    # 如果能找到，那就是存在，优先从最大的数开始找，查找次数会比较少
    print("find_sum(nums, target_sum)", nums, target_sum)
    # succeed, result = find_sum(nums, target_sum)
    # print(succeed, result)
    # 动态规划方法求解
    result = [[False] * (target_sum + 1) for _ in range(len(nums))]

    for i in range(len(nums)):
        j = 0
        result[i][j] = True

    num_0 = nums[0]
    if num_0 <= target_sum:
        result[0][num_0] = True

    for i in range(1, len(nums)):
        num = nums[i]

        if num < target_sum:
            result[i][num] = True

        for j in range(target_sum + 1):
            if j >= num:
                result[i][j] = result[i - 1][j] | result[i - 1][j - num]
            else:
                result[i][j] = result[i - 1][j]

    # print(result)

    for i in range(len(nums)):
        if result[i][-1]:
            return True

    return False


# def find_sum(arr, target_sum):
#     result = []
#     # 避免重复item尝试
#     test_item = set()
#     for i in range(len(arr)):
#         item = arr[i]
#         if item == target_sum:
#             result.append(item)
#             return True, result
#         if item > target_sum:
#             continue
#         else:
#             if i == len(arr) - 1:
#                 return False, result
#             # print("find_sum(arr[i + 1:], target_sum - item)", arr[i + 1:], target_sum - item)
#             if item in test_item:
#                 continue
#
#             test_item.add(item)
#             succeed, sub_result = find_sum(arr[i + 1:], target_sum - item)
#             if succeed:
#                 result.append(item)
#                 result.extend(sub_result)
#                 return True, result
#     if sum(result) == target_sum:
#         return True, result
#     else:
#         return False, result


def test_can_partition():
    nums = [1, 5, 11, 5]
    print(canPartition(nums))
    nums = [1, 2, 5]
    print(canPartition(nums))
    nums = [1, 2, 3, 5]
    print(canPartition(nums))
    nums = [1, 1, 2, 3, 5]
    print(canPartition(nums))
    nums = [14, 9, 8, 4, 3, 2]
    print(canPartition(nums))
    nums = [66, 90, 7, 6, 32, 16, 2, 78, 69, 88, 85, 26, 3, 9, 58, 65, 30, 96, 11, 31, 99, 49, 63, 83, 79, 97, 20, 64,
            81, 80, 25, 69, 9, 75, 23, 70, 26, 71, 25, 54, 1, 40, 41, 82, 32, 10, 26, 33, 50, 71, 5, 91, 59, 96, 9, 15,
            46, 70, 26, 32, 49, 35, 80, 21, 34, 95, 51, 66, 17, 71, 28, 88, 46, 21, 31, 71, 42, 2, 98, 96, 40, 65, 92,
            43, 68, 14, 98, 38, 13, 77, 14, 13, 60, 79, 52, 46, 9, 13, 25, 8]
    print(canPartition(nums))

    nums = [19, 33, 38, 60, 81, 49, 13, 61, 50, 73, 60, 82, 73, 29, 65, 62, 53, 29, 53, 86, 16, 83, 52, 67, 41, 53, 18,
            48, 32, 35, 51, 72, 22, 22, 76, 97, 68, 88, 64, 19, 76, 66, 45, 29, 95, 24, 95, 29, 95, 76, 65, 35, 24, 85,
            95, 87, 64, 97, 75, 88, 88, 65, 43, 79, 6, 5, 70, 51, 73, 87, 76, 68, 56, 57, 69, 77, 22, 27, 29, 12, 55,
            58, 18, 30, 66, 53, 53, 81, 94, 76, 28, 41, 77, 17, 60, 32, 62, 62, 88, 61]
    print(canPartition(nums))
    nums = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 99, 97]
    print(canPartition(nums))

    nums = [4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16,
            16, 20, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 28, 28, 28, 28, 28, 28, 28, 28, 32, 32,
            32, 32, 32, 32, 32, 32, 36, 36, 36, 36, 36, 36, 36, 36, 40, 40, 40, 40, 40, 40, 40, 40, 44, 44, 44, 44, 44,
            44, 44, 44, 48, 48, 48, 48, 48, 48, 48, 48, 52, 52, 52, 52, 52, 52, 52, 52, 56, 56, 56, 56, 56, 56, 56, 56,
            60, 60, 60, 60, 60, 60, 60, 60, 64, 64, 64, 64, 64, 64, 64, 64, 68, 68, 68, 68, 68, 68, 68, 68, 72, 72, 72,
            72, 72, 72, 72, 72, 76, 76, 76, 76, 76, 76, 76, 76, 80, 80, 80, 80, 80, 80, 80, 80, 84, 84, 84, 84, 84, 84,
            84, 84, 88, 88, 88, 88, 88, 88, 88, 88, 92, 92, 92, 92, 92, 92, 92, 92, 96, 96, 96, 96, 96, 96, 96, 96, 97,
            99]
    print(canPartition(nums))


"""
************ 划分为k个相等的子集
给定一个整数数组  nums 和一个正整数 k，找出是否有可能把这个数组分成 k 个非空子集，其总和都相等。
示例 1：
输入： nums = [4, 3, 2, 3, 5, 2, 1], k = 4
输出： True
说明： 有可能将其分成 4 个子集（5），（1,4），（2,3），（2,3）等于总和。

示例 2:
输入: nums = [1,2,3,4], k = 3
输出: false
 
提示：
1 <= k <= len(nums) <= 16
0 < nums[i] < 10000
每个元素的频率在 [1,4] 范围内

解题思路：
      暴力算法：O(k^n)复杂度，每个数据判断是在哪个子集
      算法优化：在分割2个等和子集的基础上。
              找出所有和为target的组合，并从小到大排序。
"""
import queue


def canPartitionKSubsets_wrong_answer(nums: list[int], k: int) -> bool:
    num_sum = sum(nums)
    if num_sum % k != 0:
        print(False)
        return False
    target = num_sum // k
    num_len = len(nums)
    nums = sorted(nums)
    print(nums, k, target)

    dp = np.ones(shape=(num_len, target + 1), dtype=np.uint8) * -1
    for i in range(num_len):
        dp[i][0] = i

    if nums[0] <= target:
        dp[0][nums[0]] = 0

    for i in range(1, num_len):
        num = nums[i]
        for j in range(1, target + 1):
            if dp[i - 1][j] >= 0:
                dp[i][j] = dp[i - 1][j]

            if j >= num and dp[i - 1][j - num] >= 0:
                dp[i][j] = i

    distinct_path = set()
    pathes = []
    pathes_ids = []
    used_num_id = set()
    path_ids = []
    # BFS遍历 + 剪枝
    q = queue.Queue()
    for i in range(num_len - 1, -1, -1):
        if dp[i][-1] >= 0:
            num_id = dp[i][-1]
            num = nums[num_id]
            next_cols = target - num
            cur_path = [num_id]
            print(num, next_cols, cur_path, i)
            q.put((num, next_cols, cur_path, i))

    while not q.empty():
        num, next_target, cur_path, row = q.get()
        if next_target == 0:
            new_path = True
            distinct_path.add("-".join(map(str, [nums[num_id] for num_id in cur_path])))
            if new_path:
                for num_id in cur_path:
                    used_num_id.add(num_id)
                    path_ids.append(num_id)
                pathes.append([nums[num_id] for num_id in cur_path])
                pathes_ids.append([num_id for num_id in cur_path])
        else:
            for i in range(row + 1):
                if i in used_num_id:
                    continue

                num_id = dp[i][next_target]
                cur_num = nums[num_id]
                if num_id in cur_path:
                    continue
                if num_id >= 0:
                    target_tmp = next_target - cur_num
                    path_tmp = cur_path.copy()
                    path_tmp.append(num_id)
                    q.put((cur_num, target_tmp, path_tmp, i))

    print("pathes", pathes)
    print("path_ids", path_ids)
    print("pathes_ids", pathes_ids)
    path_num = len(pathes)

    for path in distinct_path:
        print(path)

    for i in range(num_len):
        print(dp[i, :].tolist())
    print(dp)
    print(path_num, k)
    if len(pathes) == k:
        print(True)
        return True

    print(False)
    return False


def canPartitionKSubsets(nums: list[int], k: int) -> bool:
    num_sum = sum(nums)
    if num_sum % k != 0:
        print(False)
        return False
    target = num_sum // k
    num_len = len(nums)
    nums = sorted(nums, key=lambda x: -x)
    # nums = sorted(nums)
    if nums[-1] > target:
        print(False)
        return False
    if k == 1:
        return True
    print(nums, k, target)

    # 对每个元素，看放入哪个桶 O(n^k)
    # 对于每个桶，看需要哪些 O(kx2^n)
    used_id = set()
    result = k_partition_dfs(0, used_id, nums, target, k, 0, num_len)
    print(result)
    return result


def k_partition_dfs(cur_sum, used_id, nums, target, k, index, num_len):
    cur_num = nums[index]
    used_id.add(index)
    cur_sum += cur_num
    if cur_sum == target:
        k -= 1
        cur_sum = 0
    if k == 1 and index < num_len - 1:
        return True
    if cur_sum > target:
        return False

    prev = None
    for i in range(num_len):
        if i in used_id:
            continue
        used_id.add(i)
        if prev and nums[i] == prev:
            pass
        else:
            result_i = k_partition_dfs(cur_sum, used_id, nums, target, k, i, num_len)
            # print("result_i:", cur_sum, used_id, target, k, i, num_len, result_i, prev)
            if result_i:
                return True
        prev = nums[i]
        used_id.remove(i)
    return False


def test_can_partition_k_subsets():
    ###### [1, 2, 2, 4, 4, 4, 4, 6, 6, 9] 3 14 True
    nums = [4, 4, 4, 6, 1, 2, 2, 9, 4, 6]
    canPartitionKSubsets(nums, k=3)

    ###### [2, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9, 9, 9, 10, 10, 10] 5 21 True
    nums = [4, 5, 9, 3, 10, 2, 10, 7, 10, 8, 5, 9, 4, 6, 4, 9]
    canPartitionKSubsets(nums, k=5)

    nums = [4, 3, 2, 3, 5, 2, 1]  # True
    canPartitionKSubsets(nums, k=4)
    nums = [1, 2, 3, 4]  # False
    canPartitionKSubsets(nums, k=3)
    nums = [1, 1, 1, 1, 2, 2, 2, 2]  # True
    canPartitionKSubsets(nums, k=2)
    ##### [2, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9, 9, 9, 10, 10, 10] 21 # True
    nums = [4, 5, 9, 3, 10, 2, 10, 7, 10, 8, 5, 9, 4, 6, 4, 9]
    canPartitionKSubsets(nums, k=5)
    ##### [60, 202, 494, 497, 601, 625, 679, 771, 815, 883, 944, 1118, 1240, 3889, 4471, 4623] # True
    nums = [815, 625, 3889, 4471, 60, 494, 944, 1118, 4623, 497, 771, 679, 1240, 202, 601, 883]
    canPartitionKSubsets(nums, k=3)

    #### [10, 10, 10, 8, 7, 6, 6, 6, 5, 3, 3, 3, 2, 2, 2, 1] 6 14
    nums = [3, 3, 10, 2, 6, 5, 10, 6, 8, 3, 2, 1, 6, 10, 7, 2]
    canPartitionKSubsets(nums, k=6)

    # # [16, 10, 10, 5, 4, 4, 4, 4, 3] 3 20
    nums = [4, 16, 5, 3, 10, 4, 4, 4, 10]
    canPartitionKSubsets(nums, k=3)


"""
****** 鸡蛋掉落
给你 k 枚相同的鸡蛋，并可以使用一栋从第 1 层到第 n 层共有 n 层楼的建筑。
已知存在楼层 f ，满足 0 <= f <= n ，任何从 高于 f 的楼层落下的鸡蛋都会碎，从 f 楼层或比它低的楼层落下的鸡蛋都不会破。
每次操作，你可以取一枚没有碎的鸡蛋并把它从任一楼层 x 扔下（满足 1 <= x <= n）。如果鸡蛋碎了，你就不能再次使用它。如果某枚鸡蛋扔下后没有摔碎，则可以在之后的操作中 重复使用 这枚鸡蛋。
请你计算并返回要确定 f 确切的值 的 最小操作次数 是多少？

示例 1：

输入：k = 1, n = 2
输出：2
解释：
鸡蛋从 1 楼掉落。如果它碎了，肯定能得出 f = 0 。 
否则，鸡蛋从 2 楼掉落。如果它碎了，肯定能得出 f = 1 。 
如果它没碎，那么肯定能得出 f = 2 。 
因此，在最坏的情况下我们需要移动 2 次以确定 f 是多少。 

示例 2：
输入：k = 2, n = 6   6>=f>=0 
输出：3
鸡蛋从3落下，如果坏了，2>=f>=0, 否则 6>=f>=4

示例 3：
输入：k = 3, n = 14
输出：4

示例 4：
输入：k = 2, n = 7   7>=f>=0
鸡蛋从3落下，如果坏了， 2>=f>=0, 否则 7>=f>=4
鸡蛋从4落下，如果坏了， 3>=f>=0, 否则 7>=f>=5

错误思路：
有的读者也许会有这种想法：二分查找排除楼层的速度无疑是最快的，那干脆先用二分查找，等到只剩 1 个鸡蛋的时候再执行线性扫描，这样得到的结果是不是就是最少的扔鸡蛋次数呢？
很遗憾，并不是，比如说把楼层变高一些，100 层，给你 2 个鸡蛋，你在 50 层扔一下，碎了，那就只能线性扫描 1～49 层了，最坏情况下要扔 50 次。
如果不要「二分」，变成「五分」「十分」都会大幅减少最坏情况下的尝试次数。比方说第一个鸡蛋每隔十层楼扔，在哪里碎了第二个鸡蛋一个个线性扫描，总共不会超过 20 次。
最优解其实是 14 次。最优策略非常多，而且并没有什么规律可言。

解题思路：
动态规划的方法 dp[k][n]
"""


def superEggDrop(k: int, n: int) -> int:
    if k == 1:
        return n

    dp = np.zeros(shape=(k + 1, n + 1), dtype=np.int32)
    for i in range(1, k + 1):
        for j in range(1, n + 1):
            if i == 1:
                dp[i][j] = j
            elif j == 1:
                dp[i][j] = 1
            else:
                # try_times = []
                min_value = None
                for n_i in range(1, j + 1):  # 在1--n楼层尝试
                    # 当前楼层碎了 、 当前楼层没碎 两种情况
                    cur_val = max(dp[i - 1][n_i - 1], dp[i][j - n_i]) + 1
                    if min_value is None:
                        min_value = cur_val
                    else:
                        if cur_val <= min_value:
                            min_value = cur_val
                        else:
                            break
                    # try_times.append(max(dp[i - 1][n_i - 1], dp[i][j - n_i]) + 1)

                # print("try_times", try_times)
                # dp[i][j] = min(try_times)
                dp[i][j] = min_value

    print(dp)
    return dp[-1][-1]


def test_super_egg_drop():
    print(superEggDrop(k=1, n=2))
    print(superEggDrop(k=2, n=2))
    print(superEggDrop(k=2, n=6))
    print(superEggDrop(k=2, n=7))
    print(superEggDrop(k=3, n=14))
    print(superEggDrop(k=2, n=100))
    # print(superEggDrop(k=4, n=2000))


if __name__ == '__main__':
    # test_envelopes()
    # test_max_profit()
    # test_max_profit2()
    # test_maximal_square()
    # test_max_sub_array()
    # test_minimum_total()
    # test_one_edit_away()
    # test_can_partition()
    # test_can_partition_k_subsets()
    test_super_egg_drop()
