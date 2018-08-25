"""
Created by Alex Wang
On 2018-08-01
"""
import Queue


def scatter_to_buckets(connect_map):
    """
    :param connect_map: dict of {a:set(b,c,d)}
    :return:
    """
    edges = set()
    node_status = {}
    for key, node_set in connect_map.items():
        node_status[key] = 0
        for node in node_set:
            node_status[node] = 0
            # add edges
            edges.add((key, node))
            edges.add((node, key))

    buckets = []
    # start create buckets
    flag = 0
    for node in node_status.keys():
        if node_status[node] == 0:
            # new bucket
            flag += 1
            group = {node}
            node_status[node] = flag

            q = Queue.Queue()  # FIFO queue
            q.put(node)

            while not q.empty():
                current_node = q.get()
                for neighbor in node_status.keys():
                    if (current_node, neighbor) in edges:
                        if node_status[neighbor] != 0 and node_status[neighbor] != flag:
                            print('error, node_status[neighbor]={} and flag={}'.
                                  format(node_status[neighbor], flag))
                        elif node_status[neighbor] == 0:
                            group.add(neighbor)
                            node_status[neighbor] = flag
                            q.put(neighbor)

            buckets.append(group)
    return buckets


if __name__ == '__main__':
    face_map = {35: set([34, 35]), 38: set([35, 3, 38, 6, 19, 24]), 39: set([34, 35, 4, 38, 39, 20]),
                56: set([56, 2, 20, 50, 54]), 50: set([2, 50]), 19: set([19, 6]), 20: set([4, 2, 20]),
                53: set([51, 53]), 54: set([2, 20, 50, 54]), 55: set([51, 53, 55]), 24: set([24, 19]),
                25: set([25, 23]), 26: set([24, 26]), 59: set([2, 50, 20, 54, 56, 59]), 58: set([58, 51, 53, 55])}
    scatter_to_buckets(face_map)
