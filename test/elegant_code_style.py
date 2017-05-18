## Alex wang @ 20170512

def ifelse(weight):
    body = "fat" if weight > 120 else "thin"
    print(body)

def test_cnumerate():
    print("test enumerate.........")
    str_list = ["one","two","three","four"]
    for i, str in enumerate(str_list):
        print("{}\t{}".format(i,str))

def test_zip():
    print("test zip..........")
    list_one = [1,2,3,4]
    list_two = ["one","two","three","four"]
    for i, str in zip(list_one, list_two):
        print("{}\t{}".format(i, str))

    print("test zip..........")
    tuple_one = (1,2,3,4)
    tuplt_two = ("one","two","three","four")
    for i,str in zip(tuple_one, tuplt_two):
        print("{}\t{}".format(i, str))
