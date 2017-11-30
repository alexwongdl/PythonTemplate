"""
Created by Alex Wang
On 2017-11-30
测试pandas

pandas：list<dict>保存和读取
            list<dict>转化成DataFrame：data_frame = pd.DataFrame(data_feature)
            DataFrame保存到csv文件：result_data.to_csv(os.path.join(data_base, 'labeled_dataset_result.csv'), sep='\t')
            从csv文件加载DataFrame数据：
            DataFrame方法：head([n])--返回前n行；info()--信息总结；pop()--弹出列；drop()--丢弃列；shape--(行数，列数)；get_value(i,label)--获取值

"""
import pandas as pd

def test_pandas():
    print("test pandas")
    data_feature = []
    data_feature.append({'name':'Alex Wang', 'year':1990})
    data_feature.append({'name':'William', 'year':1991})
    data_frame = pd.DataFrame(data_feature)
    data_frame.info()

if __name__ == '__main__':
    test_pandas()