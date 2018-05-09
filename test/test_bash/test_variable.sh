#!/usr/bin/env bash
echo '------------------- $ ---------------------'  # 单引号内所有字符原样输出
echo $1                     # 脚本第一个变量,$0是脚本文件名
echo $#                     # 脚本变量个数
if [ $? -eq 0 ]             # 上一个程序退出状态
then
    echo 'previous program run succeed'
fi

echo "-----------------反引号---------------------"
files=`ls`                  # 反引号:运行shell指令
for file in ${files}
do
    echo ${file}
done


echo "-----------------字符串---------------------"
name="Alex Wang"
echo "length of name:" ${#name}     # 字符串长度
echo "part of name:" ${name:0:5}    # 截取子串


echo "------------------数组---------------------"
arr=("Alex" 00 "Wang")              # 数组元素之间用空格分隔
echo ${arr[1]}
arr[1]=90
echo ${arr[*]}                      # 读取数组的所有元素 *或者@
echo length of arr: ${#arr[*]}      # 获取数组长度

for item in ${arr[*]}               # 遍历数组
do
    echo item: ${item}
done


