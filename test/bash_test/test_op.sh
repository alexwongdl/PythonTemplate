#!/usr/bin/env bash
echo "-----------------算数运算符---------------------"
# + - * / %(取余) == !=
echo 2*2: `expr 2 \* 2`                 # awk或者expr进行运算,数值和操作符之间需要加空格
if [ 2 == 2 ]                           # 数值判断是否相等,不能用于字符串比较
then
    echo 2==2
fi

echo "-----------------关系运算符---------------------"
# -eq -ne -gt -lt -ge -le  只支持数字,不支持字符串
a=10
b=20
if [ ${a} -le ${b} ]
then
    echo ${a} less than ${b}
fi

echo "-----------------布尔运算符---------------------"
# ! -a -o && ||
if [[ ${a} -le ${b} && ${b} -lt 50 ]]   # 必须双中括号
then
    echo ${a} less equal ${b} and  ${b} less than 50
fi

if [ ${a} -le ${b} -a ! ${b} -gt 50 ]   # 单中括号
then
    echo ${a} less equal ${b} and  ${b} not greater than 50
fi

echo "----------------字符串运算符---------------------"
# = != -z(长度为0) -n(长度不为0)
strb=""
if [[ ! ${stra} && -z ${strb} ]]        # if [ ${stra} ]判断变量是否不为空
then
    echo stra is null and length of strb is 0
fi

echo "---------------文件测试运算符---------------------"
# -d file   检测文件是否是目录
# -f file   检测是否是普通文件(既不是目录,也不是设备文件)
# -s file   检测文件是否为空
# -e file   检测文件是否存在
file='./test_variable.sh'
if [ -e ${file} ]
then
    if [ -f ${file} ]
    then
        echo ${file} is an ordinary file
    else
        echo ${file} is not an ordinary file
    fi
else
    echo ${file} is not exist
fi

