#!/usr/bin/env bash
echo "-----------------字符串分割---------------------"
string_split() {
    local str                       # 定义局部变量
    str=$1
    delimit=$2
    OLD_IFS="$IFS"
    IFS="|"
    split_arr=($str)
    IFS="$OLD_IFS"
}

str="Alex Wang| 20 | AE"
string_split "Alex Wang| 20 | AE" "|"
echo $?
echo ${split_arr[0]}                # 函数内的变量默认是全局变量
age=${split_arr[1]}                 # 数值型字符串可以自动转化成数字
echo `expr $age + 1`