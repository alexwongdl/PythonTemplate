#!/usr/bin/env bash

foo () {                    # 可以带function fun() 定义，也可以直接fun() 定义,不带任何参数。
    echo parameters:$*      # 获取变量
    result=`ls`
    count=0
    for item in ${result}
    do
        echo item:${item}
        count=`expr ${count} + 1`
    done
    return ${count}         # 指定函数返回值
}

foo 2 3                     # 传递变量
echo exist status of foo:$? # $?获取函数返回值