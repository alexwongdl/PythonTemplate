"""
 @author: AlexWang
 @date: 2020/11/25 8:30 PM
"""


def stock_money_wan_format(org_money):
    """
    -10,017.15万元 --> -10017.15
    :param org_money:
    :return:
    """
    org_money = org_money.strip().replace(",", "")
    if org_money.endswith("万元"):
        money = org_money.replace("万元", "")
        return float(money)
    elif org_money.endswith("万"):
        money = org_money.replace("万", "")
        return float(money)
    elif org_money.endswith("元"):
        money = float(org_money.replace("元", "")) / 10000
        return money
    elif org_money.endswith("亿"):
        money = float(org_money.replace("亿", "")) * 10000
        return money
    else:
        raise ValueError("stock_money_wan_format value error")


def stock_money_shou_format(org_shou):
    """
    131,916手
    167,763手
    :param org_shou:
    :return:
    """
    org_shou = org_shou.strip().replace(",", "")
    if org_shou.endswith("手"):
        return int(org_shou.replace("手", ""))
    elif org_shou.endswith("万手"):
        return int(org_shou.replace("万手", "")) * 10000
    else:
        raise ValueError("stock_money_shou_format value error")


if __name__ == '__main__':
    print(stock_money_wan_format("-10,017.15万元"))
    print(stock_money_wan_format("-10,017.15元"))
    print(stock_money_wan_format("-10,017.15亿"))
