# -*- coding: utf-8-*-

# 抓取网易的股票信息，股票名字、代码、所属行业
import re
import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from multiprocessing import Pool
from selenium.webdriver.common.by import By
import requests
from requests.adapters import HTTPAdapter
import xlwt
import xlrd
from bs4 import BeautifulSoup
import traceback
from stock import num_util
import json
import itertools
from xlutils.copy import copy
from multiprocessing.dummy import Pool

save_file_name = "stock_sina.xls"
save_file_txt = "stock_sina.txt"


def requests_with_retry(url, header, retry_times=3):
    s = requests.Session()
    # 重试次数为3
    s.mount('http://', HTTPAdapter(max_retries=retry_times))
    s.mount('https://', HTTPAdapter(max_retries=retry_times))
    # 超时时间为5s, content/status_code
    response = s.get(url, timeout=5, headers=header)
    return response


def get_sheet_by_name(book, name):
    """Get a sheet by name from xlwt.Workbook, a strangely missing method.
    Returns None if no sheet with the given name is present.
    """
    # Note, we have to use exceptions for flow control because the
    # xlwt API is broken and gives us no other choice.
    try:
        for idx in itertools.count():
            sheet = book.get_sheet(idx)
            if sheet.name == name:
                return sheet
    except IndexError:
        return None

    # def test(self, stock_id, stock_name):
    #     # 定义网址，获取上交所创业板只需对应修改stock_num为6开头或3开头即可
    #     stock_num = str(stock_id).zfill(7)
    #     if stock_id > 600000:
    #         stock_num = str(stock_id).zfill(7)
    #     else:
    #         stock_num = '1' + str(stock_id).zfill(6)
    #
    #     # 股价信息
    #     # 当前股价, 当前涨跌幅度%, 社保基金[['全国社保基金一一七组合', '0.35%', '6,800.00', '减持2,702.95']]
    #     cur_price, change, shebao_result = self.current_info(stock_num, stock_id)
    #
    #     # 实时大单
    #     # 实时资金流入(万元), 大单主买:{}手, 大单主卖:{}手, 大单成交占比:{}%
    #     shishi_zijin_liuru, dadan_buy, dadan_sell, dadan_ratio = self.zijinliuxiang_realtime(stock_num=stock_num,
    #                                                                                          stock_id=stock_id)
    #
    #     # 历史
    #     #  日期	收盘价(元)	涨跌幅%	换手率%	资金流入（万元）	资金流出（万元）	净流入（万元）
    #     # 主力流入（万元）	主力流出（万元）	主力净流入（万元）
    #     info_list_zijin_list = self.zijin_liuxiang_history(stock_num=stock_num, stock_id=stock_id, max_day=20)
    #
    #     # 行业信息:
    #     stock_shiyinglv, domain_avg_shiyinglv, domain_other_stock = self.domain_compare(stock_num, stock_id, stock_name)
    #
    #     # 送股/派息
    #     gonggao_day, songgu, zhuanzen, paixi, guquan_dengji, guquan_chixi = self.gaosongzhuan(stock_num, stock_id)

    def get_all_shebao_stock(self, write_index):
        """
        获取所有社保投资的股票
        :return:
        """
        book = xlrd.open_workbook(save_file_name)
        print(book.sheet_names())
        sheet = book.sheet_by_index(0)
        print("表名:{}, 行数:{}, 列数:{}".format(sheet.name, sheet.nrows, sheet.ncols))
        wb = copy(book)
        ws = wb.get_sheet(write_index)
        ws.write(1, 0, u'股票代码')
        ws.write(1, 1, u'股票名称')
        ws.write(1, 2, u'股票板块')
        ws.write(1, 3, u'id')
        ws.write(1, 4, u'url')
        ws.write(1, 5, u'当前价格')
        ws.write(1, 6, u'涨跌幅%')
        ws.write(1, 7, u'社保基金')
        ws.write(1, 8, u'社保基金数量')
        ws.write(1, 9, u'市盈率%')
        ws.write(1, 10, u'行业平均市盈率%')
        ws.write(1, 11, u'行业头部股票市盈率')
        ws.write(1, 12, u'送股')
        ws.write(1, 13, u'转增')
        ws.write(1, 14, u'派息')
        ws.write(1, 15, u'除权登记')
        ws.write(1, 16, u'除权日')
        ws.write(1, 17, u'年份')

        line_index = 2
        for rx in range(2, sheet.nrows):
            row = sheet.row(rx)
            stock_num, stock_name, stock_area, stock_id, url = row
            # if "*ST" in stock_name.value:
            #     continue

            stock_num_format = stock_num.value
            stock_name = stock_name.value
            stock_area = stock_area.value
            stock_id = int(stock_id.value)
            url = url.value
            # stock_id = int(stock_id)
            stock_num = str(stock_id).zfill(7)
            if stock_id > 600000:
                stock_num = str(stock_id).zfill(7)
            else:
                stock_num = '1' + str(stock_id).zfill(6)
            print(stock_num_format, stock_num, stock_name, stock_area, stock_id, url)

            # 股价信息
            # 当前股价, 当前涨跌幅度%, 社保基金[['全国社保基金一一七组合', '0.35%', '6,800.00', '减持2,702.95']]
            cur_price, change, shebao_result = self.current_info(stock_num, stock_id)

            # 实时大单
            # 实时资金流入(万元), 大单主买:{}手, 大单主卖:{}手, 大单成交占比:{}%
            # shishi_zijin_liuru, dadan_buy, dadan_sell, dadan_ratio = self.zijinliuxiang_realtime(stock_num=stock_num,
            #                                                                                      stock_id=stock_id)

            # 历史
            #  日期	收盘价(元)	涨跌幅%	换手率%	资金流入（万元）	资金流出（万元）	净流入（万元）
            # 主力流入（万元）	主力流出（万元）	主力净流入（万元）
            # info_list_zijin_list = self.zijin_liuxiang_history(stock_num=stock_num, stock_id=stock_id, max_day=20)

            # 行业信息: 股票市盈率%, 行业平均市盈率%, 行业头部股票市盈率
            stock_shiyinglv, domain_avg_shiyinglv, domain_other_stock = self.domain_compare(stock_num, stock_id,
                                                                                            stock_name)
            # 送股/派息
            gonggao_day, songgu, zhuanzen, paixi, guquan_dengji, guquan_chixi = self.gaosongzhuan(stock_num, stock_id)

            ws.write(line_index, 0, stock_num_format)
            ws.write(line_index, 1, stock_name)
            ws.write(line_index, 2, stock_area)
            ws.write(line_index, 3, stock_id)
            ws.write(line_index, 4, url)
            ws.write(line_index, 5, cur_price)
            ws.write(line_index, 6, change)
            if shebao_result is None:
                ws.write(line_index, 7, "")
                ws.write(line_index, 8, 0)
            else:
                ws.write(line_index, 7, ",  ".join(["__".join(item) for item in shebao_result]))
                ws.write(line_index, 8, len(shebao_result))
            ws.write(line_index, 9, stock_shiyinglv)
            ws.write(line_index, 10, domain_avg_shiyinglv)
            if domain_other_stock is None:
                ws.write(line_index, 11, "")
            else:
                ws.write(line_index, 11, ",  ".join(["__".join(item) for item in domain_other_stock]))

            ws.write(line_index, 12, songgu)
            ws.write(line_index, 13, zhuanzen)
            ws.write(line_index, 14, paixi)
            ws.write(line_index, 15, guquan_dengji)
            ws.write(line_index, 16, guquan_chixi)
            if guquan_chixi is None:
                ws.write(line_index, 17, "")
            else:
                ws.write(line_index, 17, guquan_chixi.split('-')[0])
            wb.save(save_file_name)
            line_index += 1
            print("")


def get_sotck_num_url(stock_id):
    if stock_id > 600000:
        stock_num = "sh" + str(stock_id).zfill(6)
    else:
        stock_num = 'sz' + str(stock_id).zfill(6)
        # http://finance.sina.com.cn/realstock/company/sz300052/nc.shtml
        # https://finance.sina.com.cn/realstock/company/sz000001/nc.shtml
    url = "http://finance.sina.com.cn/realstock/company/{}/nc.shtml".format(stock_num)
    return stock_num, url


def get_stock_info(stock_id_list):
    options = Options()
    options.add_argument('–headless')
    options.add_argument('–no-sandbox')
    options.add_argument('–disable-dev-shm-usage')
    chrome = webdriver.Chrome(chrome_options=options)

    result_list = []
    for stock_id in stock_id_list:
        try:
            stock_zfill = str(stock_id).zfill(6)
            stock_num, url = get_sotck_num_url(stock_id)
            print("股票代码:{}, url:{}".format(stock_num, url))
            chrome.get(url)

            name = None
            try:
                name = chrome.find_element_by_class_name('c8_name')
            except Exception as e:
                print("get c8_name error:{}".format(stock_id))
            if not name:
                continue
            name = name.text

            price = chrome.find_element_by_id('price').text
            change_ratio = chrome.find_element_by_id("changeP").text.strip("%")

            exchange_ratio = chrome.find_element_by_xpath("//*[@id='hqDetails']/table/tbody/tr[2]/td[3]")
            exchange_ratio_percent = exchange_ratio.text.strip("%")

            pe = chrome.find_element_by_xpath("//*[@id='hqDetails']/table/tbody/tr[3]/td[3]").text
            pb = chrome.find_element_by_xpath("//*[@id='hqDetails']/table/tbody/tr[4]/td[3]").text

            chrome.get("http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpOtherInfo/stockid/{}/menu_num/2.phtml".
                       format(stock_zfill))
            simple_info = chrome.find_element_by_xpath("//*[@id='con02-0']/table/tbody/tr[3]/td[1]")
            industry = simple_info.text

            chrome.get(
                "http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CirculateStockHolder/stockid/{}/displaytype/30.phtml".
                    format(stock_zfill))
            shebao_list = []
            for i in range(10):
                shareholder = chrome.find_element_by_xpath(
                    "//*[@id='CirculateShareholderTable']/tbody/tr[{}]/td[2]/div".format(4 + i)).text
                share_ratio = chrome.find_element_by_xpath(
                    "//*[@id='CirculateShareholderTable']/tbody/tr[{}]/td[4]/div".format(4 + i)).text

                if "社保" in shareholder:
                    shebao_list.append(shareholder)
            shebao_num = len(shebao_list)

            chrome.get("http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{}.phtml".
                       format(stock_zfill))
            gonggao_date = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[1]").text
            songgu = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[2]").text
            zhuangu = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[3]").text
            paixi = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[4]").text
            chuquanri = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[6]").text
            dengjiri = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[7]").text

            result_list.append((stock_id, name, price, change_ratio, exchange_ratio_percent, pb, pe, industry,
                                shebao_list, shebao_num,
                                gonggao_date, songgu, zhuangu, paixi, chuquanri, dengjiri))
        except Exception as e:
            traceback.print_exc()
            print("get info error:{}".format(stock_id))

    chrome.close()
    return result_list


def devide_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]


def list_all_stock(stock_id_list, file_name):
    count1 = 1

    # wb = xlwt.Workbook()
    # ws = wb.add_sheet(u'stock')
    # ws.write(0, 0, u'股票代码')
    # ws.write(0, 1, u'股票名称')
    # ws.write(0, 2, u'股票板块')
    # ws.write(0, 3, u'id')
    # ws.write(0, 4, u'head')
    # ws2 = wb.add_sheet(u'industry')
    # ws2.write(0, 0, u'股票板块')
    # ws2.write(0, 1, u'统计时间')

    # 目前深证最大号为002725，获取上交所创业板请修改相应最大号码
    # stock_id_list = [i for i in range(1, 100)]
    # stock_id_list = [i for i in range(1, 3044)]  # 1-3043 深A
    # stock_id_list.extend([i for i in range(300001, 301289)])  # 300001-301288 深A创

    # stock_id_list.extend([i for i in range(600000, 604000)])  # 600000 - 603999 沪A
    # stock_id_list.extend([i for i in range(605001, 605600)])  # 605001 - 605599 沪A
    #### stock_id_list.extend([i for i in range(688001, 688982)])  # 688001 - 688981 沪A 科创板
    print(stock_id_list)

    pool_num = 16

    stock_id_list_n = []
    for stock_ids in devide_list(stock_id_list, int(len(stock_id_list) / pool_num / 5)):
        stock_id_list_n.append(stock_ids)
    print("stock_id_list_n", stock_id_list_n, len(stock_id_list_n))

    # pool = Pool(len(stock_id_list_n))
    pool = Pool(pool_num)
    results_list = pool.map(get_stock_info, stock_id_list_n)
    pool.close()
    pool.join()
    # results_list = get_stock_info(stock_id_list)

    with open(file_name, 'w') as writer:
        writer.write("股票id\t股票名\t价格\t涨跌幅(%)\t换手率(%)\t市盈率\t市净率\t行业板块\t社保基金\t社保基金数"
                     "\t公告日期\t送股\t转股\t派息\t除权日\t登记日\n")
        # results = get_stock_info(stock_id_list)
        # print("results", results)
        for results in results_list:
            for result in results:
                stock_id, name, price, change_ratio, exchange_ratio_percent, pe, pb, industry, \
                shebao_list, shebao_num, gonggao_date, songgu, zhuangu, paixi, chuquanri, dengjiri = result

                writer.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t"
                             "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".
                             format(stock_id, name, price, change_ratio, exchange_ratio_percent, pe, pb, industry,
                                    shebao_list, shebao_num,
                                    gonggao_date, songgu, zhuangu, paixi, chuquanri, dengjiri))

    # wb.save(save_file_name)


if __name__ == '__main__':
    # text = 'Hello, "find.me-_/\\" please help with python regex'
    # pattern = r'"([A-Za-z0-9_\./\\-]*)"'
    # m = re.search(pattern, text)
    # print(m.group())

    # 遍历所有股票
    # stock_id_list = []
    # stock_id_list.extend([i for i in range(1, 3044)])  # 1-3043 深A
    # list_all_stock(stock_id_list, "sz_stock_sina_1.txt")

    # stock_id_list = []
    # stock_id_list.extend([i for i in range(300001, 301289)])  # 300001-301288 深A创
    # list_all_stock(stock_id_list, "sz_stock_sina_2.txt")

    # stock_id_list = []
    # stock_id_list.extend([i for i in range(600000, 604000)])  # 600000 - 603999 沪A
    # list_all_stock(stock_id_list, "sh_stock_sina_1.txt")

    stock_id_list = []
    stock_id_list.extend([i for i in range(605001, 605600)])  # 605001 - 605599 沪A
    list_all_stock(stock_id_list, "sh_stock_sina_2.txt")

    pass

    # stock_info = StockInfo()
    # stock_info.test(1, "平安银行")
    # stock_info.test(600507, "方大特刚")
    # stock_info.get_all_shebao_stock(write_index=3)
