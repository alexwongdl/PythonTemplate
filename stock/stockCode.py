# -*- coding: utf-8-*-

# 抓取网易的股票信息，股票名字、代码、所属行业
import re
# , urllib2
from urllib import request
import xlwt
import xlrd
from bs4 import BeautifulSoup
import traceback
from stock import num_util
import json
import itertools
from xlutils.copy import copy

save_file_name = "stock.xls"


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


class getstock:
    def __init__(self):
        pass

    def go(self, stock_id, excel_line_num, ws):
        # 定义网址，获取上交所创业板只需对应修改stock_num为6开头或3开头即可
        stock_num = str(stock_id).zfill(7)
        if stock_id > 600000:
            stock_num = str(stock_id).zfill(7)
        else:
            stock_num = '1' + str(stock_id).zfill(6)
        url = 'http://quotes.money.163.com/' + stock_num + '.html'
        # print("股票代码:" + stock_num)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6"}
        req = request.Request(url, headers=headers)
        try:
            content = request.urlopen(req).read()
        except Exception as e:
            print("{} {} error".format(stock_id, url))
            return 0
        soup = BeautifulSoup(content)
        #  print content
        c = soup.findAll('div', {'class': 'stock_info'})
        # print c
        name = soup.find('h1', {'class': 'name'}).contents[1].contents[0]
        # print name

        c = soup.findAll('div', {'class': 'relate_stock clearfix'})
        # print c[1]
        c1 = c[1].find('li')
        industry_name = c1.contents[0].string.strip()
        # print name
        # industry = c[1].find('li')
        # industry_name = industry.contents[0].contents[0].encode('utf-8').strip()
        print("stock_id:{}, name:{}, industry_name:{}, url:{}".format(stock_num, name, industry_name, url))
        if name != '':
            # print(excel_line_num)
            # print name
            # ws.write(str(str(count).zfill(6))+'%'+str(name)+ '%'+str(industry_name) +'\n')
            ws.write(excel_line_num, 0, str(stock_id).zfill(6))
            ws.write(excel_line_num, 1, name)
            ws.write(excel_line_num, 2, industry_name)
            ws.write(excel_line_num, 3, stock_id)
            ws.write(excel_line_num, 4, url)
            return 1

        return 0


class StockInfo:
    def __init__(self):
        pass

    def get_url_content(self, stock_num, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6"}
        # headers = {
        #     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36"}
        req = request.Request(url, headers=headers)
        try:
            content = request.urlopen(req, timeout=30).read()
            return content
        except Exception as e:
            traceback.print_exc()
            print("{} {} error".format(stock_num, url))
            return None

    def zijinliuxiang_realtime(self, stock_num, stock_id):
        """
        资金流向
        :return:
        """
        # http://quotes.money.163.com/trade/zjlx_000001.html#01b02
        url = 'http://quotes.money.163.com/trade/zjlx_' + str(stock_id).zfill(6) + '.html#01b02'
        print("实时资金流向", url)
        # print("股票代码:" + stock_num)
        content = self.get_url_content(stock_num, url)
        if content is None:
            return None

        soup = BeautifulSoup(content, "lxml")

        # print(soup.prettify())
        # print(soup.find_all('script'))
        # ------------------ 大单信息 -----------------
        shishi_zijin_liuru_node = soup.find_all('div', {"class": "three_cols clearfix"})[0]
        shishi_zijin_liuru_text = shishi_zijin_liuru_node.find('strong', {"class": "chart_title"}).get_text(). \
            replace(" ", "").replace("\n", "").replace("\r", "").split("：")
        shishi_zijin_liuru = num_util.stock_money_wan_format(shishi_zijin_liuru_text[1])
        # print("---------------------------")
        print("实时资金流入:{}万".format(shishi_zijin_liuru))
        # print("---------------------------")
        dadan_info = soup.find('div', {'class': 'da_dan_info'})
        # print(dadan_info)
        # print("---------------------------")
        trs = dadan_info.find_all('tr')

        for tr in trs:
            cols = tr.find_all('td')
            cols = [ele.text.replace(" ", "").strip() for ele in cols]
            if "大单主买量" in cols[0]:
                dadan_buy = num_util.stock_money_shou_format(cols[1])  # int(cols[1].split('手')[0].replace(",", ""))
            if "大单主卖量" in cols[0]:
                dadan_sell = num_util.stock_money_shou_format(cols[1])  # int(cols[1].split('手')[0].replace(",", ""))
            if "大单成交占比" in cols[0]:
                dadan_ratio = float(cols[1].split('%')[0])
        print("大单主买:{}手, 大单主卖:{}手, 大单成交占比:{}%".format(dadan_buy, dadan_sell, dadan_ratio))
        return shishi_zijin_liuru, dadan_buy, dadan_sell, dadan_ratio

    def zijin_liuxiang_history(self, stock_num, stock_id, max_day=20):
        """
        历史资金流向
        :param stock_num:
        :param stock_id:
        :return:
        """
        # http://quotes.money.163.com/trade/lszjlx_000001.html#01b08
        url = 'http://quotes.money.163.com/trade/lszjlx_' + str(stock_id).zfill(6) + '.html#01b08'
        print("历史资金流向", url)
        # print("股票代码:" + stock_num)
        content = self.get_url_content(stock_num, url)
        if content is None:
            return None

        soup = BeautifulSoup(content, "lxml")

        # ------------------ 历史资金流向表 -----------------
        liuxiang_table = soup.find('table', {"class": "table_bg001 border_box"})
        # print(liuxiang_table)
        trs = liuxiang_table.find_all('tr')
        idx = -1

        info_list = []
        for tr in trs:
            idx += 1
            if idx <= 0:
                continue
            if idx > max_day:
                break
            cols = tr.find_all('td')
            cols = [ele.text.replace(" ", "").strip() for ele in cols]
            # print(cols)
            # 日期	收盘价	涨跌幅	换手率	资金流入（万元）	资金流出（万元）	净流入（万元）
            # 主力流入（万元）	主力流出（万元）	主力净流入（万元）
            day_str, shoupan_price, raise_fall_ratio, huanshou_ratio, money_in, money_out, pure_in, \
            main_in, main_out, main_pure_in = cols
            info_list.append((day_str, float(shoupan_price), float(raise_fall_ratio.replace("%", "")),
                              float(huanshou_ratio.replace("%", "")),
                              num_util.stock_money_wan_format("{}万".format(money_in)),
                              num_util.stock_money_wan_format("{}万".format(money_out)),
                              num_util.stock_money_wan_format("{}万".format(pure_in)),
                              num_util.stock_money_wan_format("{}万".format(main_in)),
                              num_util.stock_money_wan_format("{}万".format(main_out)),
                              num_util.stock_money_wan_format("{}万".format(main_pure_in))))
        print(info_list)
        return info_list

    def current_info(self, stock_num, stock_id):
        # http://quotes.money.163.com/1000001.html
        url = 'http://quotes.money.163.com/' + stock_num + '.html'
        print("当前信息:", url)
        content = self.get_url_content(stock_num, url)
        if content is None:
            return None, None, None

        soup = BeautifulSoup(content, "lxml")
        # print(soup.body.prettify())
        # ------------------ 股价 -----------------
        stock_detail = soup.find('div', {"class": "stock_detail"})
        shiyinglv = soup.find('td', {'title': "市盈率=最新股价/最近四个季度每股收益之和"})

        # 价格/涨跌
        pattern = re.compile(r'window.stock_info')
        # window.stock_info = {
        #     name: '平安银行',
        #     code: '000001',
        #     price: '19.7',
        #     change: '1.03%25',
        #     yesteday: '19.5',
        #     today: '20',
        #     high: '20',
        #     low: '19.38',
        #     note: '网易财经',
        #     pic: 'http://img2.money.126.net/chart/hs/time/540x360/1000001.png',
        #     symbol: '1000001',
        #     url: location.href
        # }
        price_script = soup.find('script', text=pattern).text

        # cur_price = re.match('price: \'(\d*).(\d*)\'', price_script).group(0)
        cur_price_match = re.search(r"price: '(.*)'", price_script)
        cur_price_str = cur_price_match.group(0).replace("\'", "").split("price:")[1].strip()
        # print("cur_price_str:{}".format(cur_price_str))
        if len(cur_price_str) == 0:
            cur_price = None
        else:
            cur_price = float(cur_price_str)
        change = re.search(r"change: '(.*)'", price_script).group(0)
        cur_change_str = change.replace("\'", "").split("change:")[1].split("%")[0].strip()
        # print("cur_change_str:{}".format(cur_change_str))
        if len(cur_change_str) == 0:
            change = None
        else:
            change = float(cur_change_str)

        # ------------------ 十大股东 -----------------
        # print("------------------ 十大股东 -----------------")
        gudong_table = soup.find('div', {"class": "col_ml"})

        shebao_result = None
        if gudong_table is not None:
            gudong_table = gudong_table.find("table", {"class": "table_bg001 border_box"})
            trs = gudong_table.find_all('tr')

            for tr in trs:
                cols = tr.find_all('td')
                cols = [ele.text.replace(" ", "").strip() for ele in cols]
                # print(cols)
                if len(cols) <= 0:
                    continue
                if "社保基金" in cols[0]:
                    if shebao_result is None:
                        shebao_result = [cols]
                    else:
                        shebao_result.append(cols)
        # 当前加个, 涨幅, 社保基金[['全国社保基金一一七组合', '0.35%', '6,800.00', '减持2,702.95']]
        print("当前价格:{}, 涨跌:{}%, 社保基金:{}".format(cur_price, change, shebao_result))
        return cur_price, change, shebao_result

    def gaosongzhuan(self, stock_num, stock_id):
        # http://quotes.money.163.com/f10/fhpg_600690.html#01d05
        url = 'http://quotes.money.163.com/f10/fhpg_' + str(stock_id).zfill(6) + '.html#01d05'
        print("当前信息:", url)
        content = self.get_url_content(stock_num, url)
        if content is None:
            return None, None, None, None, None, None

        soup = BeautifulSoup(content, "lxml")
        # print(soup.body.prettify())
        # ------------------ 送股/派息 -----------------
        guxi_table = soup.find('table', {"class": "table_bg001 border_box limit_sale"})
        trs = guxi_table.find_all('tr')
        tr = trs[2]
        print("trs2:{}".format(tr))
        # for tr in trs[1:-1]:
        #     print(tr)
        cols = tr.find_all('td')
        cols = [ele.text.replace(" ", "").strip() for ele in cols]
        if len(cols) < 7:
            return None, None, None, None, None, None
        gonggao_day = cols[0]
        guquan_dengji = cols[5]
        guquan_chixi = cols[6]

        songgu = 0
        zhuanzen = 0
        paixi = 0
        try:
            songgu = float(cols[2])
        except Exception as e:
            pass
        try:
            zhuanzen = float(cols[3])
        except Exception as e:
            pass
        try:
            paixi = float(cols[4])
        except Exception as e:
            pass

        print(gonggao_day, songgu, zhuanzen, paixi, guquan_dengji, guquan_chixi)
        return gonggao_day, songgu, zhuanzen, paixi, guquan_dengji, guquan_chixi

    def domain_compare(self, stock_num, stock_id, stock_name):
        url = 'http://quotes.money.163.com/f10/hydb_' + str(stock_id).zfill(6) + '.html#01g02'
        print("行业对比:", url)
        content = self.get_url_content(stock_num, url)
        if content is None:
            return None, None, None

        soup = BeautifulSoup(content, "lxml")
        # print(soup.body.prettify())

        # 排名  名称  市盈率  市净率  市现率  市销率
        hangye_compare_table = soup.find("table", {"class": "table_bg001 border_box table_sortable"})
        trs = hangye_compare_table.find_all('tr')

        domain_other_stock = []
        stock_shiyinglv = None
        for tr in trs[1:-1]:
            cols = tr.find_all('td')
            cols = [ele.text.replace(" ", "").strip() for ele in cols]

            if cols[1].strip() == stock_name:
                # 当前股票
                try:
                    stock_shiyinglv = float(cols[2])
                except Exception as e:
                    stock_shiyinglv = None
            else:
                domain_other_stock.append((cols[1], cols[2]))

        # print("-----------------------------")
        domain_average = trs[-1]
        cols = domain_average.find_all('td')
        cols = [ele.text.replace(" ", "").strip() for ele in cols]
        # print(cols)
        try:
            domain_avg_shiyinglv = float(cols[2])
        except Exception as e:
            domain_avg_shiyinglv = None

        if stock_shiyinglv is None:
            print("Error:{} 市盈率提取失败".format(stock_name))

        # 股票市盈率, 行业平均市盈率, 行业头部股票市盈率
        print("市盈率:{}, 行业平均市盈率:{}, 行业头部股票市盈率:{}".
              format(stock_shiyinglv, domain_avg_shiyinglv, domain_other_stock))
        return stock_shiyinglv, domain_avg_shiyinglv, domain_other_stock

    def test(self, stock_id, stock_name):
        # 定义网址，获取上交所创业板只需对应修改stock_num为6开头或3开头即可
        stock_num = str(stock_id).zfill(7)
        if stock_id > 600000:
            stock_num = str(stock_id).zfill(7)
        else:
            stock_num = '1' + str(stock_id).zfill(6)

        # 股价信息
        # 当前股价, 当前涨跌幅度%, 社保基金[['全国社保基金一一七组合', '0.35%', '6,800.00', '减持2,702.95']]
        cur_price, change, shebao_result = self.current_info(stock_num, stock_id)

        # 实时大单
        # 实时资金流入(万元), 大单主买:{}手, 大单主卖:{}手, 大单成交占比:{}%
        shishi_zijin_liuru, dadan_buy, dadan_sell, dadan_ratio = self.zijinliuxiang_realtime(stock_num=stock_num,
                                                                                             stock_id=stock_id)

        # 历史
        #  日期	收盘价(元)	涨跌幅%	换手率%	资金流入（万元）	资金流出（万元）	净流入（万元）
        # 主力流入（万元）	主力流出（万元）	主力净流入（万元）
        info_list_zijin_list = self.zijin_liuxiang_history(stock_num=stock_num, stock_id=stock_id, max_day=20)

        # 行业信息:
        stock_shiyinglv, domain_avg_shiyinglv, domain_other_stock = self.domain_compare(stock_num, stock_id, stock_name)

        # 送股/派息
        gonggao_day, songgu, zhuanzen, paixi, guquan_dengji, guquan_chixi = self.gaosongzhuan(stock_num, stock_id)

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


def list_all_stock():
    count1 = 1

    wb = xlwt.Workbook()
    ws = wb.add_sheet(u'stock')
    ws.write(1, 0, u'股票代码')
    ws.write(1, 1, u'股票名称')
    ws.write(1, 2, u'股票板块')
    ws.write(1, 3, u'id')
    ws2 = wb.add_sheet(u'industry')
    ws2.write(1, 0, u'股票板块')
    ws2.write(0, 1, u'统计时间')

    gs = getstock()
    # 目前深证最大号为002725，获取上交所创业板请修改相应最大号码

    stock_id_list = [i for i in range(1, 2736)]  # 1-2725 深A
    # stock_id_list.extend([i for i in range(300000, 300410)])  # 300000 - 300409 创
    stock_id_list.extend([i for i in range(600000, 603999)])  # 600000 - 603998 沪A
    print(stock_id_list)

    excel_line_num = 2
    for stock_id in stock_id_list:
        try:
            ret = gs.go(stock_id, excel_line_num, ws)
            if ret == 1:
                excel_line_num += 1
                wb.save(save_file_name)
                print('success')
                print("")
            # print(stock_id)
            count1 += 1

        except Exception as e:
            # traceback.print_exc()
            print('{} fail'.format(stock_id))
            print("")
            count1 += 1

    wb.save(save_file_name)


if __name__ == '__main__':
    # text = 'Hello, "find.me-_/\\" please help with python regex'
    # pattern = r'"([A-Za-z0-9_\./\\-]*)"'
    # m = re.search(pattern, text)
    # print(m.group())

    # 遍历所有股票
    # list_all_stock()


    stock_info = StockInfo()
    # stock_info.test(1, "平安银行")
    # stock_info.test(600507, "方大特刚")
    stock_info.get_all_shebao_stock(write_index=3)
