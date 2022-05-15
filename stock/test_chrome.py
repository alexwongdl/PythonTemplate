"""
selenium
1.pip3 install selenium
2.chrome浏览器-->帮助-->关于Google Chrome查看版本
3.下载对应版本驱动：http://chromedriver.storage.googleapis.com/index.html
4. mv chromedriver /usr/local/bin
5. 验证是否安装成功：chromedriver -v
"""

if __name__ == '__main__':
    """
    查找页面上唯一的元素
    find_element_by_name
    find_element_by_id
    find_element_by_xpath
    find_element_by_link_text
    find_element_by_partial_link_text
    find_element_by_tag_name
    find_element_by_class_name
    find_element_by_css_selector
    查找页面上重复的元素
    find_elements_by_name
    find_elements_by_id
    find_elements_by_xpath
    find_elements_by_link_text
    find_elements_by_partial_link_text
    find_elements_by_tag_name
    find_elements_by_class_name
    find_elements_by_css_selector
    
    from selenium.webdriver.common.by import By
    #然后就可以
    input_first = driver.find_element(By.ID,"q")
    
    获取到页面节点后
    节点.get_attribute('class')
    """
    from selenium import webdriver
    from selenium.webdriver.remote.webelement import WebElement

    chrome = webdriver.Chrome()

    chrome.get("https://finance.sina.com.cn/realstock/company/sz000001/nc.shtml")
    # print(chrome.page_source)
    name = chrome.find_element_by_class_name('c8_name').text
    print(name)

    price = chrome.find_element_by_id('price').text
    print(price)

    change_ratio = chrome.find_element_by_id("changeP").text.strip("%")
    print(change_ratio)

    pe_table_info = chrome.find_element_by_id('hqDetails')
    pe_table = pe_table_info.find_element_by_tag_name("table")
    print(pe_table.text)

    exchange_ratio = chrome.find_element_by_xpath("//*[@id='hqDetails']/table/tbody/tr[2]/td[3]")
    exchange_ratio_percent = exchange_ratio.text.strip("%")
    print(exchange_ratio.text, exchange_ratio_percent)

    pe = chrome.find_element_by_xpath("//*[@id='hqDetails']/table/tbody/tr[3]/td[3]")
    pb = chrome.find_element_by_xpath("//*[@id='hqDetails']/table/tbody/tr[4]/td[3]")
    pe_ratio = pe.text
    pb_ratio = pb.text
    print(pe_ratio, pb_ratio)

    chrome.get("http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpOtherInfo/stockid/000001/menu_num/2.phtml")
    simple_info = chrome.find_element_by_xpath("//*[@id='con02-0']/table/tbody/tr[3]/td[1]")
    industry = simple_info.text
    print(industry)

    chrome.get(
        "http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CirculateStockHolder/stockid/000001/displaytype/30.phtml")
    # print(chrome.page_source)
    # shareholder_table = chrome.find_element_by_id('con02-3')
    # print(shareholder_table.text)
    for i in range(10):
        shareholder = chrome.find_element_by_xpath(
            "//*[@id='CirculateShareholderTable']/tbody/tr[{}]/td[2]/div".format(4 + i)).text
        share_ratio = chrome.find_element_by_xpath(
            "//*[@id='CirculateShareholderTable']/tbody/tr[{}]/td[4]/div".format(4 + i)).text
        print(shareholder)
        print(share_ratio)

    print("--------------------------------------")
    chrome.get("http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/000001.phtml")
    # print(chrome.page_source)
    # share_time = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[1]")
    songgu = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[2]").text
    zhuangu = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[3]").text
    paixi = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[4]").text
    chuquanri = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[6]").text
    dengjiri = chrome.find_element_by_xpath("//*[@id='sharebonus_1']/tbody/tr[1]/td[7]").text

    print(songgu, zhuangu, paixi, chuquanri, dengjiri)
    chrome.close()
