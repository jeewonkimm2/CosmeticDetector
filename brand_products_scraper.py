from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# 스크롤된 정적 페이지에서 상품 이름, url 추출
def get_brand_and_product_info(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    
    # show more button이 존재하는지 확인
    button_exists = bool(soup.select_one("button.css-bk5oor.eanm77i0[data-comp='StyledComponent BaseComponent ']"))
    
    # 첫 번째 선택자로 brand_name_element를 찾아보고, 없다면 두 번째 선택자로 찾기
    brand_name_element = soup.select_one("div.css-hc7wlr.eanm77i0[data-comp='StyledComponent BaseComponent ']") or \
                         soup.select_one("div.css-16egu7d.eanm77i0[data-comp='StyledComponent BaseComponent ']")
    
    product_containers = soup.select('div.css-foh208, div.css-1qe8tjm')
    
    if len(product_containers) == 0:
        print(f"No products found for brand: {brand_name_element.text}")
    
    product_names = []
    product_links = []
    
    for idx, container in enumerate(product_containers):
        product_element = container.find('a', class_='css-klx76')
        
        if not product_element:
#             print(f"Container {idx} does not have a 'a.css-klx76' tag.")
#             print(container.prettify())  # 해당 container의 구조를 출력
            continue

        product_name_element = product_element.find('span', class_='ProductTile-name')

        product_names.append(product_name_element.text)
        product_links.append(product_element['href'])
    
    return brand_name_element.text, product_names, product_links, button_exists


def fetch_brand_and_product_info(brand_links, existing_products):
    columns = ['brand_name', 'product_name', 'product_detail_url']
    df = pd.DataFrame(columns=columns)
    
    BASE_URL = "https://www.sephora.com/"
    
    for link in brand_links:
        full_url = BASE_URL + link
        print(full_url)
        
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246")
        driver = webdriver.Chrome(service=Service(
            ChromeDriverManager().install()), options=options)
        # driver = webdriver.Chrome(options=options)
            
        driver.get(full_url)
        
        # lazy loading으로 정상적이지 않은 크롤링 결과를 방지하기 위한 스크롤링 
        
        pause = 0.5 # 조금씩 쉴 시간
        SCROLL_STEP = 400  # 한 번에 스크롤할 픽셀 수
        lastHeight = driver.execute_script("return document.body.scrollHeight")

        while True:
            currentHeight = driver.execute_script("return window.scrollY")
            driver.execute_script(f"window.scrollBy(0, {SCROLL_STEP});")
            time.sleep(pause)
            newHeight = driver.execute_script("return window.scrollY")
            if newHeight == currentHeight:
                break

        # 초기 brand_name, product_names, product_links 추출
        brand_name, product_names, product_links, has_show_more_button = get_brand_and_product_info(driver.page_source)

        # 상품 정보를 DataFrame에 추가
        for product_name, product_link in zip(product_names, product_links):
            # csv 파일에 해당 brand_name, product_name 조합이 없는 경우만 추가
            if not ((existing_products['brand_name'] == brand_name) & 
                    (existing_products['product_name'] == product_name)).any():
                next_idx = len(df)
                df.loc[next_idx] = [brand_name, product_name, product_link]
                print(f"Retrieving information for {brand_name} - {product_name}")

        current_page = 2  # "Show More Products" 버튼이 존재할 경우 시작 페이지 번호
        while has_show_more_button:
            full_url = BASE_URL + link + f"?currentPage={current_page}"
            print(full_url)
            
            driver.get(full_url)
            
            # lazy loading으로 정상적이지 않은 크롤링 결과를 방지하기 위한 스크롤링 
        
            pause = 0.5 # 조금씩 쉴 시간
            SCROLL_STEP = 400  # 한 번에 스크롤할 픽셀 수
            lastHeight = driver.execute_script("return document.body.scrollHeight")

            while True:
                currentHeight = driver.execute_script("return window.scrollY")
                driver.execute_script(f"window.scrollBy(0, {SCROLL_STEP});")
                time.sleep(pause)
                newHeight = driver.execute_script("return window.scrollY")
                if newHeight == currentHeight:
                    break

            brand_name, product_names, product_links, has_show_more_button = get_brand_and_product_info(driver.page_source)

            # 상품 정보를 DataFrame에 추가
            for product_name, product_link in zip(product_names, product_links):
                next_idx = len(df)
                df.loc[next_idx] = [brand_name, product_name, product_link]
            
            current_page += 1  # 다음 페이지 번호로 업데이트

    return df