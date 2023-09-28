import requests
from bs4 import BeautifulSoup

def fetch_brand_urls():
    URL = "https://www.sephora.com/brands-list"
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"
    }
    
    response = requests.get(url=URL, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 해당 <div> 요소 내의 모든 <a> 태그를 선택
    links = soup.select("div.css-166dgmb.eanm77i0[data-comp='StyledComponent BaseComponent '] > a")

    brand_urls = {}

    # 각 <a> 태그의 텍스트를 이용하여 해당 div를 찾은 다음 그 안의 모든 <a> 태그의 href를 가져와서 dictionary에 저장
    for link in links:
        div_id = "brands-" + link.text
        if link.text == "#":
            brand_links = soup.select(f'div[id="brands-#"] > div > ul > li > a')
        else:
            brand_links = soup.select(f"div#{div_id} > div > ul > li > a")

        brand_urls[link.text] = [brand_link['href'] for brand_link in brand_links]

    return brand_urls