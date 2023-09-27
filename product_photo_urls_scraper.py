import requests
from bs4 import BeautifulSoup
import pandas as pd

# detail_url에서 product photo url scraping
def get_product_details(product_detail_url):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"
    }
    response = requests.get(url=product_detail_url, headers=headers, timeout=10)

    soup = BeautifulSoup(response.content, 'lxml')
    picture_elements = soup.select("picture.css-4rx3j2 > source")

    photo_urls = [element.get('srcset').split('?')[0].split(' ')[0] for element in picture_elements]

    # 만약 photo_urls의 element가 하나도 없다면 다른 element에서 url 가져오기
    if not photo_urls:
        div_element = soup.find('div', class_='css-1a2dflv eanm77i0')
        if div_element:
            img_element = div_element.find('img', class_='css-1rovmyu eanm77i0')
            photo_urls.append(img_element['src'].split('?')[0])

    # 가격 정보 가져오기
    price_element = soup.select_one("span.css-18jtttk > b.css-0")
    price = price_element.text if price_element else None

    return photo_urls, price

def create_dataframe_with_photos_and_price(df):
    new_columns = ['brand_name', 'product_name', 'product_detail_url', 'product_photo_urls', 'product_price']
    new_df = pd.DataFrame(columns=new_columns)

    for index, row in df.iterrows():
        product_name = row['product_name']
        brand_name = row['brand_name']
        product_detail_url = row['product_detail_url']

        photo_urls, price = get_product_details(product_detail_url)

        # photo_urls의 요소가 1개 이상일 경우에만 new_df에 추가
        if len(photo_urls) > 0:
            new_df.loc[index] = [brand_name, product_name, product_detail_url, photo_urls, price]

    return new_df