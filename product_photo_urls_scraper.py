import requests
from bs4 import BeautifulSoup
import pandas as pd

# detail_url에서 product photo url scraping
def get_product_photo_urls(product_detail_url):
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
    response = requests.get(url=product_detail_url, headers=headers, timeout=10)

    soup = BeautifulSoup(response.content, 'lxml')
    picture_elements = soup.select("picture.css-4rx3j2 > source")

    photo_urls = []

    for element in picture_elements:
        srcset = element.get('srcset')

        photo_url = srcset.split('?')[0].split(' ')[0]
        photo_urls.append(photo_url)

    return photo_urls

# 모든 정보가 저장된 df 생성
def create_dataframe_with_photos(df):
    new_columns = ['brand_name', 'product_name', 'product_detail_url', 'product_photo_urls']
    new_df = pd.DataFrame(columns=new_columns)

    for index, row in df.iterrows():
        product_name = row['product_name']
        brand_name = row['brand_name']
        product_detail_url = row['product_detail_url']
        
        photo_urls = get_product_photo_urls(product_detail_url)
        
        # photo_urls의 요소가 1개 이상일 경우에만 new_df에 추가
        if len(photo_urls) > 0:
            new_df.loc[index] = [brand_name, product_name, product_detail_url, photo_urls]
        
    return new_df