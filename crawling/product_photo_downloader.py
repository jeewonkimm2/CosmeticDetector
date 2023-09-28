import os
import requests
import pandas as pd
import re

# 폴더 이름 생성
def make_directory_name_safe(name):
    # 대문자 앞에 공백 추가
    name_with_spaces = re.sub(r'(?<=[a-z])([A-Z])', r' \1', name).strip()

    # 특수문자를 제거하거나 대체
    safe_name = ''.join(e for e in name_with_spaces if e.isalnum() or e.isspace())

    # 연속된 공백을 하나의 공백으로 대체
    safe_name = re.sub(r"\s+", " ", safe_name).strip()
    
    return safe_name

def load_existing_products(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename, encoding='utf-8-sig')
    else:
        return pd.DataFrame(columns=['brand_name', 'product_name'])
    
# photo url에서 특정 위치에 사진 저장
def download_image_from_url(url, save_path):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"
    }
    response = requests.get(url=url, headers=headers, timeout=10)
    data = response.content

    with open(save_path, 'wb') as f:
        f.write(data)

# df의 정보를 이용해서 이미지 데이터 최종 생성
def save_product_images_from_df(df):
    BASE_DIR = 'product_images'
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

    for index, row in df.iterrows():
        brand_name = row['brand_name']
        product_name = row['product_name']
        photo_urls = row['product_photo_urls']

        safe_brand_name = make_directory_name_safe(brand_name)
        safe_product_name = make_directory_name_safe(product_name)

        brand_dir = os.path.join(BASE_DIR, safe_brand_name)
        product_dir = os.path.join(brand_dir, safe_product_name)

        if not os.path.exists(brand_dir):
            os.mkdir(brand_dir)
        if not os.path.exists(product_dir):
            os.mkdir(product_dir)

        for idx, url in enumerate(photo_urls):
            img_path = os.path.join(product_dir, f"{idx}.jpg")
            download_image_from_url(url, img_path)

# product_list를 csv형태로 저장
def save_to_csv(df, filename):
    # 파일이 이미 존재하면 파일을 불러옴
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename, encoding='utf-8-sig')
        
        # 변경된 컬럼 정보를 반영하기 위해 저장할 데이터 구성
        save_df = df.copy()
        save_df['product_photo_url'] = df['product_photo_urls'].apply(lambda x: x[0] if x else None)
        combined_df = pd.concat([existing_df, save_df[['brand_name', 'product_name', 'product_detail_url', 'product_price', 'product_photo_url']]], ignore_index=True)
    else:
        # 새로운 파일 생성하는 경우
        df['product_photo_url'] = df['product_photo_urls'].apply(lambda x: x[0] if x else None)
        combined_df = df[['brand_name', 'product_name', 'product_detail_url', 'product_price', 'product_photo_url']]
        
    # 중복 데이터 제거
    combined_df = combined_df.drop_duplicates()
    
    # CSV 파일로 저장
    combined_df.to_csv(filename, index=False, encoding='utf-8-sig')