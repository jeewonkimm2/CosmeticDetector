from brand_urls_scraper import fetch_brand_urls
from products_list_loader import load_existing_products
from brand_products_scraper import fetch_brand_and_product_info
from product_photo_urls_scraper import create_dataframe_with_photos_and_price
from product_photo_downloader import save_to_csv, save_product_images_from_df

import logging

# 로깅 설정
logging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)

brand_urls = fetch_brand_urls()
existing_products = load_existing_products("products_list.csv")

for key in brand_urls:
    try:
        brand_links = brand_urls[key]
        brand_info_df = fetch_brand_and_product_info(brand_links, existing_products)

        new_df = create_dataframe_with_photos_and_price(brand_info_df)

    except Exception as e:
        logging.error(f"Error occurred while processing key {key} during fetch and creation: {e}")
        continue

    try:
        save_product_images_from_df(new_df)
        save_to_csv(new_df, "products_list.csv")
    except Exception as e:
        logging.error(f"Error occurred while processing key {key} during saving: {e}")
        continue