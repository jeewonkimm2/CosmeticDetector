import os
import pandas as pd

def load_existing_products(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename, encoding='utf-8-sig')
    else:
        return pd.DataFrame(columns=['brand_name', 'product_name'])