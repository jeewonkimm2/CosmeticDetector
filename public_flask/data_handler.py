import csv


class DataHandler:
    def __init__(self):
        pass

    def read_csv_to_list(self, csv_path):
        product_info_list = []
        with open(csv_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                product_info_list.append(row)

        return product_info_list

    def get_matching_products(self, response_data, product_info_list):
        matching_products = []

        if not response_data:
            return "No cosmetic product found."

        # brand_name과 product_name이 일치하는 항목 찾기
        for product_info in product_info_list:
            if (product_info['brand_name'].lower() == response_data['brand_name'].lower() and
                    product_info['product_name'].lower() == response_data['product_name'].lower()):
                matching_products.append(product_info)

        return matching_products if matching_products else "No matching product found in the list."
