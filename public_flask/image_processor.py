import os
import shutil
import pandas as pd


class ImageProcessor:
    def __init__(self):
        pass

    def extract_unique_brands(csv_path):
        df = pd.read_csv(csv_path)
        brand_list = df['brand_name'].unique().tolist()
        brand_list = [brand.replace(" ", "").replace(
            "-", "").replace("&", "").replace(".", "").replace("+", "").lower() for brand in brand_list]
        return brand_list

    def extract_time_from_filename(self, filename, transition_times):
        try:
            # Extracts index from filename like 'frame_0-1.jpg'
            index_str = filename.split('_')[1].split('-')[0]
            index = int(index_str)
            return transition_times[index]
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            return None

    def find_brand_name_in_transcript(self, time, transcript):
        if time is None:
            print("Error: Time value is None.")
            return []
        for entry in transcript:
            start, duration = entry['start'], entry['duration']
            if start <= time <= start + duration:
                return entry['brand_names']
        return []

    def update_brand_in_image(self, brand_in_image, brand_in_transcript, transition_times):
        for image_entry in brand_in_image:
            time = self.extract_time_from_filename(
                image_entry['filename'], transition_times)
            additional_brands = self.find_brand_name_in_transcript(
                time, brand_in_transcript)
            if additional_brands:
                image_entry['brand_name'] = (
                    [image_entry['brand_name']] if isinstance(
                        image_entry['brand_name'], str) else image_entry['brand_name']
                )
                image_entry['brand_name'].extend(additional_brands)
        return brand_in_image

    def copy_images_to_new_folder(self, brand_with_image, source_folder, dest_folder):
        for item in brand_with_image:
            filename = item['filename']
            brand_name = item['brand_name']

            src_path = os.path.join(source_folder, filename)

            new_folder_path = os.path.join(dest_folder, "APE_inference", "data", "cosmetic",
                                           "images", "test", brand_name, "test_" + brand_name)
            os.makedirs(new_folder_path, exist_ok=True)
            dest_path = os.path.join(new_folder_path, filename)
            shutil.copy2(src_path, dest_path)

    def delete_except_APE_folder(self, save_path):
        for item in os.listdir(save_path):
            item_path = os.path.join(save_path, item)

            if item != 'APE_inference':
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
