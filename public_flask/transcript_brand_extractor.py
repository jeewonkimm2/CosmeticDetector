from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
import Levenshtein


class TranscriptBrandExtractor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TranscriptBrandExtractor, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def fetch_transcript(youtube_url):
        video_id = youtube_url.split("v=")[1].split("&")[0]
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            filtered_transcript = [
                entry for entry in transcript if entry['text'] != '[Music]']
            return filtered_transcript
        except (TranscriptsDisabled, NoTranscriptFound):
            return []

    @staticmethod
    def find_most_similar_brands(ocr_text, brand_list):
        similar_brands = []
        threshold_similarity = 80
        ocr_text = ocr_text.replace(" ", "").replace(
            "-", "").replace("&", "").replace(".", "").replace("+", "").lower()
        for brand in brand_list:
            if brand in ocr_text:
                similar_brands.append(brand)
                continue
            distances = [Levenshtein.distance(
                brand_word, ocr_word) for brand_word in brand.split() for ocr_word in ocr_text.split()]
            distance = min(distances)
            similarity = (1 - distance / max(len(brand), len(ocr_text))) * 100
            if similarity >= threshold_similarity:
                similar_brands.append(brand)
        return similar_brands

    def extract_brand_name_entries(self, transcript_list, brand_list):
        brand_entries = []
        for entry in transcript_list:
            text = entry.get('text', '')
            similar_brands = self.find_most_similar_brands(text, brand_list)
#                 if similar_brands:
            brand_entries.append({
                'start': entry['start'],
                'duration': entry['duration'],
                'brand_names': similar_brands
            })
        return brand_entries

    def extract_brands_from_youtube_url(self, youtube_url, brand_list):
        transcripts = self.fetch_transcript(youtube_url)
        brand_entries = self.extract_brand_name_entries(
            transcripts, brand_list)
        return brand_entries
