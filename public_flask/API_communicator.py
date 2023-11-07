import requests
import json


class APICommunicator:
    def __init__(self):
        pass

    def send_post_request_to_api(self, api_url, save_path, class_names):
        data_to_send = {'path': save_path, 'class_names': class_names}
        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(
                api_url, data=json.dumps(data_to_send), headers=headers)
            response_data = response.json()
        except Exception as e:
            print(
                f"An error occurred while sending the POST request: {str(e)}")
            response_data = {}
        return response_data
