from django.http import JsonResponse
from django.shortcuts import render
from .scene_change import divide_video
import json

def get_url(request):
    return render(request, 'image_processing/get_url.html')

def process_video(request):
    if request.method == "POST":
        data = json.loads(request.body)
        url = data.get("url")

        transition_times = divide_video(url)

        print(transition_times)

        return JsonResponse({"status": "success"})
    else:
        return JsonResponse({"status:": "error"})
