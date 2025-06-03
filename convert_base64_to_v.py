# Get result in video.json  and convert base64 to mp4
import json
import base64

# Read the video.json file
with open('video.json', 'r') as json_file:
    video_data = json.load(json_file)

# Get the base64 string from the JSON
base64_video = video_data['result']['video']  # Adjust key name if different

# Decode base64 string to bytes
video_bytes = base64.b64decode(base64_video)

# Write bytes to mp4 file
with open('output.mp4', 'wb') as video_file:
    video_file.write(video_bytes)

print("Video has been successfully converted and saved as output.mp4")
