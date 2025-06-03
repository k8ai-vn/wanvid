# import requests

# invoke_url = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"

# headers = {
#     "Authorization": "Bearer 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",
#     "Accept": "application/json",
# }

# payload = {
#     "prompt": "The pyramid in the sea",
#     "mode": "canny",
#     "image": "data:image/png;example_id,3",
#     "cfg_scale": 3.5,
#     "width": 1024,
#     "height": 1024,
#     "seed": 0,
#     "steps": 50
# }

# response = requests.post(invoke_url, headers=headers, json=payload)

# response.raise_for_status()
# response_body = response.json()
# print(response_body)
# print(response_body)


import json
import base64

try:
    with open("response.json", "r") as f:
        response_body = json.load(f)
        
        # Save all 6 images
        for i in range(1,7):
            image_key = f"image{i}"
            if image_key in response_body:
                print(f"Saving {image_key}")
                # Save each base64 image to a separate file
                with open(f"image{i}.png", "wb") as img_file:
                    img_file.write(base64.b64decode(response_body[image_key]))
            else:
                print(f"Could not find {image_key} in response")

except json.decoder.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except FileNotFoundError:
    print("response.json file not found")
except KeyError as e:
    print(f"Key error accessing JSON response: {e}")