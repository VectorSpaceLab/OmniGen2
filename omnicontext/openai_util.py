import requests
import base64
from io import BytesIO
from PIL import Image, ImageOps
from typing import Union, Optional, Tuple, List
import os

def encode_pil_image(pil_image):
    # Create an in-memory binary stream
    image_stream = BytesIO()
    
    # Save the PIL image to the binary stream in JPEG format (you can change the format if needed)
    pil_image.save(image_stream, format='JPEG')
    
    # Get the binary data from the stream and encode it as base64
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    return base64_image

def load_image(image: Union[str, Image.Image], format: str = "RGB", size: Optional[Tuple] = None) -> Image.Image:
    """
    Load an image from a given path or URL and convert it to a PIL Image.

    Args:
        image (Union[str, Image.Image]): The image path, URL, or a PIL Image object to be loaded.
        format (str, optional): Desired color format of the resulting image. Defaults to "RGB".
        size (Optional[Tuple], optional): Desired size for resizing the image. Defaults to None.

    Returns:
        Image.Image: A PIL Image in the specified format and size.

    Raises:
        ValueError: If the provided image format is not recognized.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = ImageOps.exif_transpose(image)
    image = image.convert(format)
    if (size != None):
        image = image.resize(size, Image.LANCZOS)
    return image

def prepare_prompt(image_links: List = [], text_prompt: str = ""):
    prompt_content = []
    text_dict = {"type": "text", "text": text_prompt}
    prompt_content.append(text_dict)

    if not isinstance(image_links, list):
        image_links = [image_links]
    for image_link in image_links:
        image = load_image(image_link)
        visual_dict = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_image(image)}"},
        }
        prompt_content.append(visual_dict)
    return prompt_content

def ask_gpt4o(image_path, prompt, url, api_key):
    prompt = prepare_prompt(image_path, prompt)
    payload = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1400,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=180)  # Set timeout to 5 minutes (300 seconds)
    except Exception as e:
        print(f"Error: {e}")
        return ""
    return extract_response(response)

def extract_response(response):
    try:
        response = response.json()
        out = response["choices"][0]["message"]["content"]
        return out
    except:
        if response["error"]["code"] == "content_policy_violation":
            print("Code is content_policy_violation")
        elif response["error"]["code"] in [
            "rate_limit_exceeded",
            "insufficient_quota",
            "insufficient_user_quota",
        ]:
            print(f"Code is {response['error']['code']}", flush=True)
            print(response["error"]["message"], flush=True)
            return "rate_limit_exceeded"
        else:
            print("Code is different")
            print(response)
            print(f"{response['error']['code']=}")
    return ""