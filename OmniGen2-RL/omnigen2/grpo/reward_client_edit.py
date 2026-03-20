#!/usr/bin/env python3
"""
Pure Reward Client - Only responsible for data transmission
"""

import base64
from io import BytesIO
import requests
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

class RewardClient:
    """
    Pure Reward Client - Only responsible for communicating with proxy server
    """
    
    def __init__(self, proxy_host: str = "127.0.0.1", proxy_port: int = 23456, 
                 timeout: int = 300, max_retries: int = 3):
        """
        Initialize client
        
        Args:
            proxy_host: Proxy server host address
            proxy_port: Proxy server port
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.proxy_url = f"http://{proxy_host}:{proxy_port}"
        self.timeout = timeout
        self.max_retries = max_retries
        
        logger.info(f"Initialize Reward client: {self.proxy_url}")

    @staticmethod
    def _encode_image(image_obj: Any) -> str:
        """Encode PIL image/bytes to base64 PNG string for safe transport."""
        if isinstance(image_obj, str):
            return image_obj
        if isinstance(image_obj, (bytes, bytearray)):
            return base64.b64encode(image_obj).decode("ascii")
        if not isinstance(image_obj, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image_obj)}")

        buffer = BytesIO()
        image_obj.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _build_safe_request_payload(
        self,
        input_images: List[Any],
        output_image: List[Any],
        meta_datas: List[Dict[str, Any]],
        server_type: str,
    ) -> Dict[str, Any]:
        encoded_input_images = []
        for sample in input_images:
            if not isinstance(sample, list):
                raise TypeError(f"Each input_images item must be a list, got: {type(sample)}")
            encoded_input_images.append([self._encode_image(img) for img in sample])

        encoded_output_images = [self._encode_image(img) for img in output_image]
        return {
            "input_images": encoded_input_images,
            "output_image": encoded_output_images,
            "meta_datas": meta_datas,
            "server_type": server_type,
        }
    
    def evaluate(self, input_images: List[bytes], output_image: List[bytes], meta_datas: List[Dict[str, Any]], 
                 server_type: str = 'geneval') -> Optional[Tuple[List[float], List[float], List[str], List[Dict]]]:
        """
        Evaluate images and return rewards
        
        Args:
            input_images: List of input image byte data
            output_image: List of output image byte data
            meta_datas: List of metadata
            server_type: Server type ('geneval', 'ocr', etc.)
            
        Returns:
            tuple: (scores, rewards, reasoning, meta_data) 
            - scores: List of scores
            - rewards: List of rewards
            - reasoning: List of reasoning results
            - meta_data: List of metadata
        """
        if not output_image:
            return [], [], [], []
        
        # Prepare request data
        request_data = self._build_safe_request_payload(
            input_images=input_images,
            output_image=output_image,
            meta_datas=meta_datas,
            server_type=server_type,
        )
        
        # Retry logic
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.proxy_url,
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    # Parse results
                    result = response.json()
                    scores = result.get('scores', [])
                    rewards = result.get('rewards', [])
                    reasoning = result.get('reasoning', [])
                    meta_data = result.get('meta_data', [])
                    
                    # Basic validation
                    if len(scores) != len(output_image) or len(rewards) != len(output_image):
                        logger.warning(f"Return data length mismatch: expected {len(output_image)}, got scores={len(scores)}, rewards={len(rewards)}")
                    
                    return scores, rewards, reasoning, meta_data
                else:
                    logger.error(f"HTTP error: {response.status_code}")
                    last_exception = RuntimeError(f"HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout as e:
                logger.error(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                last_exception = e
                
            except Exception as e:
                logger.error(f"Request exception: {e} (attempt {attempt + 1}/{self.max_retries})")
                last_exception = e
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        
        logger.error(f"All retries failed, last exception: {last_exception}")
        return None
    
    def ping(self) -> bool:
        """Check if server is reachable"""
        try:
            response = requests.get(f"{self.proxy_url}/ping", timeout=5)
            return response.status_code == 200
        except:
            return False

# Convenience function
def evaluate_images(input_images: List[bytes], output_image: List[bytes], meta_datas: List[Dict[str, Any]], 
                   proxy_host: str = "127.0.0.1", proxy_port: int = 23456,
                   server_type: str = 'vlm') -> Optional[Tuple[List[float], List[float], List[str], List[Dict]]]:
    """
    Convenience function: directly evaluate images
    """
    client = RewardClient(proxy_host, proxy_port, timeout=600, max_retries=1)
    return client.evaluate(input_images, output_image, meta_datas, server_type)

# Usage example
if __name__ == "__main__":
    # Create client
    client = RewardClient()
    
    # Check connection
    if not client.ping():
        print("❌ Server unreachable")
        exit(1)
    
    # Mock data
    input_images = [b"fake_input_image"]  # Input images
    output_images = [b"fake_output_image"]  # Output images
    meta_datas = [{"tag": "test", "prompt": "a simple test"}]  # Metadata
    
    # Evaluate images
    print("🔥 Evaluating images:")
    result = client.evaluate(input_images, output_images, meta_datas, server_type='vlm')
    if result:
        scores, rewards, reasoning, meta_data = result
        print(f"Scores: {scores}")
        print(f"Rewards: {rewards}")
        print(f"Reasoning: {reasoning}")
        print(f"Meta data: {meta_data}")
    else:
        print("Evaluation failed")
