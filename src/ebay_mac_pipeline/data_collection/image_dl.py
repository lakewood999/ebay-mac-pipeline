import asyncio, aiohttp, os, datetime, signal
from aiolimiter import AsyncLimiter

EBAY_IMAGE_PREFIX = "https://i.ebayimg.com/images/g"
EBAY_IMAGE_FILE = "s-l1600.jpg"

class ImageDownloadClient:
    """
    Rate-limited async client for downloading images from eBay given
    the image ID
    """
    def __init__(self, rate_limit: int = 60, rate_limit_period: int = 60):
        self.limiter = AsyncLimiter(rate_limit, rate_limit_period)
        self.http_client = aiohttp.ClientSession()

    async def download_image(self, id: str):
        """
        Basic async function to download an image from eBay given
        the image ID. The image ID is the part of the URL after
        "https://i.ebayimg.com/images/g/" and before "/s-l1600.jpg".
        Assumes the largest image size (as naturally occurs on eBay) in JPG
        format is desired.

        :param id: The image ID to download.
        """
        await self.limiter.acquire()
        image_url = f"{EBAY_IMAGE_PREFIX}/{id}/{EBAY_IMAGE_FILE}"
        response = await self.http_client.get(image_url)
        if response.status == 200:
            return await response.read()
        else:
            print(f"Failed to download image for {id}")

    async def close(self):
        await self.http_client.close()
