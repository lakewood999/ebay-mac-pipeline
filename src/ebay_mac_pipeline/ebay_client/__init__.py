import aiohttp
from datetime import datetime, timedelta
from urllib.parse import quote
from base64 import b64encode
from typing import List, Dict
from . import types


class EbayClientAsync:
    def __init__(self, client_id: str, client_secret: str, runame: str):
        """
        Initialize the EbayClientAsync object and authenticate it with the provided credentials.

        :param client_id: The client ID of the application. From eBay API credentials.
        :param client_secret: The client secret of the application. From eBay API credentials.
        :param runame: The runame of the application. From eBay API credentials.
        """
        # Static properties of the client
        self._client = aiohttp.ClientSession()
        self._client_id = client_id
        self._client_secret = client_secret
        self._runame = runame
        # Auth token
        self._auth_token = None
        self._auth_token_expiry = None

    async def auth_token(self):
        """
        Get the auth token of the client, re-authenticate if needed.
        """
        current_time = datetime.now()
        if (
            self._auth_token is None
            or self._auth_token_expiry is None
            or current_time > self._auth_token_expiry
        ):
            await self.authenticate()
        return self._auth_token

    async def authenticate(self):
        """
        Authenticate the client with the provided credentials.
        """
        if not (self._auth_token is None):
            raise Exception("Client is already authenticated.")

        # Generate the auth request
        auth_token = b64encode(
            bytes(f"{self._client_id}:{self._client_secret}", "utf-8")
        )
        auth_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {auth_token.decode('utf-8')}",
        }
        auth_data = {
            "grant_type": "client_credentials",
            "redirect_uri": self._runame,
            "scope": "https://api.ebay.com/oauth/api_scope",
        }

        results = await self._client.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers=auth_headers,
            data=auth_data,
        )
        
        # Parse the results
        if results.status != 200:
            body = await results.text()
            raise Exception(
                f"Failed to authenticate client with HTTP code {results.status} and body {body}."
            )
        else:
            body = await results.json()
            self._auth_token = body["access_token"]
            # Compute the expiry
            self._auth_token_expiry = datetime.now() + timedelta(0, body["expires_in"])

    async def item(
        self,
        item_id: str,
        fieldgroups: List[types.FieldGroup] = [],
        marketplace: types.MarketplaceId = types.MarketplaceId.EBAY_US,
    ):
        # API Authentication
        auth_token = await self.auth_token()
        item_headers = {
            "Authorization": f"Bearer {auth_token}",
            "X-EBAY-C-MARKETPLACE-ID": marketplace.value,
        }
        # Build the search
        params = {}
        if fieldgroups:
            params["fieldgroups"] = ",".join([f.value for f in fieldgroups])
        get_params = "&".join([f"{quote(k)}={v}" for k, v in params.items()])
        item_url = f"https://api.ebay.com/buy/browse/v1/item/{item_id}?{get_params}"
        
        # Make the API call
        results = await self._client.get(item_url, headers=item_headers)
        # Parse the results
        if results.status != 200:
            body = await results.text()
            raise Exception(
                f"Failed to get item {item_id} with HTTP code {results.status} and body {body}."
            )
        else:
            body = await results.json()
            return body

    async def search(
        self,
        query: str,
        fieldgroups: List[types.FieldGroup] = [],
        filters: Dict[types.FilterGroup, str] = {},
        sort: List[types.SortField] = [],
        category_ids: List[int] = [],
        aspect_filter: Dict[str, List[str]] = {},
        limit: int = 200,
        offset: int = 0,
        marketplace: types.MarketplaceId = types.MarketplaceId.EBAY_US,
    ):
        """
        Search for the provided query on eBay.

        Currently supports: q, fieldgroups, filter, sort, limit, offset,
        Does not support: gtin, charity_ids, compatibility_filter,
        auto_correct, category_ids, aspect_filter, epid
        """
        # API Authentication
        auth_token = await self.auth_token()
        search_headers = {
            "Authorization": f"Bearer {auth_token}",
            "X-EBAY-C-MARKETPLACE-ID": marketplace.value,
        }
        # Build the search
        params = {"q": query, "limit": limit, "offset": offset}
        if fieldgroups:
            params["fieldgroups"] = ",".join([f.value for f in fieldgroups])
        if sort:
            params["sort"] = ",".join([f.value for f in sort])
        if category_ids:
            params["category_ids"] = ",".join([str(i) for i in category_ids])
        if filters:
            parsed = []
            for k, v in filters.items():
                parsed.append(f"{quote(k.value)}:{quote(v)}")
            params["filter"] = ",".join(parsed)
        if aspect_filter:
            if not category_ids:
                raise Exception("Aspect filters require category_ids to be set")
            parsed = []
            parsed.append(
                f"categoryId:{quote('|'.join([str(i) for i in category_ids]))}"
            )
            for k, v in aspect_filter.items():
                parsed.append(f"{quote(k)}:{quote('{' + ('|'.join(v))) + '}'}")
            params["aspect_filter"] = ",".join(parsed)
        get_params = "&".join([f"{quote(k)}={v}" for k, v in params.items()])
        search_url = (
            f"https://api.ebay.com/buy/browse/v1/item_summary/search?{get_params}"
        )
        # Make the API call
        results = await self._client.get(search_url, headers=search_headers)
        # Parse the results
        if results.status != 200:
            body = await results.text()
            raise Exception(
                f"Failed to search for {query} with HTTP code {results.status} and body {body}."
            )
        else:
            body = await results.json()
            return body

    async def close(self):
        """
        Close the client properly.
        """
        await self._client.close()
