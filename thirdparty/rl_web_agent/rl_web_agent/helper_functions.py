"""Helper functions for evaluation - adapted from WebArena to use our config system"""

import asyncio
import json
import logging
from urllib.parse import urlparse

import httpx

# Retry configuration
MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 3.0
RETRYABLE_STATUS_CODES = {502, 503, 504}


class HelperFunctions:
    """Helper functions for evaluation that use our config system"""

    def __init__(self, config, extra_headers):
        """Initialize with our config containing accounts and site URLs"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.extra_headers = extra_headers

        # Set up proxy if enabled
        self.proxy_url = None
        if config.proxy.enabled:
            self.proxy_url = config.proxy.server

    def _get_site_url(self, site_name: str) -> str:
        """Get site URL from config"""
        site_host = self.config.sites[site_name]
        return f"http://{site_host}"

    def _get_account_info(self, account_key: str) -> dict:
        """Get account info from config"""
        return self.config.accounts[account_key]

    def _log_request(self, method: str, url: str, headers: dict, body: str | None = None) -> None:
        """Log detailed request information"""
        self.logger.debug(f"HTTP Request: {method} {url}")
        self.logger.debug(f"  Headers: {json.dumps(headers, indent=2)}")
        if body:
            self.logger.debug(f"  Body: {body}")
        self.logger.debug(f"  Proxy: {self.proxy_url}")

    def _log_response(self, response: httpx.Response, attempt: int) -> None:
        """Log detailed response information"""
        self.logger.debug(f"HTTP Response (attempt {attempt}):")
        self.logger.debug(f"  Status: {response.status_code} {response.reason_phrase}")
        self.logger.debug(f"  Headers: {json.dumps(dict(response.headers), indent=2)}")
        try:
            body = response.text
            # Truncate very long responses for logging
            if len(body) > 1000:
                self.logger.debug(f"  Body (truncated): {body[:1000]}...")
            else:
                self.logger.debug(f"  Body: {body}")
        except Exception as e:
            self.logger.debug(f"  Body: <failed to read: {e}>")

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        headers: dict,
        content: str | None = None,
        params: dict | None = None,
    ) -> httpx.Response:
        """Make HTTP request with retry on 502/503/504 errors and timeouts"""
        self._log_request(method, url, headers, content)

        response: httpx.Response | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(proxy=self.proxy_url, timeout=30.0) as client:
                    if method.upper() == "GET":
                        response = await client.get(url, headers=headers, params=params)
                    elif method.upper() == "POST":
                        response = await client.post(url, headers=headers, content=content)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                self._log_response(response, attempt)

                if response.status_code in RETRYABLE_STATUS_CODES:
                    if attempt < MAX_RETRIES:
                        self.logger.warning(f"Received {response.status_code}, retrying in {RETRY_DELAY_SECONDS}s (attempt {attempt}/{MAX_RETRIES})")
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                        continue
                    else:
                        self.logger.error(f"Max retries ({MAX_RETRIES}) exceeded for {method} {url}")
                break

            except httpx.TimeoutException as e:
                if attempt < MAX_RETRIES:
                    self.logger.warning(f"Request timed out: {e}, retrying in {RETRY_DELAY_SECONDS}s (attempt {attempt}/{MAX_RETRIES})")
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    self.logger.error(f"Max retries ({MAX_RETRIES}) exceeded for {method} {url} due to timeout")
                    raise

        assert response is not None, "No response received after retries"
        return response

    async def shopping_get_auth_token(self) -> str:
        """Get shopping site auth token"""
        shopping_url = self._get_site_url("shopping")
        admin_account = self._get_account_info("shopping_shopping_admin")

        headers = {"content-type": "application/json"}
        headers.update(self.extra_headers)

        body = json.dumps(
            {
                "username": admin_account["username"],
                "password": admin_account["password"],
            }
        )

        response = await self._request_with_retry(
            method="POST",
            url=f"{shopping_url}/rest/default/V1/integration/admin/token",
            headers=headers,
            content=body,
        )
        response.raise_for_status()
        token: str = response.json()
        return token

    async def shopping_get_latest_order_url(self) -> str:
        """Get the latest order url from the shopping website."""
        shopping_url = self._get_site_url("shopping")
        token = await self.shopping_get_auth_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        params = {
            "searchCriteria[sortOrders][0][field]": "created_at",
            "searchCriteria[sortOrders][0][direction]": "DESC",
            "searchCriteria[pageSize]": "1",
            "searchCriteria[filter_groups][0][filters][0][field]": "customer_id",
            "searchCriteria[filter_groups][0][filters][0][value]": "27",
        }

        response = await self._request_with_retry(
            method="GET",
            url=f"{shopping_url}/rest/V1/orders",
            headers=headers,
            params=params,
        )
        response.raise_for_status()

        response_obj = response.json()
        order_item = response_obj["items"][0]
        order_id = int(order_item["increment_id"])
        order_url = f"{shopping_url}/sales/order/view/order_id/{order_id}/"
        return order_url

    async def shopping_get_sku_latest_review_author(self, sku: str) -> str:
        """Get the latest review author for a product SKU."""
        shopping_url = self._get_site_url("shopping")
        token = await self.shopping_get_auth_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        response = await self._request_with_retry(
            method="GET",
            url=f"{shopping_url}/rest/V1/products/{sku}/reviews",
            headers=headers,
        )
        response.raise_for_status()

        response_obj = response.json()
        if len(response_obj) == 0:
            return ""
        author: str = response_obj[-1]["nickname"]
        return author

    async def shopping_get_sku_latest_review_rating(self, sku: str) -> str:
        """Get the latest review rating for a product SKU."""
        shopping_url = self._get_site_url("shopping")
        token = await self.shopping_get_auth_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        response = await self._request_with_retry(
            method="GET",
            url=f"{shopping_url}/rest/V1/products/{sku}/reviews",
            headers=headers,
        )
        response.raise_for_status()

        response_obj = response.json()
        if len(response_obj) == 0:
            return ""
        assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
        latest_review = response_obj[-1]
        rating: str = str(latest_review["ratings"][0]["percent"])
        return rating

    def reddit_get_post_url(self, url: str) -> str:
        """Get the post url from a Reddit comment/post URL"""
        # Url is http://domain/f/subreddit/post_id/...
        # get domain, subreddit, post_id
        parsed = urlparse(url)
        domain = parsed.netloc
        tok_url = parsed.path.split("/")

        # Validate URL structure - return original URL if invalid
        if len(tok_url) < 4:
            return url
        if tok_url[1] != "f":
            return url

        subreddit = tok_url[2]
        post_id = tok_url[3]
        scheme = parsed.scheme
        post_url = f"{scheme}://{domain}/f/{subreddit}/{post_id}/"
        return post_url

    async def gitlab_get_project_member_role(self, page, account_name: str) -> str:
        """Get project member role from GitLab page (async version)"""
        try:
            # get the account index
            account_idx = await page.evaluate(
                f"""(() => {{
                    const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                    let index = -1;  // Default value if not found

                    for(let i = 0; i < elements.length; i++) {{
                        if(elements[i].outerText === '@{account_name}') {{
                            index = i;
                            break;
                        }}
                    }}

                    return index;
                }})()"""
            )

            if account_idx == -1:
                return ""

            # get the role
            role: str = await page.evaluate(
                f"""(() => {{
                    const roleElements = document.querySelectorAll("td.col-max-role span");
                    return roleElements[{account_idx}].outerText;
                }})()"""
            )
            return role
        except Exception:
            return ""

    async def llm_fuzzy_match(self, pred: str, reference: str, question: str) -> float:
        """Use LiteLLM for fuzzy matching evaluation"""
        # Empty prediction can't match — no need to call LLM
        if not pred.strip():
            return 0.0

        import litellm

        # Get evaluator LLM model from environment config
        model = self.config.evaluator_llm.model

        # Load prompt from file
        from rl_web_agent.prompts import load_prompt

        user_prompt = load_prompt("fuzzy_match_evaluator").format(question=question, reference=reference, pred=pred)

        messages = [{"role": "user", "content": user_prompt}]

        # Call LiteLLM async completion (API keys read from environment automatically)
        response = await litellm.acompletion(
            model=model,
            messages=messages,
        )

        content = response.choices[0].message.content
        response_lower = content.lower().strip()

        if "partially correct" in response_lower or "partially_correct" in response_lower or "incorrect" in response_lower:
            return 0.0
        elif "correct" in response_lower:
            return 1.0
        else:
            # Fail fast - don't provide fallbacks for unclear responses
            raise ValueError(f"Unclear LLM response for fuzzy match: {content}")

    async def llm_ua_match(self, pred: str, reference: str, question: str) -> float:
        """Use LiteLLM for unachievable task matching"""
        # Empty prediction can't match — no need to call LLM
        if not pred.strip():
            return 0.0

        import litellm

        # Get evaluator LLM model from environment config
        model = self.config.evaluator_llm.model

        # Load prompt from file
        from rl_web_agent.prompts import load_prompt

        user_prompt = load_prompt("ua_match_evaluator").format(question=question, reference=reference, pred=pred)

        messages = [{"role": "user", "content": user_prompt}]

        # Call LiteLLM async completion (API keys read from environment automatically)
        response = await litellm.acompletion(
            model=model,
            messages=messages,
        )

        content = response.choices[0].message.content
        response_lower = content.lower().strip()

        if "different" in response_lower:
            return 0.0
        elif "same" in response_lower:
            return 1.0
        else:
            # Fail fast - don't provide fallbacks for unclear responses
            raise ValueError(f"Unclear LLM response for UA match: {content}")


def get_helper_functions(config, extra_headers) -> HelperFunctions:
    """Create a new helper functions instance with config and headers"""
    return HelperFunctions(config, extra_headers or {})


async def shopping_get_latest_order_url(config=None, extra_headers=None) -> str:
    """Global function for backward compatibility"""
    helper = get_helper_functions(config, extra_headers or {})
    return await helper.shopping_get_latest_order_url()


async def shopping_get_sku_latest_review_author(sku: str, config=None, extra_headers=None) -> str:
    """Global function for backward compatibility"""
    helper = get_helper_functions(config, extra_headers or {})
    return await helper.shopping_get_sku_latest_review_author(sku)


async def shopping_get_sku_latest_review_rating(sku: str, config=None, extra_headers=None) -> str:
    """Global function for backward compatibility"""
    helper = get_helper_functions(config, extra_headers or {})
    return await helper.shopping_get_sku_latest_review_rating(sku)


def reddit_get_post_url(url: str, config=None, extra_headers=None) -> str:
    """Global function for backward compatibility"""
    helper = get_helper_functions(config, extra_headers or {})
    return helper.reddit_get_post_url(url)


async def gitlab_get_project_member_role(page, account_name: str, config=None, extra_headers=None) -> str:
    """Global function for backward compatibility"""
    helper = get_helper_functions(config, extra_headers or {})
    return await helper.gitlab_get_project_member_role(page, account_name)
