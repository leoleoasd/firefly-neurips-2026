import asyncio
import logging
import uuid
from pathlib import Path
from typing import ClassVar

import httpx
from omegaconf import DictConfig, OmegaConf
from playwright.async_api import Playwright, async_playwright


class WebAgentEnv:
    _shared_playwright: ClassVar[Playwright | None] = None
    _shared_playwright_users: ClassVar[int] = 0

    def __init__(self, environment_config: DictConfig, client_session_id: str | None = None):
        self.config = environment_config
        self.context_manager = None
        self.browser = None
        self.context = None
        self.page = None  # Current active page
        # Note: pages are managed by self.context.pages
        self.uuid = environment_config.uuid if hasattr(environment_config, "uuid") else str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        self.task_config: dict | None = None
        self.server_ips: dict[str, str] = {}  # Mapping of site name to server IP
        self.model_answer: str | None = None  # Model's final answer/response
        self.extra_headers: dict[str, str] = {}  # Host rewrite headers for proxy
        self.launched_containers: list[str] = []  # Track containers launched for this environment
        self.trace_file_path: str | None = None  # Path to the current trace file
        # Client session ID for proxy stickiness (ensures all requests go to same backend)
        self.client_session_id = client_session_id if client_session_id else str(uuid.uuid4())

        # Disable evaluation if recording is enabled
        if self.config.recording.enabled:
            self.config.evaluation.enabled = False
            self.logger.warning("Recording is enabled - evaluation disabled to avoid interference")

    @classmethod
    async def _ensure_playwright(cls) -> Playwright:
        """Ensure shared Playwright instance exists and return it"""
        if cls._shared_playwright is None:
            cls._shared_playwright = await async_playwright().start()
        cls._shared_playwright_users += 1
        return cls._shared_playwright

    @classmethod
    async def _cleanup_playwright(cls) -> None:
        """Cleanup shared Playwright instance if no more users"""
        cls._shared_playwright_users -= 1
        if cls._shared_playwright_users == 0 and cls._shared_playwright is not None:
            await cls._shared_playwright.stop()
            cls._shared_playwright = None

    async def _setup_tracing(self) -> None:
        """Setup Playwright tracing if enabled in config"""
        if not self.config.tracing.enabled:
            self.logger.debug("Tracing not enabled")
            return

        # Use trace output path directly as file path
        self.trace_file_path = self.config.tracing.output_path

        # Start tracing with configured options
        await self.context.tracing.start(
            screenshots=self.config.tracing.get("screenshots", True),
            snapshots=self.config.tracing.get("snapshots", True),
            sources=self.config.tracing.get("sources", True),
        )

        self.logger.info(f"Tracing started, will save to: {self.trace_file_path}")

    async def _stop_tracing(self) -> None:
        """Stop tracing and save trace file"""
        if self.trace_file_path and self.context:
            try:
                await self.context.tracing.stop(path=self.trace_file_path)
                self.logger.info(f"Trace saved to: {self.trace_file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save trace: {e}")
            finally:
                self.trace_file_path = None

    async def _start_recording(self) -> None:
        """Start screen recording using QuickRecorder"""
        try:
            self.logger.info("Starting screen recording setup...")

            # Create a new page for recording identification
            recording_page = await self.context.new_page()

            # Set the page title to the current UUID for window identification
            await recording_page.evaluate(
                f"""
                document.title = "{self.uuid}";
            """
            )

            # Give the page a moment to update the title
            await asyncio.sleep(2)

            # Use AppleScript to start QuickRecorder
            applescript = f"""
            tell application "QuickRecorder"
                activate
                record window titled "{self.uuid}" in application "Chromium"
            end tell
            """

            # Execute AppleScript
            import subprocess

            result = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                self.logger.info(f"Successfully started recording for window with UUID: {self.uuid}")
            else:
                self.logger.error(f"Failed to start recording: {result.stderr}")

            await asyncio.sleep(2)

            # Close the recording setup page
            await recording_page.close()

        except Exception as e:
            self.logger.error(f"Error starting recording: {e}")
            # Don't raise the exception - recording failure shouldn't stop the task

    async def _get_tabs_info(self) -> list[dict]:
        """Get information about all open tabs"""
        tabs_info = []
        for i, page in enumerate(self.context.pages):
            tabs_info.append({"id": i, "title": await page.title(), "url": page.url, "is_active": page == self.page})
        return tabs_info

    async def _wait_for_containers_online(self) -> None:
        """Wait for all launched containers to be online using HTTP HEAD requests with retry logic"""
        self.logger.info("Waiting for containers to come online...")

        # Get timeout from config (convert from milliseconds to seconds)
        timeout_seconds = self.config.browser.timeouts.container_health_check / 1000
        retry_interval = 2.0  # Wait 2 seconds between retries

        # Set up proxy if enabled
        proxy = None
        if self.config.proxy.enabled:
            proxy = self.config.proxy.server
            self.logger.info(f"Using proxy for health checks: {self.config.proxy.server}")

        # Track which sites still need to come online
        pending_sites = {site_name: (ip_address, self.config.server_port_map[site_name]) for site_name, ip_address in self.server_ips.items()}

        if not pending_sites:
            self.logger.info("No containers to health check (all using placeholder IPs)")
            return

        # Track start time for overall timeout
        start_time = asyncio.get_event_loop().time()

        # Create httpx client with per-request timeout
        health_check_headers = {
            "x-client-session-id": self.client_session_id,
        }
        async with httpx.AsyncClient(
            timeout=10.0,  # Shorter per-request timeout
            proxy=proxy,  # Use 'proxy' not 'proxies' for httpx
            follow_redirects=True,
            headers=health_check_headers,
        ) as client:
            while pending_sites and (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
                # Try each pending site
                sites_to_remove = []

                for site_name, (ip_address, port) in pending_sites.items():
                    try:
                        # Construct health check URL
                        health_url = f"http://{ip_address}:{port}"

                        # Use HEAD request to check if port is open and responding
                        response = await client.head(health_url, follow_redirects=False)

                        if response.status_code < 400:  # Accept any 2xx or 3xx status
                            self.logger.info(f"✅ {site_name} is now online (status: {response.status_code})")
                            sites_to_remove.append((site_name, ip_address, port))
                        else:
                            self.logger.debug(f"⏳ {site_name} returned status {response.status_code}, retrying...")

                    except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError):
                        # These are expected during startup, just continue retrying
                        self.logger.debug(f"⏳ {site_name} not ready yet, retrying...")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Unexpected error for {site_name}: {e}, retrying...")

                # Remove sites that are now online
                for site_name, ip_address, port in sites_to_remove:
                    del pending_sites[site_name]

                # If all sites are online, we're done
                if not pending_sites:
                    break

                # Wait before next retry attempt
                self.logger.info(f"⏳ Waiting for {len(pending_sites)} containers: {list(pending_sites.keys())}")
                await asyncio.sleep(retry_interval)

        # Check if we timed out
        if pending_sites:
            elapsed = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"❌ Timeout after {elapsed:.1f}s waiting for containers: {list(pending_sites.keys())}")
            self.logger.warning("Proceeding with setup despite some containers not being ready...")
        else:
            elapsed = asyncio.get_event_loop().time() - start_time
            self.logger.info(f"✅ All containers online after {elapsed:.1f}s!")

    async def _login_attempt(self, site_name: str, login_page, username: str, password: str) -> None:
        """Single login attempt for a specific site. Raises on failure."""
        if site_name == "shopping":
            login_url = f"http://{self.config.sites[site_name]}/customer/account/login/"
            await login_page.goto(login_url, wait_until="networkidle", timeout=self.config.browser.timeouts.page_load_networkidle)
            await login_page.get_by_label("Email", exact=True).fill(username)
            await login_page.get_by_label("Password", exact=True).fill(password)
            await asyncio.sleep(2)
            await login_page.get_by_role("button", name="Sign In").click()
            await login_page.wait_for_load_state("networkidle", timeout=self.config.browser.timeouts.page_load_networkidle)
            await asyncio.sleep(2)

        elif site_name == "reddit":
            login_url = f"http://{self.config.sites[site_name]}/login"
            await login_page.goto(login_url, wait_until="networkidle", timeout=self.config.browser.timeouts.page_load_networkidle)
            await login_page.get_by_label("Username").fill(username)
            await login_page.get_by_label("Password").fill(password)
            await login_page.get_by_role("button", name="Log in").click()
            await login_page.wait_for_load_state("networkidle", timeout=self.config.browser.timeouts.page_load_networkidle)
            await asyncio.sleep(2)

        elif site_name == "shopping_admin":
            login_url = f"http://{self.config.sites[site_name]}/admin/dashboard"
            await login_page.goto(login_url, wait_until="networkidle", timeout=self.config.browser.timeouts.page_load_networkidle)
            await login_page.get_by_placeholder("user name").fill(username)
            await login_page.get_by_placeholder("password").fill(password)
            await login_page.get_by_role("button", name="Sign in").click()
            await login_page.wait_for_load_state("networkidle", timeout=self.config.browser.timeouts.page_load_networkidle)
            await asyncio.sleep(2)

        elif site_name == "gitlab":
            login_url = f"http://{self.config.sites[site_name]}/users/sign_in"
            await login_page.goto(login_url, wait_until="networkidle", timeout=self.config.browser.timeouts.page_load_networkidle)
            await login_page.get_by_test_id("username-field").click()
            await login_page.get_by_test_id("username-field").fill(username)
            await login_page.get_by_test_id("username-field").press("Tab")
            await login_page.get_by_test_id("password-field").fill(password)
            await login_page.get_by_test_id("sign-in-button").click()
            await login_page.wait_for_load_state("networkidle", timeout=self.config.browser.timeouts.page_load_networkidle)
            await asyncio.sleep(2)

        else:
            self.logger.warning(f"No login logic implemented for site: {site_name}")

    async def login_to_site(self, site_name: str, max_retries: int = 3) -> None:
        """Login to a specific site with retry logic"""
        if not hasattr(self.config, "accounts") or site_name not in self.config.accounts:
            self.logger.warning(f"No account configured for site: {site_name}")
            return

        account = self.config.accounts[site_name]
        username = account["username"]
        password = account["password"]

        last_error = None
        for attempt in range(1, max_retries + 1):
            login_page = await self.context.new_page()
            try:
                self.logger.info(f"[{site_name}] Login attempt {attempt}/{max_retries}")
                await self._login_attempt(site_name, login_page, username, password)
                self.logger.info(f"[{site_name}] Login successful on attempt {attempt}")
                return
            except Exception as e:
                last_error = e
                self.logger.warning(f"[{site_name}] Login attempt {attempt}/{max_retries} failed: {e}")
            finally:
                await login_page.close()

            if attempt < max_retries:
                self.logger.info(f"[{site_name}] Retrying login in 3 seconds...")
                await asyncio.sleep(3)

        self.logger.error(f"[{site_name}] All {max_retries} login attempts failed")
        raise last_error

    async def ensure_logged_in(self, required_sites: list[str]) -> None:
        """Ensure user is logged into all required sites"""
        for site_name in required_sites:
            if site_name in self.config.sites:
                await self.login_to_site(site_name)
            else:
                self.logger.warning(f"Site not configured: {site_name}")

    async def setup(self, task_config: dict | None = None):
        """Initialize the browser environment with configuration"""
        self.task_config = task_config
        if task_config and "task_id" in task_config:
            self.logger = logging.getLogger(f"{__name__}.task_{task_config['task_id']}")
        self.context_manager = await self._ensure_playwright()

        # Launch web servers based on task_config["sites"]
        if self.task_config and "sites" in self.task_config:
            from rl_web_agent.incus_client import InsufficientMemoryError, health_check, launch_container

            # Get Incus server URL from config - fail fast if not configured
            incus_server_url = self.config.incus_server_url

            # Get proxy configuration
            proxy_server = self.config.proxy.server if self.config.proxy.enabled else None

            # Check if Incus server is available
            if not await health_check(incus_server_url, proxy_server=proxy_server, extra_headers={"x-client-session-id": self.client_session_id}):
                self.logger.fatal(f"Incus server not available at {incus_server_url}")
                raise RuntimeError(f"Incus server not available at {incus_server_url}")
            else:
                # Launch containers for each required site
                for site in self.task_config["sites"]:
                    try:
                        # Map site names to base container names
                        base_container_name = site.replace("_", "-")  # shopping_admin -> shopping-admin

                        # Generate unique container name using environment UUID
                        container_name = f"{site.replace('_', '-')}-{self.uuid}"

                        # Launch container and get IP
                        ip_address = await launch_container(incus_server_url, base_container_name, container_name, proxy_server=proxy_server, extra_headers={"x-client-session-id": self.client_session_id})
                        self.server_ips[site] = ip_address
                        self.launched_containers.append(container_name)

                        self.logger.info(f"Launched container {container_name} for site {site} with IP {ip_address}")

                    except InsufficientMemoryError:
                        # Host is out of memory; surface the specific exception
                        # to the caller of setup() so it can back off, trigger
                        # a cleanup, or abort the whole batch instead of
                        # treating this as a generic launch failure.
                        self.logger.error(f"Insufficient host memory while launching container for site {site}")
                        raise
                    except Exception as e:
                        self.logger.error(f"Failed to launch container for site {site}: {e}")
                        raise RuntimeError(f"Failed to launch container for site {site}: {e}") from e

                # Wait for all launched containers to be online
                if self.server_ips:
                    await self._wait_for_containers_online()
        else:
            self.logger.warning("No sites specified in task config")

        # Get launch options from config and convert to dict
        launch_options = OmegaConf.to_container(self.config.browser.launch_options, resolve=True)

        # Add cache directory if configured
        if hasattr(self.config.browser, "cache_dir") and self.config.browser.cache_dir:
            # Use absolute path for cache directory
            cache_dir = Path(self.config.browser.cache_dir).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            cache_arg = f"--disk-cache-dir={cache_dir}"
            launch_options["args"] = launch_options.get("args", []) + [cache_arg]
            self.logger.info(f"Browser cache configured: {cache_arg}")

        # Add proxy if enabled
        if self.config.proxy.enabled:
            launch_options["proxy"] = {"server": self.config.proxy.server}

        # Get context options from config and convert to dict
        context_options = OmegaConf.to_container(self.config.browser.context_options, resolve=True)

        # Add host rewrite headers for each site
        extra_headers = {}
        rewrite_mappings = []
        for site_name, hostname in self.config.sites.items():
            if site_name in self.server_ips:
                server_ip = self.server_ips[site_name]
                rewrite_mapping = f"{hostname}={server_ip}:{self.config.server_port_map[site_name]}"
                rewrite_mappings.append(rewrite_mapping)
                self.logger.info(f"Added host rewrite for {site_name}: {rewrite_mapping}")

        if rewrite_mappings:
            # Use the first mapping as primary header (most common case is single site)
            extra_headers["x-target-host-rewrite"] = rewrite_mappings[0]
            # For multiple sites, we may need additional headers but this handles the common case

        # Add client session ID header for proxy stickiness
        extra_headers["x-client-session-id"] = self.client_session_id
        self.logger.info(f"Using client session ID for proxy stickiness: {self.client_session_id}")

        # Store extra headers for later use in evaluation
        self.extra_headers = extra_headers

        if extra_headers:
            context_options["extra_http_headers"] = extra_headers

        # Check if user_data_dir is specified - use launch_persistent_context if so
        user_data_dir = None
        if hasattr(self.config.browser, "user_data_dir") and self.config.browser.user_data_dir:
            user_data_dir = self.config.browser.user_data_dir

        if user_data_dir:
            # Use launch_persistent_context for user data directory
            # Remove --disk-cache-dir from args since persistent context manages its own cache
            persistent_options = {**launch_options, **context_options}
            if "args" in persistent_options:
                persistent_options["args"] = [arg for arg in persistent_options["args"] if not arg.startswith("--disk-cache-dir")]

            self.context = await self.context_manager.chromium.launch_persistent_context(user_data_dir, **persistent_options)
            self.browser = self.context.browser
            self.logger.info(f"Using persistent context with cache in user data dir: {user_data_dir}")
        else:
            # Regular launch without persistent context
            self.browser = await self.context_manager.chromium.launch(**launch_options)
            self.context = await self.browser.new_context(**context_options)

        # Start tracing if enabled
        await self._setup_tracing()

        # Set default timeout for all locator actions
        self.context.set_default_timeout(self.config.browser.timeouts.default)

        # Add init script if it exists
        init_script_path = Path(self.config.init_script_path)
        if init_script_path.exists():
            with open(init_script_path) as f:
                await self.context.add_init_script(f.read())
        else:
            self.logger.warning(f"Init script not found: {init_script_path}")

        # Create initial page (or use existing one from persistent context)
        if self.context.pages:
            # Use existing page from persistent context
            self.page = self.context.pages[0]
        else:
            # Create new page for regular context
            self.page = await self.context.new_page()

        # Handle authentication before navigating to start_url
        if self.task_config and "sites" in self.task_config:
            required_sites = self.task_config["sites"]
            await self.ensure_logged_in(required_sites)

        # Start recording if enabled
        if self.config.recording.enabled:
            await self._start_recording()

        # Navigate to start URL from task config
        if self.task_config and "start_url" in self.task_config:
            if " |AND| " in self.task_config["start_url"]:
                urls = self.task_config["start_url"].split(" |AND| ")
                new_pages = []
                for url in urls:
                    new_page = await self.context.new_page()
                    await new_page.goto(url, wait_until="domcontentloaded", timeout=self.config.browser.timeouts.page_load_domcontent)
                    new_pages.append(new_page)
                self.page = new_pages[0]
            else:
                await self.page.goto(self.task_config["start_url"], wait_until="domcontentloaded", timeout=self.config.browser.timeouts.page_load_domcontent)
        else:
            self.logger.warning("No start_url specified in task config")
        return await self.observation(skip_evaluation=True)

    async def new_tab(self, url: str | None = None) -> int:
        """Create a new tab and optionally navigate to URL. Returns tab ID."""
        page = await self.context.new_page()
        if url:
            await page.goto(url, wait_until="domcontentloaded")
        self.page = page  # Make new tab active
        return len(self.context.pages) - 1

    async def switch_tab(self, tab_id: int) -> None:
        """Switch to a different tab by ID"""
        if 0 <= tab_id < len(self.context.pages):
            self.page = self.context.pages[tab_id]
            await self.page.bring_to_front()
        else:
            raise ValueError(f"Invalid tab ID: {tab_id}")

    async def close_tab(self, tab_id: int) -> None:
        """Close a tab by ID"""
        if 0 <= tab_id < len(self.context.pages):
            page = self.context.pages[tab_id]
            await page.close()
            # If we closed the active tab, switch to the currently activated tab from context
            if page == self.page and self.context.pages:
                # Find the currently active/focused tab in the context
                for p in self.context.pages:
                    try:
                        if await p.evaluate("document.hasFocus()"):
                            self.page = p
                            break
                    except Exception:
                        continue
                else:
                    # Fallback to last tab if no focused tab found
                    self.page = self.context.pages[-1]

                # Ensure the new active page is brought to front
                await self.page.bring_to_front()
        else:
            raise ValueError(f"Invalid tab ID: {tab_id}")

    async def reset(self):
        """Reset the environment to initial state"""
        # Close all tabs
        for page in self.context.pages:
            await page.close()
        self.page = await self.context.new_page()

        # Return to start URL from task config
        if self.task_config and "start_url" in self.task_config:
            await self.page.goto(self.task_config["start_url"], wait_until="domcontentloaded")
        else:
            self.logger.warning("No start_url specified in task config")
        return await self.observation()

    async def step(self, action: str):
        """
        Execute an action in the environment using JSON string format and return the next observation.

        Args:
            action: JSON string describing the action to execute

        Returns:
            dict: The observation after executing the action (same format as observation() method)

        Examples:
            obs = await env.step('{"action": "click", "target": "login_button"}')
            obs = await env.step('{"action": "type", "target": "username", "text": "john_doe", "enter": true}')
            obs = await env.step('{"action": "select", "target": "country", "value": "US"}')
            obs = await env.step('{"action": "goto_url", "url": "https://example.com"}')
            obs = await env.step('{"action": "back"}')
            obs = await env.step('{"action": "new_tab", "url": "https://example.com"}')
            obs = await env.step('{"action": "switch_tab", "tab_id": 1}')
            obs = await env.step('{"action": "close_tab", "tab_id": 1}')
            obs = await env.step('{"action": "terminate", "answer": "The product costs $29.99"}')
        """
        import json

        try:
            action_data = json.loads(action)
            action_name = action_data.get("action")

            if action_name == "click":
                await self.click(action_data["target"])

            elif action_name == "type":
                text = action_data["text"]
                target = action_data["target"]
                press_enter = action_data.get("enter", False)
                await self.type(target, text, press_enter)

            elif action_name == "hover":
                await self.hover(action_data["target"])

            elif action_name == "select":
                await self.select(action_data["target"], action_data["value"])

            elif action_name == "clear":
                await self.clear(action_data["target"])

            elif action_name == "key_press":
                key = action_data["key"]
                target = action_data.get("target")
                await self.key_press(key, target)

            elif action_name == "goto_url":
                await self.goto_url(action_data["url"])

            elif action_name == "back":
                await self.back()

            elif action_name == "forward":
                await self.forward()

            elif action_name == "refresh":
                await self.refresh()

            elif action_name == "new_tab":
                url = action_data.get("url")
                await self.new_tab(url)

            elif action_name == "switch_tab":
                tab_id = action_data["tab_id"]
                await self.switch_tab(tab_id)

            elif action_name == "close_tab":
                tab_id = action_data["tab_id"]
                await self.close_tab(tab_id)

            elif action_name == "terminate":
                answer = action_data.get("answer", "")
                await self.terminate(answer)

            else:
                self.logger.error(f"Unknown action: {action_name}")
                raise ValueError(f"Unknown action: {action_name}")

            # Sleep after action if configured
            if self.config.browser.sleep_after_action > 0:
                await asyncio.sleep(self.config.browser.sleep_after_action)

            # Return the next observation after executing the action
            observation = await self.observation()
            observation["error"] = None
            return observation

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON action format: {action}")
            observation = await self.observation()
            observation["error"] = f"Invalid JSON action format: {e}"
            return observation
        except KeyError as e:
            self.logger.error(f"Missing required parameter in action: {e}")
            observation = await self.observation()
            observation["error"] = f"Missing required parameter in action: {e}"
            return observation
        except Exception as e:
            self.logger.error(f"Error executing action: {action}, error: {e}")
            observation = await self.observation()
            observation["error"] = f"Error executing action: {e}"
            return observation

    # ===================================================================
    # ACTION METHODS
    # ===================================================================

    async def click(self, semantic_id: str) -> None:
        """
        Click on an element identified by its semantic ID.

        Args:
            semantic_id: The data-semantic-id of the element to click

        Example:
            await env.click("login_button")
            await env.click("menu.settings")
        """
        selector = f'[data-semantic-id="{semantic_id}"]'
        element = self.page.locator(selector)

        # Short timeout scroll - fail fast on hallucinated elements
        # Since we provide full page content, elements should exist
        await element.scroll_into_view_if_needed(timeout=500)
        await element.click(force=True)
        self.logger.info(f"Clicked element: {semantic_id}")

    async def type(self, semantic_id: str, text: str, press_enter: bool = False) -> None:
        """
        Type text into an input element.

        Args:
            semantic_id: The data-semantic-id of the input element
            text: Text to type
            press_enter: Whether to press Enter after typing

        Example:
            await env.type("search_input", "hello world")
            await env.type("username", "john_doe", press_enter=True)
        """
        selector = f'[data-semantic-id="{semantic_id}"]'
        element = self.page.locator(selector)

        # Short timeout scroll - fail fast on hallucinated elements
        await element.scroll_into_view_if_needed(timeout=500)
        await element.fill(text, force=True)  # Clear and type

        if press_enter:
            await element.press("Enter")

        self.logger.info(f"Typed '{text}' into element: {semantic_id}")

    async def hover(self, semantic_id: str) -> None:
        """
        Hover over an element to trigger tooltips or dropdown menus.

        Args:
            semantic_id: The data-semantic-id of the element to hover over

        Example:
            await env.hover("menu_item")
            await env.hover("tooltip_trigger")
        """
        selector = f'[data-semantic-id="{semantic_id}"]'
        element = self.page.locator(selector)

        # Short timeout scroll - fail fast on hallucinated elements
        await element.scroll_into_view_if_needed(timeout=500)
        await element.hover(force=True)
        self.logger.info(f"Hovered over element: {semantic_id}")

    async def select(self, semantic_id: str, value: str) -> None:
        """
        Select an option from a dropdown/select element.

        Args:
            semantic_id: The data-semantic-id of the select element
            value: The value of the option to select

        Example:
            await env.select("country_dropdown", "USA")
            await env.select("language_select", "en")
        """
        selector = f'[data-semantic-id="{semantic_id}"]'
        element = self.page.locator(selector)

        # Short timeout scroll - fail fast on hallucinated elements
        await element.scroll_into_view_if_needed(timeout=500)
        await element.select_option(value, force=True)
        self.logger.info(f"Selected '{value}' in element: {semantic_id}")

    async def clear(self, semantic_id: str) -> None:
        """
        Clear the content of an input element.

        Args:
            semantic_id: The data-semantic-id of the input element to clear

        Example:
            await env.clear("search_input")
            await env.clear("comment_textarea")
        """
        selector = f'[data-semantic-id="{semantic_id}"]'
        element = self.page.locator(selector)

        # Short timeout scroll - fail fast on hallucinated elements
        await element.scroll_into_view_if_needed(timeout=500)
        await element.clear(force=True)
        self.logger.info(f"Cleared element: {semantic_id}")

    async def key_press(self, key: str, semantic_id: str | None = None) -> None:
        """
        Press a keyboard key, optionally on a specific element.

        Args:
            key: Key to press (e.g., "Enter", "Escape", "Tab", "ArrowDown")
            semantic_id: Optional element to focus before pressing key

        Example:
            await env.key_press("Escape")  # Press Escape globally
            await env.key_press("Enter", "search_input")  # Press Enter on search input
            await env.key_press("ArrowDown", "dropdown")  # Navigate dropdown
        """
        if semantic_id:
            selector = f'[data-semantic-id="{semantic_id}"]'
            element = self.page.locator(selector)
            # Short timeout scroll - fail fast on hallucinated elements
            await element.scroll_into_view_if_needed(timeout=500)
            await element.press(key, force=True)
            self.logger.info(f"Pressed '{key}' on element: {semantic_id}")
        else:
            await self.page.keyboard.press(key)
            self.logger.info(f"Pressed '{key}' globally")

    # ===================================================================
    # NAVIGATION ACTIONS
    # ===================================================================

    async def goto_url(self, url: str) -> None:
        """
        Navigate to a specific URL in the current tab.

        Args:
            url: URL to navigate to

        Example:
            await env.goto_url("https://google.com")
            await env.goto_url("http://localhost:3000/login")
        """
        await self.page.goto(url, wait_until="domcontentloaded")
        self.logger.info(f"Navigated to: {url}")

    async def back(self) -> None:
        """
        Navigate back in browser history.

        Example:
            await env.back()
        """
        await self.page.go_back(wait_until="domcontentloaded")
        self.logger.info("Navigated back")

    async def forward(self) -> None:
        """
        Navigate forward in browser history.

        Example:
            await env.forward()
        """
        await self.page.go_forward(wait_until="domcontentloaded")
        self.logger.info("Navigated forward")

    async def refresh(self) -> None:
        """
        Refresh/reload the current page.

        Example:
            await env.refresh()
        """
        await self.page.reload(wait_until="domcontentloaded")
        self.logger.info("Page refreshed")

    async def terminate(self, answer: str = "") -> None:
        """
        Terminate the task with an optional answer.

        Args:
            answer: The model's final answer/response for the task

        Example:
            await env.terminate("The product costs $29.99")
            await env.terminate()  # Terminate without answer
        """
        self.model_answer = answer
        if answer:
            self.logger.info(f"Task terminated with answer: {answer}")
        else:
            self.logger.info("Task terminated without answer")

    async def _wait_for_custom_network_idle(self, timeout_ms: int = 10000, idle_time_ms: int = 500, page=None) -> None:
        """
        Custom network idle detection that works with XHR/fetch requests.
        Uses async JavaScript Promise-based waiting for better performance.

        Args:
            timeout_ms: Maximum time to wait for network idle
            idle_time_ms: Time of no network activity to consider idle
            page: Playwright page to wait on (defaults to self.page)
        """
        if page is None:
            page = self.page

        self.logger.info(f"Waiting for custom network idle (timeout: {timeout_ms}ms, idle: {idle_time_ms}ms)")

        try:
            # Add Python-side timeout as a safety net
            timeout_future = asyncio.create_task(asyncio.sleep(timeout_ms / 1000))
            evaluate_future = asyncio.create_task(
                page.evaluate(
                    """
                async ([idleTimeMs, timeoutMs]) => {
                    if (typeof window.__networkActivity === 'undefined') {
                        console.log('Network activity tracker not available');
                        return true; // Fallback if tracker not available
                    }

                    console.log('Starting network idle wait...');
                    try {
                        const isIdle = await window.__networkActivity.waitForIdle(idleTimeMs, timeoutMs);
                        console.log('Network idle wait completed:', isIdle);
                        return isIdle;
                    } catch (error) {
                        console.warn('Network idle wait error:', error);
                        return false;
                    }
                }
            """,
                    [idle_time_ms, timeout_ms],
                )
            )

            # Race between evaluation and timeout
            done, pending = await asyncio.wait([evaluate_future, timeout_future], return_when=asyncio.FIRST_COMPLETED)

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            if evaluate_future in done:
                result = await evaluate_future
                if result:
                    self.logger.info("Custom network idle detected")
                else:
                    self.logger.warning(f"Custom network idle timeout after {timeout_ms}ms")
            else:
                self.logger.warning("Custom network idle detection timed out on Python side")

        except Exception as e:
            self.logger.warning(f"Custom network idle check failed: {e}")
            # Fallback to old polling method
            await self._wait_for_custom_network_idle_fallback(timeout_ms, idle_time_ms, page=page)

    async def _wait_for_custom_network_idle_fallback(self, timeout_ms: int = 10000, idle_time_ms: int = 500, page=None) -> None:
        """
        Fallback polling-based network idle detection.

        Args:
            timeout_ms: Maximum time to wait for network idle
            idle_time_ms: Time of no network activity to consider idle
            page: Playwright page to wait on (defaults to self.page)
        """
        if page is None:
            page = self.page

        start_time = asyncio.get_event_loop().time()
        timeout_seconds = timeout_ms / 1000

        self.logger.info("Using fallback network idle detection")

        while True:
            try:
                # Check if our network tracker is available and if network is idle
                is_idle = await page.evaluate(
                    """
                    (idleTimeMs) => {
                        if (typeof window.__networkActivity === 'undefined') {
                            return true; // Fallback if tracker not available
                        }
                        return window.__networkActivity.isIdle(idleTimeMs);
                    }
                """,
                    idle_time_ms,
                )

                if is_idle:
                    self.logger.info("Custom network idle detected (fallback)")
                    break

                # Check timeout
                if (asyncio.get_event_loop().time() - start_time) >= timeout_seconds:
                    self.logger.warning(f"Custom network idle timeout after {timeout_ms}ms (fallback)")
                    break

                # Wait a bit before checking again
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.warning(f"Custom network idle fallback check failed: {e}")
                break

    async def observation(self, skip_evaluation: bool = False):
        """Get parsed page content using the parser script"""
        parser_script_path = Path(self.config.parser_script_path)
        content = {}

        # Wait for page to be fully loaded and stable
        try:
            self.logger.info("Waiting for page to be fully loaded and stable")
            await self.page.wait_for_load_state("domcontentloaded", timeout=self.config.browser.timeouts.page_load_domcontent)

            # Use both original networkidle (for page loads) and custom detection (for XHR/fetch)
            try:
                # First wait for Playwright's networkidle (handles initial page loads well)
                await self.page.wait_for_load_state("networkidle", timeout=self.config.browser.timeouts.page_load_networkidle)  # Shorter timeout
                self.logger.info("Playwright networkidle detected")
            except Exception as e:
                self.logger.info(f"Playwright networkidle timeout (normal): {e}")

            # Then wait for custom network idle detection (handles XHR/fetch after interactions)
            await self._wait_for_custom_network_idle(timeout_ms=self.config.browser.timeouts.page_load_networkidle, idle_time_ms=self.config.browser.timeouts.custom_network_idle)

            self.logger.info("Page loaded and stable")
        except Exception as e:
            self.logger.warning(f"Page load wait timeout: {e}")

        # Additional safety check - wait for body element
        try:
            await self.page.wait_for_selector("body", timeout=self.config.browser.timeouts.element_wait)
        except Exception as e:
            self.logger.warning(f"Body element not found: {e}")

        with open(parser_script_path) as f:
            parser_code = f.read()
        try:
            content = await self.page.evaluate(parser_code)
        except Exception as e:
            self.logger.error(f"Parser script failed: {e}")
            # Fallback to basic HTML content
            content = {"html": await self.page.content()}
        print("RAW HTML LENGTH", len(await self.page.content()))
        print("PARSED HTML LENGTH", len(content["html"]))
        print("-" * 100)

        # Add tabs information to the observation
        content["tabs"] = await self._get_tabs_info()

        # Add model answer if available
        content["model_answer"] = self.model_answer

        # Add evaluation information
        if not skip_evaluation and self.task_config and "eval" in self.task_config and self.config.get("evaluation", {}).get("enabled", True):
            score = await self.evaluate_task()
            content["score"] = score

            # Always terminate if model called terminate
            content["terminated"] = self.model_answer is not None or score != 0.0
        else:
            content["score"] = 0.0
            content["terminated"] = self.model_answer is not None

        return content

    async def evaluate_task(self) -> float:
        """
        Evaluate current task using self.task_config.

        Returns:
            float: Score between 0.0 and 1.0 indicating task success

        Raises:
            ValueError: If task_config is not set or evaluation fails
            ImportError: If WebArena evaluation modules are not available
        """
        if self.task_config is None:
            raise ValueError("task_config must be set before evaluation")

        # Import our simplified evaluator (no WebArena dependencies)
        from rl_web_agent.evaluator import evaluate_task

        # Run evaluation using our simplified evaluator
        # Pass individual parameters directly
        score = await evaluate_task(
            answer=self.model_answer or "",
            page=self.page,
            task_config=self.task_config,
            env_config=self.config,  # This has accounts, sites, etc.
            extra_headers=self.extra_headers,
            env=self,  # Pass env for network idle detection on temp pages
        )

        self.logger.info(f"Task evaluation score: {score}")
        return score

    async def close(self):
        """Clean up and close the browser"""
        # Stop tracing if active
        await self._stop_tracing()

        # Clean up launched containers
        if self.launched_containers:
            self.logger.info(f"Cleaning up {len(self.launched_containers)} launched containers...")
            try:
                from rl_web_agent.incus_client import health_check

                incus_server_url = self.config.incus_server_url

                # Get proxy configuration
                proxy_server = self.config.proxy.server if self.config.proxy.enabled else None

                # Check if Incus server is still available
                if await health_check(incus_server_url, proxy_server=proxy_server, extra_headers={"x-client-session-id": self.client_session_id}):
                    # Delete containers in parallel for faster cleanup
                    deletion_tasks = []
                    for container_name in self.launched_containers:
                        task = asyncio.create_task(self._delete_container_with_retry(incus_server_url, container_name, proxy_server))
                        deletion_tasks.append(task)

                    # Wait for all deletions to complete
                    if deletion_tasks:
                        results = await asyncio.gather(*deletion_tasks, return_exceptions=True)

                        # Log results
                        success_count = sum(1 for result in results if result is True)
                        failure_count = len(results) - success_count

                        if success_count > 0:
                            self.logger.info(f"✅ Successfully cleaned up {success_count} containers")
                        if failure_count > 0:
                            self.logger.warning(f"⚠️ Failed to clean up {failure_count} containers")
                else:
                    self.logger.warning("Incus server not available for cleanup - containers may still be running")

                self.launched_containers.clear()

            except Exception as e:
                self.logger.error(f"Error during container cleanup: {e}")

        # Explicitly close browser context and browser process.
        # We must do this rather than relying on playwright.stop() because
        # the Playwright instance is shared (refcounted) across all envs on
        # this process -- only the last env to close would trigger stop().
        # Without explicit close, Chromium processes and their file
        # descriptors leak for the lifetime of the process.
        try:
            if self.context:
                await self.context.close()
        except Exception as e:
            self.logger.warning(f"Failed to close browser context: {e}")
        finally:
            self.context = None

        try:
            if self.browser:
                await self.browser.close()
        except Exception as e:
            self.logger.warning(f"Failed to close browser: {e}")
        finally:
            self.browser = None

        if self.context_manager:
            await self._cleanup_playwright()
            self.context_manager = None

    async def _delete_container_with_retry(self, incus_server_url: str, container_name: str, proxy_server: str | None, max_retries: int = 2) -> bool:
        """
        Delete a container with retry logic.

        Args:
            incus_server_url: Incus server URL
            container_name: Name of container to delete
            proxy_server: Optional proxy server URL for HTTP requests
            max_retries: Maximum number of retry attempts

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        from rl_web_agent.incus_client import delete_container

        for attempt in range(max_retries + 1):
            try:
                await delete_container(incus_server_url, container_name, proxy_server=proxy_server, extra_headers={"x-client-session-id": self.client_session_id})
                self.logger.info(f"🗑️ Deleted container {container_name}")
                return True
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(f"⚠️ Failed to delete {container_name} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    await asyncio.sleep(1)  # Wait 1 second before retry
                else:
                    self.logger.error(f"❌ Final failure deleting {container_name}: {e}")
                    return False
        return False
