# dump_cookies.py
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        user_data_dir="fb_copy",
        headless=True,
    )
    context.storage_state(path="storage_state.json")
    context.close()

print("[+] Successfully exported session to storage_state.json!")