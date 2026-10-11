"""Visit the deployed demo in a real headless browser.

Streamlit Community Cloud only counts a visit that opens a websocket
session, so a plain HTTP GET doesn't reset the 12h sleep timer. If the
app is already asleep this also clicks the "get this app back up" button.

Usage: python scripts/keep_alive.py https://your-app.streamlit.app
"""
import sys

from playwright.sync_api import sync_playwright


def main(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=90_000)

        wake = page.get_by_role("button", name="Yes, get this app back up!")
        try:
            wake.wait_for(timeout=8_000)
            print("app was asleep — waking it")
            wake.click()
        except Exception:
            print("app already awake")

        # Streamlit renders inside .stApp (also inside an iframe on Community Cloud)
        selectors = ["[data-testid='stApp']", "iframe[title='streamlitApp']"]
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=120_000)
                print(f"app loaded ({sel})")
                break
            except Exception:
                continue
        else:
            browser.close()
            sys.exit("app never rendered")

        page.wait_for_timeout(10_000)  # keep the websocket session open briefly
        browser.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
