"""Verify the 30-minute login cookie through the real localhost UI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def run() -> dict:
    result: dict = {}
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        )
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("http://localhost:8513/", wait_until="domcontentloaded")
        await page.get_by_label("ชื่อบัญชี").first.fill("demo-agent-a")
        await page.get_by_label("รหัสผ่าน", exact=True).first.fill("Demo-Agent-A-2026!")
        await page.get_by_role("button", name="เข้าสู่พื้นที่ทำงาน", exact=True).click()
        await page.get_by_text("ถามต่อเรื่องไหนดี", exact=True).wait_for(timeout=120_000)

        cookies = await context.cookies("http://localhost:8513/")
        auth_cookie = next((item for item in cookies if item["name"] == "insurex_login"), None)
        result["cookie_created"] = auth_cookie is not None
        result["cookie_http_only"] = auth_cookie.get("httpOnly") if auth_cookie else None
        result["cookie_same_site"] = auth_cookie.get("sameSite") if auth_cookie else None

        await page.reload(wait_until="domcontentloaded")
        await page.get_by_text("ถามต่อเรื่องไหนดี", exact=True).wait_for(timeout=120_000)
        result["refresh_kept_login"] = await page.get_by_text("demo-agent-a", exact=True).count() > 0
        result["login_form_after_refresh"] = await page.get_by_role(
            "button", name="เข้าสู่พื้นที่ทำงาน", exact=True
        ).count()

        await page.get_by_role("button", name="ออกจากระบบ", exact=True).click()
        await page.get_by_role("button", name="เข้าสู่พื้นที่ทำงาน", exact=True).wait_for(timeout=30_000)
        await page.reload(wait_until="domcontentloaded")
        await page.get_by_role("button", name="เข้าสู่พื้นที่ทำงาน", exact=True).wait_for(timeout=30_000)
        result["logout_survived_refresh"] = (
            await page.get_by_role("button", name="เข้าสู่พื้นที่ทำงาน", exact=True).count() > 0
        )
        result["cookie_removed_on_logout"] = not any(
            item["name"] == "insurex_login"
            for item in await context.cookies("http://localhost:8513/")
        )
        await browser.close()

    result["passed"] = all(
        result[key]
        for key in (
            "cookie_created",
            "refresh_kept_login",
            "logout_survived_refresh",
            "cookie_removed_on_logout",
        )
    ) and result["login_form_after_refresh"] == 0
    output = Path("rag/reports/demo/persistent_login_browser.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    report = asyncio.run(run())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
