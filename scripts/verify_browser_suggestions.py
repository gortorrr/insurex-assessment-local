"""Exercise suggested questions through the real localhost Streamlit UI.

This script requires the local app to be running and Playwright to be installed
in the active environment. It uses the installed Chrome binary, starts a new
chat for every round, clicks a suggestion in the main pane, and records the
visible RAG answer and source links.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from playwright.async_api import async_playwright


PRODUCT_RE = re.compile(r"(คุ้ม|ประกัน|ตลอดชีพ|ออมสุข|มั่นใจ|whole|khum)", re.I)


async def run(args: argparse.Namespace) -> dict:
    rounds: list[dict] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=not args.show_browser,
            executable_path=args.chrome,
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        await page.goto(args.url, wait_until="domcontentloaded")
        username = page.locator('input[aria-label="ชื่อบัญชี"]').first
        password = page.locator('input[aria-label="รหัสผ่าน"]').first
        await username.wait_for(timeout=60_000)
        await username.fill(args.username)
        await password.fill(args.password)
        await page.get_by_role("button", name="เข้าสู่พื้นที่ทำงาน", exact=True).click()
        await page.get_by_text("ถามต่อเรื่องไหนดี", exact=True).wait_for(timeout=120_000)

        for index in range(1, args.rounds + 1):
            await page.get_by_role("button", name=re.compile("เริ่มบทสนทนาใหม่")).click()
            await page.get_by_text("ถามต่อเรื่องไหนดี", exact=True).wait_for(timeout=120_000)
            verified_label = await page.get_by_text(
                re.compile("ตรวจหลักฐานแล้ว"), exact=False
            ).count() > 0

            main_buttons = page.locator('[data-testid="stMainBlockContainer"] button')
            await main_buttons.first.wait_for(timeout=120_000)
            choices: list[tuple[object, str]] = []
            for button_index in range(await main_buttons.count()):
                button = main_buttons.nth(button_index)
                text = (await button.inner_text()).strip()
                if len(text) >= 8 and PRODUCT_RE.search(text):
                    choices.append((button, text))
            if not choices:
                raise RuntimeError(
                    f"round {index}: no suggestion in main pane; "
                    f"buttons={await main_buttons.all_inner_texts()}"
                )

            # Rotate through the four visible suggestions so the test covers
            # more than one repeated question while retaining real UI clicks.
            button, question = choices[(index - 1) % len(choices)]
            before = await page.locator('[data-testid="stChatMessage"]').count()
            await button.click()
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                await page.wait_for_timeout(750)
                count = await page.locator('[data-testid="stChatMessage"]').count()
                if count >= before + 2:
                    break

            messages = await page.locator('[data-testid="stChatMessage"]').all_inner_texts()
            answer = messages[-1].strip() if messages else ""
            assistant = page.locator('[data-testid="stChatMessage"]').last
            citation_links = await assistant.locator("a").count() if messages else 0
            controlled_no_answer = "ไม่พบข้อมูลนี้ในเอกสาร" in answer
            service_error = "temporarily unavailable" in answer or "Service state:" in answer
            passed = bool(answer) and citation_links > 0 and not controlled_no_answer and not service_error
            passed = passed and before == 0 and verified_label
            result = {
                "round": index,
                "question": question,
                "answer": answer,
                "citation_links": citation_links,
                "messages_before_question": before,
                "evidence_checked_before_display": verified_label,
                "passed": passed,
            }
            rounds.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

        await browser.close()

    report = {
        "url": args.url,
        "username": args.username,
        "round_count": len(rounds),
        "new_chat_each_round": True,
        "passed_rounds": sum(item["passed"] for item in rounds),
        "passed": len(rounds) == args.rounds and all(item["passed"] for item in rounds),
        "rounds": rounds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8513/")
    parser.add_argument("--username", default="demo-agent-a")
    parser.add_argument("--password", default="Demo-Agent-A-2026!")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument(
        "--chrome",
        default=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    )
    parser.add_argument("--show-browser", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rag/reports/demo/browser_suggested_questions_10_rounds.json"),
    )
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print(json.dumps({"passed": report["passed"], "passed_rounds": report["passed_rounds"]}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
