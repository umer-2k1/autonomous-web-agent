from langchain_groq import ChatGroq
from pydantic import BaseModel
from playwright.async_api import async_playwright
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
import base64
import asyncio
from langgraph.prebuilt import create_react_agent
import os
from datetime import datetime
from playwright.sync_api import sync_playwright
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


model = "meta-llama/llama-4-scout-17b-16e-instruct"

class UserPrompt(BaseModel):
    prompt: str

# -------------------- Playwright Setup --------------------
_browser = None
_page = None


async def init_browser():
    global _browser, _page
    playwright = await async_playwright().start()
    _browser = await playwright.chromium.launch(headless=True)
    _page = await _browser.new_page()
    return _browser, _page


async def close_browser():
    global _browser
    if _browser:
        await _browser.close()
        _browser = None


# -------------------- LangChain Tools --------------------
@tool("Navigate to a URL")
async def navigate_tool(url: str) -> str:
    """Navigate to the given URL in the browser."""
    global _page
    await _page.goto(url, timeout=30000)
    return f"Navigated to {url}"


@tool("Click an element on the page using CSS selector")
async def click_tool(input_str: str) -> str:
    """Click an element using its CSS selector. Pass as 'selector=.class'."""
    global _page
    if not input_str.startswith("selector="):
        return "Invalid input. Use format: selector=.class"
    selector = input_str.replace("selector=", "")
    el = await _page.query_selector(selector)
    if not el:
        return f"Element not found: {selector}"
    await el.click()
    return f"Clicked {selector}"



@tool("Extract text from an element using CSS selector")
async def extract_text_tool(selector: str) -> str:
    """Extract text from an element using its CSS selector."""
    global _page
    el = await _page.query_selector(selector)
    if not el:
        return f"Element not found: {selector}"
    text = await el.inner_text()
    return f"Extracted text: {text}"


@tool("Take a screenshot of the page")
async def screenshot_tool(url: str = "") -> str:
    """Take a screenshot of the current page and return it as a base64 string."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    path = os.path.join(SCREENSHOT_DIR, filename)

    global _page
    # buf = await _page.screenshot(full_page=True, type="jpeg", quality=70)
    # return base64.b64encode(buf).decode("utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.screenshot(path=path, full_page=True)
        browser.close()

    return f"Screenshot saved to {os.path.abspath(path)}"



# -------------------- LangChain Agent --------------------
llm = ChatGroq( model=model, temperature=0.2)
tools = [navigate_tool, click_tool, extract_text_tool, screenshot_tool]
agent = create_react_agent(
tools=tools, 
model=llm 
)



async def main():
    await init_browser()
    try:
        # Example task
        # result = await agent.arun("Go to https://wjayesh.medium.com/how-autonomous-agents-work-the-concept-and-its-langchain-implementation-e1a77f612427 and extract the main heading")
        url = "https://wjayesh.medium.com/how-autonomous-agents-work-the-concept-and-its-langchain-implementation-e1a77f612427"
        inputs = {
        "messages": [
        ("user", 
         f"Navigate to {url}. Extract the main heading of the article clearly. "
         "Also, take a full-page screenshot and save it with a timestamped filename. "
         "Finally, return both the heading text and the screenshot file path.")
        ]
        }
        # inputs = {"messages": [("user", "Go to https://wjayesh.medium.com/how-autonomous-agents-work-the-concept-and-its-langchain-implementation-e1a77f612427 and extract the main heading")]}
        async for event in agent.astream(inputs):
            if "messages" in event:
                print(event["messages"][-1].content)

        # Get final result
        result = await agent.ainvoke(inputs)
        print("\n--- Final Result ---")
        print(result["messages"][-1].content)

        # print("\n--- Result ---")
        # print(result)
    finally:
        await close_browser()

if __name__ == "__main__":
    asyncio.run(main())
