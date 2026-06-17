import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from app.config.config import HUB_THONGBAO_URL, WEB_SCRAPE_TOP_N

_cache = {"data": None}


def _scrape_thong_bao() -> list[dict]:
    response = requests.get(HUB_THONGBAO_URL, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    items = []
    for h3 in soup.select("h3 a")[:WEB_SCRAPE_TOP_N]:
        title = h3.get_text(strip=True)
        link = h3.get("href")
        items.append({"title": title, "link": link})

    return items


@tool
def hub_announcements(query: str = "") -> str:
    """Lấy danh sách thông báo mới nhất từ trang web chính thức của
    Trường Đại học Ngân hàng TP.HCM (hub.edu.vn/thong-bao).
    Dùng tool này khi câu hỏi liên quan đến thông báo, tin tức, lịch học,
    quyết định, tuyển sinh mới nhất của trường.
    """
    if _cache["data"] is None:
        _cache["data"] = _scrape_thong_bao()

    items = _cache["data"]
    lines = [f"- {item['title']} ({item['link']})" for item in items]
    return "Các thông báo mới nhất từ hub.edu.vn:\n" + "\n".join(lines)