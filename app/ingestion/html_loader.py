"""HTML loader: keeps the article content, drops site boilerplate."""

from bs4 import BeautifulSoup, Tag

from app.dataset.schemas import DocumentMeta
from app.ingestion.blocks import BlockBuilder, clean
from app.ingestion.models import Block

BOILERPLATE = ["header", "nav", "aside", "footer", "script", "style", "noscript", "form"]
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "pre"]
# Elements whose descendants are handled by the element itself.
CONTAINERS = ["table", "pre", "li"]
HEADING_LEVELS = {"h1": 1, "h2": 1, "h3": 2, "h4": 2, "h5": 2, "h6": 2}


def load_html(raw: bytes, meta: DocumentMeta) -> tuple[list[Block], int | None]:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup.find_all(BOILERPLATE):
        tag.decompose()
    for tag in soup.select(".breadcrumb, .site-header, .site-footer, .related"):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup

    b = BlockBuilder()
    for el in root.find_all(BLOCK_TAGS):
        if el.find_parent(CONTAINERS) is not None:
            continue
        name = el.name
        text = el.get_text(" ")
        if name in HEADING_LEVELS:
            if name == "h1" and clean(text) == clean(meta.title):
                continue  # the in-file title is not a block
            b.heading(text, HEADING_LEVELS[name])
        elif name == "p":
            b.add("paragraph", text)
        elif name == "li":
            b.add("list_item", text)
        elif name == "pre":
            b.add("code", el.get_text())
        elif name == "table":
            b.table(_table_rows(el))
    return b.blocks, None


def _table_rows(table: Tag) -> list[list[str]]:
    return [
        [cell.get_text(" ") for cell in tr.find_all(["th", "td"])] for tr in table.find_all("tr")
    ]
