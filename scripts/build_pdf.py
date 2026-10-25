"""Render paper/berth_paper.md to a two-column-free, print-styled PDF (paper/berth_paper.pdf) via pandoc HTML + Chromium.

usage: python -I build_pdf.py
"""
import asyncio
import subprocess
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "paper"
CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: 'Charter', 'Georgia', 'DejaVu Serif', serif; font-size: 10.2pt; line-height: 1.42; color: #111; }
h1 { font-size: 17pt; line-height: 1.2; margin: 0 0 4pt; text-align: center; }
h1 + p { text-align: center; margin-top: 0; font-size: 11pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; border-bottom: 0.6pt solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; }
p, li { text-align: justify; hyphens: auto; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; background: #f3f3f1; padding: 0 2px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 6pt 0 8pt; }
th, td { border-bottom: 0.5pt solid #ccc; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { border-bottom: 0.9pt solid #555; }
img { display: block; max-width: 100%; margin: 6pt auto; }
h2#abstract + p { font-size: 9.6pt; margin: 0 6mm; }
ol, ul { padding-left: 16pt; }
h2#references ~ p { font-size: 8.6pt; text-align: left; margin: 2pt 0; }
"""


def main():
    md = P / "berth_paper.md"
    html = P / "berth_paper.html"
    # one reference per line -> separate paragraphs
    text = md.read_text().replace("\n[", "\n\n[")
    tmp = P / ".berth_print.md"
    tmp.write_text(text)
    (P / ".print.css").write_text(CSS)
    subprocess.run(["pandoc", str(tmp), "-f", "gfm", "-t", "html5", "-s", "--metadata", "pagetitle=Berth",
                    "--css", ".print.css", "--embed-resources", "--resource-path", str(P), "-o", str(html)],
                   check=True, cwd=P)

    async def render():
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch()
            pg = await b.new_page()
            await pg.goto(html.as_uri())
            await pg.pdf(path=str(P / "berth_paper.pdf"), format="A4", print_background=True,
                         margin={"top": "18mm", "bottom": "18mm", "left": "17mm", "right": "17mm"})
            await b.close()

    asyncio.run(render())
    tmp.unlink()
    print("wrote", P / "berth_paper.pdf")


if __name__ == "__main__":
    main()
