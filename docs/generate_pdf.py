"""
Generate a professional PDF report from project_report.md using fpdf2.
Uses built-in helvetica with unicode character sanitisation.
"""

from fpdf import FPDF
from pathlib import Path
import re
import unicodedata

MD_FILE  = Path("project_report.md")
PDF_FILE = Path("project_report.pdf")

lines = MD_FILE.read_text(encoding="utf-8").splitlines()

# ── Colour palette ─────────────────────────────────────────────────────────────
C_NAVY  = (28,  63, 170)
C_DARK  = (26,  26,  46)
C_GREY  = (100, 110, 130)
C_WHITE = (255, 255, 255)
C_LIGHT = (240, 244, 255)
C_CODE  = (244, 246, 251)
C_ACNT  = (59,  91, 219)


def sanitise(text: str) -> str:
    """Replace non-latin-1 characters with safe ASCII equivalents."""
    replacements = {
        "\u2014": "-",  "\u2013": "-",  "\u2018": "'",  "\u2019": "'",
        "\u201c": '"',  "\u201d": '"',  "\u2026": "...","\u00b1": "+/-",
        "\u2192": "->", "\u2190": "<-", "\u2714": "[OK]","\u274c": "[X]",
        "\u2190": "<-", "\u2192": "->", "\u2193": "v",   "\u2191": "^",
        "\u00ae": "(R)", "\u00a9": "(C)","\u2122": "(TM)",
        # Circled numbers -> plain
        "\u2460": "(1)", "\u2461": "(2)", "\u2462": "(3)",
        "\u2463": "(4)", "\u2464": "(5)", "\u2465": "(6)",
        "\u2466": "(7)", "\u2467": "(8)", "\u2468": "(9)",
        # Emoji / special
        "\U0001f3c6": "[WIN]", "\U0001f680": "[DEPLOY]",
        "\u2550": "=",  "\u2502": "|",
    }
    for ch, rep in replacements.items():
        text = text.replace(ch, rep)
    # Strip anything still outside latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


def strip_md_inline(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'`(.+?)`',       r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*C_GREY)
        self.cell(0, 8, "Document Classification System -- Technical Report", align="L")
        self.set_draw_color(*C_ACNT)
        self.line(self.l_margin, 18, self.w - self.r_margin, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*C_GREY)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


pdf = ReportPDF(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(20, 22, 20)
pdf.add_page()

W = pdf.w - pdf.l_margin - pdf.r_margin   # usable width

# ── Cover ──────────────────────────────────────────────────────────────────────
pdf.set_fill_color(*C_NAVY)
pdf.rect(0, 0, 210, 58, style="F")
pdf.set_y(12)
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(*C_WHITE)
pdf.set_x(pdf.l_margin)
pdf.multi_cell(W, 10, "Document Classification System", align="C")
pdf.set_font("Helvetica", "", 12)
pdf.set_x(pdf.l_margin)
pdf.multi_cell(W, 8, "End-to-End Technical Report", align="C")
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(200, 210, 255)
pdf.set_x(pdf.l_margin)
pdf.multi_cell(W, 7, "GitHub: Charanbtech/Document_Analyse   |   2026-04-29", align="C")
pdf.set_y(66)
pdf.set_text_color(*C_DARK)


# ── Helpers ────────────────────────────────────────────────────────────────────
def write_body(text, size=10):
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(*C_DARK)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(W, 5.5, sanitise(strip_md_inline(text)))


def draw_hr():
    pdf.set_draw_color(*C_ACNT)
    pdf.set_line_width(0.3)
    y = pdf.get_y() + 1
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(4)
    pdf.set_line_width(0.2)


def write_heading(text, level):
    clean = sanitise(strip_md_inline(text))
    if level == 1:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 17)
        pdf.set_text_color(*C_NAVY)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(W, 9, clean)
        draw_hr()
    elif level == 2:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*C_NAVY)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(W, 7, clean)
        pdf.ln(1)
    elif level == 3:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(44, 62, 120)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(W, 6.5, clean)
        pdf.ln(1)
    pdf.set_text_color(*C_DARK)


def write_code(code_lines):
    pdf.set_fill_color(*C_CODE)
    text = "\n".join(sanitise(l) for l in code_lines)
    pdf.set_font("Courier", "", 7.5)
    pdf.set_text_color(30, 30, 80)
    x0 = pdf.l_margin
    y0 = pdf.get_y()
    pdf.set_x(x0)
    pdf.multi_cell(W, 4.2, text, border=0, fill=True)
    y1 = pdf.get_y()
    pdf.set_draw_color(*C_ACNT)
    pdf.set_line_width(0.8)
    pdf.line(x0 - 2, y0, x0 - 2, y1)
    pdf.set_line_width(0.2)
    pdf.ln(3)
    pdf.set_text_color(*C_DARK)


def write_table(hdr, rows):
    n = len(hdr)
    if n == 0:
        return
    cw = W / n
    # Header
    pdf.set_fill_color(*C_NAVY)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_x(pdf.l_margin)
    for cell in hdr:
        pdf.cell(cw, 7, sanitise(strip_md_inline(cell)).strip(), border=1, fill=True)
    pdf.ln()
    # Rows
    pdf.set_font("Helvetica", "", 8)
    for idx, row in enumerate(rows):
        while len(row) < n:
            row.append("")
        pdf.set_fill_color(*(C_LIGHT if idx % 2 == 0 else C_WHITE))
        pdf.set_text_color(*C_DARK)
        pdf.set_x(pdf.l_margin)
        for cell in row[:n]:
            pdf.cell(cw, 6.5, sanitise(strip_md_inline(cell)).strip(), border=1, fill=True)
        pdf.ln()
    pdf.ln(3)
    pdf.set_text_color(*C_DARK)


# ── Parse ─────────────────────────────────────────────────────────────────────
in_code  = False
code_buf = []
in_table = False
tbl_hdr  = []
tbl_rows = []


def flush_table():
    global in_table, tbl_hdr, tbl_rows
    if in_table and tbl_hdr:
        write_table(tbl_hdr, tbl_rows)
    in_table = False
    tbl_hdr  = []
    tbl_rows = []


for line in lines:
    # Code fence
    if line.strip().startswith("```"):
        if in_code:
            write_code(code_buf)
            code_buf = []
            in_code  = False
        else:
            flush_table()
            in_code = True
        continue
    if in_code:
        code_buf.append(line)
        continue

    # Table row
    if line.strip().startswith("|"):
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if not in_table:
            in_table = True
            tbl_hdr  = cols
        elif all(re.match(r'^[-: ]+$', c) for c in cols if c):
            pass   # separator row
        else:
            tbl_rows.append(cols)
        continue
    else:
        flush_table()

    # Headings
    if   line.startswith("#### "): write_heading(line[5:], 3)
    elif line.startswith("### "):  write_heading(line[4:], 3)
    elif line.startswith("## "):   write_heading(line[3:], 2)
    elif line.startswith("# "):    write_heading(line[2:], 1)
    # HR
    elif line.strip() in ("---", "***", "___"):
        draw_hr()
    # Blank
    elif line.strip() == "":
        pdf.ln(2)
    # Bullet
    elif re.match(r'^\s*[-*+]\s', line):
        indent = len(line) - len(line.lstrip())
        text   = re.sub(r'^\s*[-*+]\s+', '', line)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*C_DARK)
        pdf.set_x(pdf.l_margin + 4 + indent * 0.5)
        safe = sanitise("* " + strip_md_inline(text))
        pdf.multi_cell(W - 4 - indent * 0.5, 5.5, safe)
    # Numbered
    elif re.match(r'^\d+\.\s', line):
        text = re.sub(r'^\d+\.\s+', '', line)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*C_DARK)
        pdf.set_x(pdf.l_margin + 4)
        pdf.multi_cell(W - 4, 5.5, sanitise(strip_md_inline(text)))
    # Blockquote
    elif line.startswith("> "):
        pdf.set_font("Helvetica", "I", 9.5)
        pdf.set_text_color(44, 62, 120)
        pdf.set_fill_color(*C_LIGHT)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(W, 5.5, sanitise(strip_md_inline(line[2:])), fill=True)
        pdf.set_text_color(*C_DARK)
    # Normal paragraph
    else:
        if line.strip():
            write_body(line)

flush_table()

pdf.output(str(PDF_FILE))
print(f"PDF saved: {PDF_FILE.resolve()}")
