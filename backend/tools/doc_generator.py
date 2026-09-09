"""
Document Generator Tool: Creates Word (.docx) and PDF (.pdf) files.
Supports explicit output_format routing (pdf, docx, pptx, xlsx).
"""

import html
import logging
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Preformatted, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from backend.tools.path_safety import safe_resolve_output_path

logger = logging.getLogger(__name__)

# Output directory for generated documents
OUTPUT_DIR = Path(__file__).parent.parent.parent / "workspace" / "outputs"


# Control characters that are invalid in XML (keep tab=0x09, LF=0x0A, CR=0x0D)
_XML_INVALID_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')


def _sanitize_xml_str(text: str) -> str:
    """Remove XML-invalid control characters from a string."""
    return _XML_INVALID_RE.sub('', text)


def _format_inline_markdown(text: str) -> str:
    """Convert common inline markdown (*bold*, `code`, etc.) to ReportLab XML tags."""
    # First HTML escape XML-sensitive characters
    escaped = html.escape(text)
    # Convert bold: **text** or __text__ -> <b>text</b>
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
    escaped = re.sub(r'__(.+?)__', r'<b>\1</b>', escaped)
    # Convert italic: *text* or _text_ -> <i>text</i>
    escaped = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', escaped)
    # Convert inline code: `code` -> <font name="Courier">code</font>
    escaped = re.sub(r'`([^`]+?)`', r'<font name="Courier">\1</font>', escaped)
    return escaped


def generate_pdf(filename: str, title: str, content: str) -> str:
    """
    Generate a professional PDF document using ReportLab.

    Args:
        filename: Name of the output file (e.g. "report.pdf").
        title: The document title.
        content: The body text (supports markdown headings, bullets, code blocks).

    Returns:
        Absolute path to the created PDF file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure filename has .pdf extension
    if not filename.lower().endswith(".pdf"):
        filename = f"{Path(filename).stem}.pdf"

    output_path = safe_resolve_output_path(filename, OUTPUT_DIR)

    safe_title = _sanitize_xml_str(str(title))
    safe_content = _sanitize_xml_str(str(content))

    # Setup styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        "DocH1",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading3"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=body_style,
        leftIndent=14,
        spaceAfter=3,
    )
    code_style = ParagraphStyle(
        "DocCode",
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )

    story = [
        Paragraph(_format_inline_markdown(safe_title), title_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#00e5a0"), spaceAfter=10),
    ]

    in_code_block = False
    code_lines = []

    for raw_line in safe_content.splitlines():
        line_stripped = raw_line.strip()

        # Handle fenced code blocks
        if line_stripped.startswith("```"):
            if in_code_block:
                code_text = "\n".join(code_lines)
                pref = Preformatted(code_text, code_style)
                tbl = Table([[pref]], colWidths=[520])
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 6))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(raw_line)
            continue

        if not line_stripped:
            story.append(Spacer(1, 4))
            continue

        # Handle horizontal divider
        if line_stripped in ("---", "***", "___"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6, spaceBefore=4))
            continue

        # Handle headings
        if line_stripped.startswith("### "):
            heading_text = _format_inline_markdown(line_stripped[4:])
            story.append(Paragraph(heading_text, h2_style))
        elif line_stripped.startswith("## "):
            heading_text = _format_inline_markdown(line_stripped[3:])
            story.append(Paragraph(heading_text, h1_style))
        elif line_stripped.startswith("# "):
            heading_text = _format_inline_markdown(line_stripped[2:])
            story.append(Paragraph(heading_text, h1_style))
        elif line_stripped.startswith("- ") or line_stripped.startswith("* "):
            bullet_text = _format_inline_markdown(line_stripped[2:])
            story.append(Paragraph(f"&bull; {bullet_text}", bullet_style))
        else:
            p_text = _format_inline_markdown(line_stripped)
            story.append(Paragraph(p_text, body_style))

    # Close any unclosed code block
    if in_code_block and code_lines:
        code_text = "\n".join(code_lines)
        pref = Preformatted(code_text, code_style)
        tbl = Table([[pref]], colWidths=[520])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(tbl)

    # Save atomically via a temp file in OUTPUT_DIR
    temp_fd, temp_path_str = tempfile.mkstemp(
        prefix=f"{output_path.stem}_", suffix=".tmp", dir=str(OUTPUT_DIR)
    )
    os.close(temp_fd)
    temp_path = Path(temp_path_str)

    try:
        doc = SimpleDocTemplate(
            str(temp_path),
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )
        doc.build(story)

        # Validate PDF integrity
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise ValueError(f"Generated PDF at {temp_path} is empty or was not created")

        with open(str(temp_path), "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                raise ValueError(f"Generated file at {temp_path} does not have valid %PDF- header")

        temp_path.replace(output_path)
        logger.info(f"PDF generated atomically: {output_path}")
        return str(output_path)

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"PDF generation failed: {e}")
        raise


def generate_doc(filename: str, title: str, content: str, output_format: Optional[str] = None) -> str:
    """
    Generate a document with a title and body content.
    Routes to the appropriate generator based on output_format or file extension.

    Args:
        filename: Name of the output file (e.g. "report.docx" or "report.pdf").
        title: The document title.
        content: The body text.
        output_format: Explicit format ('pdf', 'docx', 'pptx', 'xlsx'). If omitted, inferred from filename.

    Returns:
        Absolute path to the created file.
    """
    # Normalize output format
    fmt = (output_format or "").lower().strip()
    if not fmt:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            fmt = "pdf"
        elif ext == ".pptx":
            fmt = "pptx"
        elif ext == ".xlsx":
            fmt = "xlsx"
        else:
            fmt = "docx"

    # Route to PDF generator
    if fmt == "pdf":
        if not filename.lower().endswith(".pdf"):
            filename = f"{Path(filename).stem}.pdf"
        return generate_pdf(filename, title, content)

    # Route to PPT generator if requested
    if fmt == "pptx":
        if not filename.lower().endswith(".pptx"):
            filename = f"{Path(filename).stem}.pptx"
        from backend.tools.ppt_generator import generate_ppt
        bullets = [c.strip() for c in content.splitlines() if c.strip()]
        return generate_ppt(filename, title, bullets)

    # Route to Spreadsheet generator if requested
    if fmt == "xlsx":
        if not filename.lower().endswith(".xlsx"):
            filename = f"{Path(filename).stem}.xlsx"
        from backend.tools.spreadsheet_generator import generate_sheet
        rows = [[title], [content]]
        return generate_sheet(filename, rows)

    # Default: Word (.docx) generation
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not filename.lower().endswith(".docx"):
        filename = f"{Path(filename).stem}.docx"

    output_path = safe_resolve_output_path(filename, OUTPUT_DIR)

    safe_title = _sanitize_xml_str(str(title))
    safe_content = _sanitize_xml_str(str(content))

    try:
        doc = Document()
        doc.add_heading(safe_title, level=1)

        # Split content by paragraphs/lines
        for p_text in safe_content.split("\n\n"):
            p_text = p_text.strip()
            if p_text:
                doc.add_paragraph(p_text)

        # Save atomically via a temp file in the same directory, then validate OOXML
        temp_fd, temp_path_str = tempfile.mkstemp(
            prefix=f"{output_path.stem}_", suffix=".tmp", dir=str(OUTPUT_DIR)
        )
        os.close(temp_fd)
        temp_path = Path(temp_path_str)

        try:
            doc.save(str(temp_path))

            if not zipfile.is_zipfile(str(temp_path)):
                raise ValueError(f"Generated docx at {temp_path} is not a valid ZIP/OOXML archive")

            with zipfile.ZipFile(str(temp_path)) as zf:
                namelist = zf.namelist()
                if "[Content_Types].xml" not in namelist:
                    raise ValueError("Generated docx is corrupt: missing [Content_Types].xml")

            temp_path.replace(output_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        logger.info(f"Document generated atomically: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        raise


def generate_document(filename: str, title: str, content: str, output_format: Optional[str] = None) -> str:
    """Convenience alias for generate_doc."""
    return generate_doc(filename, title, content, output_format=output_format)
