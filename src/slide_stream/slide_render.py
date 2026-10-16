"""Render designed slides to pictures (the PPTX -> video middle step).

A designed deck — one the user already styled in PowerPoint, or one Presenton
generated from a template — cannot be fed to the video pipeline as a .pptx:
SlideStream reads text out of a .pptx and builds its own frames. Preserving
the design means rasterising each slide exactly as designed. PDF is the
exchange format: Presenton exports it directly, PowerPoint users can export
it, and LibreOffice (when installed) converts a .pptx headlessly. pypdfium2
(Google's PDFium, Apache-2.0/BSD-3) then renders each page to a PNG at video
width. It replaced PyMuPDF to keep the dependency tree free of AGPL code.

pypdfium2 is an optional extra (``slide-stream[presenton]``) so the bare
install stays lean.
"""

import shutil
import subprocess
from pathlib import Path

from PIL import Image

_DEFAULT_TIMEOUT_SECONDS = 300


class SlideRenderError(Exception):
    """Raised when designed slides cannot be rendered to pictures."""


def pdf_to_slide_images(
    pdf_path: str | Path, out_dir: str | Path, *, width: int = 1920
) -> list[Path]:
    """Render every PDF page to ``slide_N.png`` in ``out_dir``, in order.

    Each picture is exactly ``width`` pixels wide; the height keeps the
    page's aspect ratio.
    """
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]
    except ImportError:
        raise SlideRenderError(
            "Rendering designed slides from a PDF needs pypdfium2: "
            'pip install "slide-stream[presenton]"'
        ) from None

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise SlideRenderError(f"Slides PDF not found: {pdf_path}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as e:  # pdfium raises PdfiumError on unreadable files
        raise SlideRenderError(f"Could not open {pdf_path} as a PDF: {e}") from None
    try:
        images: list[Path] = []
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                page_width = float(page.get_width())
                scale = width / page_width if page_width else 1.0
                # pdfium's hint says int (from the default of 1); it takes
                # any float.
                bitmap = page.render(scale=scale)  # pyright: ignore[reportArgumentType]
                image: Image.Image = bitmap.to_pil()
            finally:
                page.close()
            if image.width != width:
                # Point-to-pixel rounding can leave the bitmap a pixel off;
                # the video frame needs the exact width.
                height = max(1, round(image.height * width / image.width))
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            dest = out / f"slide_{i + 1}.png"
            image.convert("RGB").save(dest)
            images.append(dest)
    finally:
        pdf.close()
    if not images:
        raise SlideRenderError(f"No pages found in {pdf_path}.")
    return images


def pptx_to_pdf(
    pptx_path: str | Path,
    out_dir: str | Path,
    *,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    """Convert a .pptx to PDF with a headless LibreOffice, preserving design."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise SlideRenderError(
            "Preserving a PowerPoint's design needs LibreOffice (soffice) on "
            "your PATH to render it to PDF. Install LibreOffice, or export "
            "the deck to PDF from PowerPoint and pass it with --slides-pdf."
        )
    pptx_path = Path(pptx_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf",
             "--outdir", str(out), str(pptx_path)],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise SlideRenderError(f"LibreOffice failed to render {pptx_path}: {e}") from None
    pdf = out / f"{pptx_path.stem}.pdf"
    if not pdf.is_file():
        raise SlideRenderError(
            f"LibreOffice did not produce {pdf}. Export the deck to PDF "
            "manually and pass it with --slides-pdf."
        )
    return pdf
