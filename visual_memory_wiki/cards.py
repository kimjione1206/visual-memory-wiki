from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from visual_memory_wiki.models import Node, slug


PALETTES = [
    ("#17324D", "#E8F1F2", "#F2A541", "#6CA6C1"),
    ("#31493C", "#F5F1E3", "#D65A31", "#7FB069"),
    ("#463F3A", "#F4F3EE", "#6CA6C1", "#D1495B"),
    ("#2E4057", "#F6F5AE", "#D1495B", "#56A3A6"),
    ("#1B4965", "#CAE9FF", "#5FA8D3", "#EF8354"),
    ("#2B2D42", "#EDF2F4", "#EF233C", "#8BC34A"),
    ("#355070", "#EAAC8B", "#B56576", "#4ECDC4"),
    ("#283618", "#FEFAE0", "#BC6C25", "#619B8A"),
]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _palette(node: Node) -> tuple[str, str, str, str]:
    idx = int("".join(ch for ch in node.id if ch.isdigit()) or "1") - 1
    return PALETTES[idx % len(PALETTES)]


def _motif_kind(node: Node) -> str:
    text = f"{node.title} {' '.join(node.keywords)} {node.prompt}".lower()
    if any(word in text for word in ["clip", "embedding", "vector", "space"]):
        return "map"
    if any(word in text for word in ["walker", "task", "research", "evidence"]):
        return "path"
    if any(word in text for word in ["hardware", "chip", "trace", "pcb"]):
        return "hardware"
    if any(word in text for word in ["limit", "source", "ground", "safe"]):
        return "anchor"
    if any(word in text for word in ["canvas", "obsidian", "graph"]):
        return "graph"
    return "cards"


def _draw_motif(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], kind: str, ink: str, accent: str, alt: str) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    if kind == "map":
        for x in range(x0 + 18, x1, 46):
            draw.line((x, y0 + 12, x, y1 - 12), fill="#CBD5DF", width=2)
        for y in range(y0 + 18, y1, 42):
            draw.line((x0 + 12, y, x1 - 12, y), fill="#CBD5DF", width=2)
        points = [(x0 + 62, y0 + 72), (cx, y0 + 42), (x1 - 74, y0 + 98), (cx - 24, y1 - 64), (x1 - 108, y1 - 44)]
    elif kind == "path":
        points = [(x0 + 30, y1 - 34), (x0 + 118, y0 + 70), (cx, cy), (x1 - 104, y0 + 58), (x1 - 38, y1 - 54)]
    elif kind == "hardware":
        for i in range(14):
            x = x0 + 18 + i * 18
            draw.line((x, y0 + 16, x, y1 - 16), fill=ink, width=7)
        draw.line((x0 + 16, cy, x1 - 16, cy), fill=accent, width=10)
        return
    elif kind == "anchor":
        draw.ellipse((cx - 58, cy - 58, cx + 58, cy + 58), outline=ink, width=10)
        draw.line((cx, y0 + 28, cx, cy + 92), fill=ink, width=10)
        draw.arc((cx - 100, cy + 36, cx - 18, cy + 132), 15, 185, fill=accent, width=10)
        draw.arc((cx + 18, cy + 36, cx + 100, cy + 132), 355, 165, fill=accent, width=10)
        return
    elif kind == "graph":
        points = [(x0 + 42, y0 + 46), (cx - 18, y0 + 98), (x1 - 54, y0 + 58), (x0 + 96, y1 - 50), (x1 - 94, y1 - 42)]
    else:
        for i in range(5):
            x = x0 + 34 + i * 44
            y = y0 + 32 + (i % 2) * 22
            draw.rounded_rectangle((x, y, x + 62, y + 72), radius=8, outline=ink, width=5)
        points = [(x0 + 48, y1 - 44), (cx, y0 + 42), (x1 - 46, y1 - 44)]
    for a, b in zip(points, points[1:]):
        draw.line((a[0], a[1], b[0], b[1]), fill=ink, width=5)
    for i, (x, y) in enumerate(points):
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=accent if i % 2 == 0 else alt, outline=ink, width=3)


def generate_text_card(node: Node, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bg, fg, accent, alt = _palette(node)
    image = Image.new("RGB", (512, 512), bg)
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    body_font = _font(22)
    small_font = _font(18)

    draw.rounded_rectangle((22, 22, 490, 490), radius=22, fill=bg, outline=fg, width=4)
    _draw_motif(draw, (56, 70, 456, 250), _motif_kind(node), fg, accent, alt)
    y = 292
    for line in _wrap(draw, node.title, title_font, 410)[:2]:
        draw.text((52, y), line, font=title_font, fill=fg)
        y += 40
    draw.line((52, y + 6, 460, y + 6), fill=accent, width=4)
    y += 28
    for line in _wrap(draw, " / ".join(node.keywords[:4]), body_font, 408)[:3]:
        draw.text((52, y), line, font=body_font, fill=fg)
        y += 30
    draw.text((52, 458), "keyword image card", font=small_font, fill=accent)
    path = out_dir / f"{node.id}_{slug(node.title)}.png"
    image.save(path)
    return path


def generate_textless_card(node: Node, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bg, fg, accent, alt = _palette(node)
    image = Image.new("RGB", (512, 512), bg)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, 488, 488), radius=30, fill="#FFFCF4", outline=accent, width=6)
    draw.rounded_rectangle((48, 48, 464, 464), radius=18, outline=bg, width=3)
    _draw_motif(draw, (82, 92, 430, 390), _motif_kind(node), "#17202A", accent, alt)
    for i in range(9):
        angle = i * math.tau / 9
        x = 256 + int(math.cos(angle) * 170)
        y = 256 + int(math.sin(angle) * 170)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=alt)
    path = out_dir / f"{node.id}_{slug(node.title)}_textless.png"
    image.save(path)
    return path


def attach_generated_cards(nodes: list[Node], image_dir: Path, out_dir: Path) -> list[Node]:
    out_dir.mkdir(parents=True, exist_ok=True)
    attached: list[Node] = []
    for node in nodes:
        matches = sorted(Path(image_dir).glob(f"{node.id}_*.png")) + sorted(Path(image_dir).glob(f"{node.id}_*.jpg"))
        if not matches:
            raise FileNotFoundError(f"No generated card found for {node.id} in {image_dir}")
        src = matches[0]
        dst = out_dir / f"{node.id}_{slug(node.title)}_generated{src.suffix.lower()}"
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        attached.append(Node(**{**node.__dict__, "image_path": dst}))
    return attached


def generate_cards(nodes: list[Node], out_dir: Path, style: str = "text") -> list[Node]:
    makers = {
        "text": generate_text_card,
        "textless": generate_textless_card,
    }
    if style not in makers:
        raise ValueError(f"Unsupported card style: {style}")
    result: list[Node] = []
    for node in nodes:
        path = makers[style](node, out_dir)
        result.append(Node(**{**node.__dict__, "image_path": path}))
    return result
