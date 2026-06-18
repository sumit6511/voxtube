"""
PDF export module for VoxTube — professional multi-page report.

Pages:
  1. Cover    – title, video info, key stats, sentiment + language donuts
  2. Sentiment – timeline chart + distribution breakdown table
  3. Topics    – per-topic bar chart + full topics table
  4. Toxicity  – overall stat + category bars + flagged excerpts
  5. Comments  – annotated sample of 30 comments
"""

from __future__ import annotations

import io
import json
import re
import warnings
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore", ".*Devanagari.*", UserWarning)
warnings.filterwarnings("ignore", ".*Glyph.*missing.*",  UserWarning)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, NextPageTemplate, Flowable, HRFlowable,
)

# ── Palette ───────────────────────────────────────────────────────────────────

C_AMBER  = colors.HexColor("#F59E0B")
C_DARK   = colors.HexColor("#1E2330")
C_DARKER = colors.HexColor("#0D0F14")
C_POS    = colors.HexColor("#10B981")
C_NEU    = colors.HexColor("#9CA3AF")
C_NEG    = colors.HexColor("#F43F5E")
C_TOX    = colors.HexColor("#EF4444")
C_GRAY1  = colors.HexColor("#F9FAFB")
C_GRAY2  = colors.HexColor("#F3F4F6")
C_GRAY3  = colors.HexColor("#E5E7EB")
C_GRAY4  = colors.HexColor("#6B7280")
C_TEXT   = colors.HexColor("#111827")
C_SUB    = colors.HexColor("#374151")

MPL_POS  = "#10B981"
MPL_NEU  = "#9CA3AF"
MPL_NEG  = "#F43F5E"
MPL_AMB  = "#F59E0B"
MPL_LANG = {"nepali": "#10B981", "english": "#378ADD", "neplish": "#F59E0B"}

A4_W, A4_H = A4
MARGIN = 1.8 * cm

# ── Styles ────────────────────────────────────────────────────────────────────

_SS = getSampleStyleSheet()

def _style(name, **kw):
    return ParagraphStyle(name, parent=_SS["Normal"], **kw)

ST = {
    "cover_title":    _style("CoverTitle",    fontSize=28, textColor=C_TEXT,
                              fontName="Helvetica-Bold", spaceBefore=0, spaceAfter=4),
    "cover_vox":      _style("CoverVox",      fontSize=28, textColor=C_AMBER,
                              fontName="Helvetica-Bold", spaceBefore=0, spaceAfter=4),
    "cover_sub":      _style("CoverSub",      fontSize=10, textColor=C_SUB,
                              spaceAfter=3, leading=14),
    "cover_meta":     _style("CoverMeta",     fontSize=9,  textColor=C_GRAY4,
                              spaceAfter=2),
    "section_label":  _style("SecLabel",      fontSize=9,  textColor=C_AMBER,
                              fontName="Helvetica-Bold", spaceAfter=0),
    "section_title":  _style("SecTitle",      fontSize=16, textColor=C_TEXT,
                              fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=12),
    "subsection":     _style("Subsection",    fontSize=11, textColor=C_TEXT,
                              fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6),
    "body":           _style("Body",          fontSize=9,  textColor=C_TEXT,
                              leading=13, spaceAfter=6),
    "note":           _style("Note",          fontSize=8,  textColor=C_GRAY4,
                              leading=11, spaceAfter=4),
    "footer":         _style("Footer",        fontSize=7.5, textColor=C_GRAY4,
                              alignment=TA_CENTER),
    "tbl_hdr":        _style("TblHdr",        fontSize=8, textColor=C_AMBER,
                              fontName="Helvetica-Bold"),
    "tbl_body":       _style("TblBody",       fontSize=8, textColor=C_TEXT, leading=10),
    "tbl_body_sm":    _style("TblBodySm",     fontSize=7.5, textColor=C_TEXT, leading=10),
    "tbl_pos":        _style("TblPos",        fontSize=8, textColor=C_POS,
                              fontName="Helvetica-Bold"),
    "tbl_neg":        _style("TblNeg",        fontSize=8, textColor=C_NEG,
                              fontName="Helvetica-Bold"),
    "tbl_neu":        _style("TblNeu",        fontSize=8, textColor=C_NEU),
}

# ── Section header band flowable ──────────────────────────────────────────────

class SectionBand(Flowable):
    """Full-width dark header band with section title and subtitle."""

    def __init__(self, label: str, title: str, subtitle: str = ""):
        super().__init__()
        self.label    = label
        self.title    = title
        self.subtitle = subtitle
        self._width   = A4_W - 2 * MARGIN
        self._height  = 48

    def wrap(self, *_):
        return self._width, self._height

    def draw(self):
        c = self.canv
        w, h = self._width, self._height

        # Background
        c.setFillColor(C_DARK)
        c.rect(0, 0, w, h, fill=1, stroke=0)

        # Amber left accent bar
        c.setFillColor(C_AMBER)
        c.rect(0, 0, 4, h, fill=1, stroke=0)

        # Label
        c.setFillColor(C_AMBER)
        c.setFont("Helvetica-Bold", 7)
        c.drawString(12, h - 14, self.label.upper())

        # Title
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(12, h - 30, self.title)

        # Subtitle
        if self.subtitle:
            c.setFillColor(C_GRAY4)
            c.setFont("Helvetica", 8)
            c.drawString(12, h - 42, self.subtitle)


# ── Stat card table ───────────────────────────────────────────────────────────

def _stat_cards(stats: list[tuple[str, str, str]]) -> Table:
    """
    stats: [(label, value, color_name)]
    color_name: 'pos' | 'neg' | 'amber' | 'default'
    """
    color_map = {"pos": C_POS, "neg": C_NEG, "amber": C_AMBER, "default": C_TEXT}
    n = len(stats)
    col_w = (A4_W - 2 * MARGIN) / n

    header_row = [Paragraph(label, _style(f"sh{i}", fontSize=7.5, textColor=C_GRAY4,
                              fontName="Helvetica", alignment=TA_CENTER))
                  for i, (label, _, _) in enumerate(stats)]
    value_row  = [Paragraph(value, _style(f"sv{i}", fontSize=20,
                              textColor=color_map.get(clr, C_TEXT),
                              fontName="Helvetica-Bold", alignment=TA_CENTER))
                  for i, (_, value, clr) in enumerate(stats)]

    t = Table([header_row, value_row], colWidths=[col_w] * n)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_GRAY2),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("LINEAFTER",     (0, 0), (-2, -1), 0.5, C_GRAY3),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_GRAY3),
    ]))
    return t


# ── Figure helpers ────────────────────────────────────────────────────────────

def _fig_to_img(fig, width_cm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage
    pil = PILImage.open(buf)
    aspect = pil.height / pil.width
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=width_cm * cm * aspect)


def _axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.yaxis.grid(True, linewidth=0.4, color="#E5E7EB", zorder=0)
    ax.set_axisbelow(True)


def _safe_scores(j: str | None) -> dict[str, float]:
    try: return json.loads(j) if j else {}
    except: return {}


# ── Chart builders ────────────────────────────────────────────────────────────

def _donut(counts: dict[str, int], clrs: dict[str, str],
            labels: dict[str, str], title: str, figsize=(3.8, 3.0)):
    fig, ax = plt.subplots(figsize=figsize)
    data   = [(labels.get(k, k), v, clrs.get(k, "#999")) for k, v in counts.items() if v > 0]
    total  = sum(v for _, v, _ in data)
    if data:
        wedges, _, autotexts = ax.pie(
            [v for _, v, _ in data],
            colors=[c for _, _, c in data],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.40, edgecolor="white", linewidth=1.5),
            autopct=lambda p: f"{p:.0f}%" if p >= 6 else "",
            pctdistance=0.77,
        )
        for at in autotexts:
            at.set_fontsize(8.5); at.set_color("white"); at.set_fontweight("bold")
        ax.legend(wedges, [f"{l}  ({v})" for l, v, _ in data],
                  loc="center left", bbox_to_anchor=(0.98, 0.5),
                  fontsize=8, frameon=False)
        ax.text(0, 0, str(total), ha="center", va="center",
                fontsize=13, fontweight="bold", color="#1F2937")
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1F2937", pad=6)
    fig.tight_layout(pad=0.4)
    return fig


def _timeline_chart(comments: list):
    dated = [c for c in comments if c.published_at]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))

    if not dated:
        ax.text(0.5, 0.5, "No timestamp data — re-analyze video to enable this chart",
                ha="center", va="center", fontsize=10, color="#9CA3AF",
                transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout(pad=0.5)
        return fig

    span = (max(c.published_at for c in dated) -
             min(c.published_at for c in dated)).days

    granularity = "day" if span <= 14 else "week" if span <= 120 else "month"

    def key(dt):
        if granularity == "day":   return dt.strftime("%Y-%m-%d")
        if granularity == "week":
            iso = dt.isocalendar(); return f"{iso[0]}-W{iso[1]:02d}"
        return dt.strftime("%Y-%m")

    fmt = {"day": "%b %d", "week": "%b %d", "month": "%b '%y"}[granularity]

    buckets: dict = {}
    for c in dated:
        k = key(c.published_at)
        if k not in buckets:
            buckets[k] = {"pos": 0, "neu": 0, "neg": 0, "dt": c.published_at}
        label = c.sentiment_label if c.sentiment_label in ("positive","negative") else "neutral"
        buckets[k][{"positive":"pos","neutral":"neu","negative":"neg"}[label]] += 1

    keys   = sorted(buckets)
    x      = np.arange(len(keys))
    xl     = [buckets[k]["dt"].strftime(fmt) for k in keys]
    pos    = [buckets[k]["pos"] for k in keys]
    neu    = [buckets[k]["neu"] for k in keys]
    neg    = [buckets[k]["neg"] for k in keys]

    ax.plot(x, pos, color=MPL_POS, lw=2, marker="o", ms=3.5, label="Positive", zorder=3)
    ax.plot(x, neu, color=MPL_NEU, lw=2, marker="o", ms=3.5, label="Neutral",  zorder=3)
    ax.plot(x, neg, color=MPL_NEG, lw=2, marker="o", ms=3.5, label="Negative", zorder=3)
    ax.fill_between(x, pos, alpha=0.10, color=MPL_POS)
    ax.fill_between(x, neg, alpha=0.10, color=MPL_NEG)

    step = max(1, len(xl) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(xl[::step], rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_ylabel("Comments", fontsize=8)
    _axis_style(ax)
    fig.tight_layout(pad=0.5)
    return fig


def _topics_bar(topics: list):
    top = sorted(topics, key=lambda t: t.comment_count, reverse=True)[:8]
    names = [t.label.split(" | ")[0] if t.label else f"Topic {t.topic_id}" for t in top]
    pos   = [t.positive_count for t in top]
    neu   = [t.neutral_count  for t in top]
    neg   = [t.negative_count for t in top]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(names))
    b1 = ax.bar(x, pos, label="Positive", color=MPL_POS, zorder=3)
    b2 = ax.bar(x, neu, bottom=pos, label="Neutral", color=MPL_NEU, zorder=3)
    b3 = ax.bar(x, neg, bottom=[p+n for p,n in zip(pos,neu)],
                label="Negative", color=MPL_NEG, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("Comments", fontsize=8)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    _axis_style(ax)
    for bar_set in [b1, b2, b3]:
        for bar in bar_set:
            bar.set_edgecolor("white"); bar.set_linewidth(0.5)
    fig.tight_layout(pad=0.5)
    return fig


def _toxicity_bar(comments: list):
    total = len(comments) or 1
    cats  = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    lbls  = ["Toxic","Severe toxic","Obscene","Threat","Insult","Identity hate"]
    clrs  = ["#EF4444","#DC2626","#F97316","#8B5CF6","#EC4899","#F43F5E"]

    avgs = [sum(_safe_scores(c.toxicity_json).get(cat, 0) for c in comments) / total
             for cat in cats]

    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    y = np.arange(len(lbls))
    bars = ax.barh(y, avgs, color=clrs, height=0.55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean confidence score (0 = clean, 1 = certain)", fontsize=8)
    xlim = max(max(avgs) * 1.25, 0.06)
    ax.set_xlim(0, xlim)
    ax.tick_params(axis="x", labelsize=8)
    for bar, v in zip(bars, avgs):
        ax.text(v + xlim * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8, color="#374151")
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.grid(True, linewidth=0.4, color="#E5E7EB", zorder=0)
    ax.set_axisbelow(True)
    for bar in bars:
        bar.set_edgecolor("white"); bar.set_linewidth(0.5)
    fig.tight_layout(pad=0.5)
    return fig


# ── Page template with footer ─────────────────────────────────────────────────

class _FooterCanvas:
    """Mixin that adds page number + project name to every page."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._saved_page_states = []
        self.video_title = ""

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved_page_states)
        for i, state in enumerate(self._saved_page_states, 1):
            self.__dict__.update(state)
            self._draw_footer(i, n)
            super().showPage()
        super().save()

    def _draw_footer(self, page_num: int, total: int):
        self.saveState()
        self.setFont("Helvetica", 7)
        self.setFillColor(C_GRAY4)
        y = 12 * mm
        self.drawString(MARGIN, y, "VoxTube — Multidimensional YouTube Comment Analysis")
        self.drawRightString(A4_W - MARGIN, y, f"Page {page_num} of {total}")
        self.setStrokeColor(C_GRAY3)
        self.setLineWidth(0.5)
        self.line(MARGIN, y + 4 * mm, A4_W - MARGIN, y + 4 * mm)
        self.restoreState()


from reportlab.pdfgen.canvas import Canvas

class FooterCanvas(_FooterCanvas, Canvas):
    pass


# ── Report builder ────────────────────────────────────────────────────────────

def generate_pdf_report(job, comments: list, topics: list) -> io.BytesIO:
    buf = io.BytesIO()

    content_w = A4_W - 2 * MARGIN
    content_h = A4_H - 2 * MARGIN - 14 * mm   # leave room for footer

    frame = Frame(MARGIN, 14 * mm, content_w, content_h,
                  leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)

    def make_doc():
        return BaseDocTemplate(
            buf, pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=14 * mm,
        )

    doc = make_doc()
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    # Pre-compute data
    total      = len(comments)
    sent       = Counter(c.sentiment_label or "neutral" for c in comments)
    lang       = Counter(c.lang if c.lang in MPL_LANG else "neplish" for c in comments)
    toxic_cnt  = sum(1 for c in comments if c.is_toxic)
    pos_pct    = f"{sent['positive']/total*100:.0f}%" if total else "0%"
    tox_pct    = f"{toxic_cnt/total*100:.1f}%" if total else "0%"

    # ── Shared table style helpers ──────────────────────────────────────────

    HDR_STYLE = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_AMBER),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, C_GRAY1]),
        ("GRID",          (0, 0), (-1, -1), 0.4, C_GRAY3),
    ])

    SENT_COLOR = {"positive": C_POS, "neutral": C_NEU, "negative": C_NEG}

    def p(text, style="body"):
        s = ST[style] if isinstance(style, str) else style
        return Paragraph(str(text), s)
    def sp(h=6):               return Spacer(1, h)
    def hr():                  return HRFlowable(width="100%", thickness=0.5,
                                                  color=C_GRAY3, spaceAfter=12, spaceBefore=12)

    elements = []

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ══════════════════════════════════════════════════════════════════════════

    # VoxTube wordmark — explicit leading=36 prevents the HR from overlapping the 28pt text
    title_tbl = Table([[
        Paragraph('<font color="#111827">Vox</font><font color="#F59E0B">Tube</font>',
                  _style("wm", fontSize=28, fontName="Helvetica-Bold",
                         leading=36, spaceAfter=0)),
        Paragraph("Analysis Report",
                  _style("wm2", fontSize=28, fontName="Helvetica",
                         textColor=C_SUB, leading=36, spaceAfter=0)),
    ]], colWidths=[5.5*cm, content_w - 5.5*cm])
    title_tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "BOTTOM"),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING", (0,0), (-1,-1), 0),
    ]))
    elements.append(title_tbl)
    elements.append(sp(18))
    elements.append(HRFlowable(width="100%", thickness=2, color=C_AMBER,
                                spaceAfter=16, spaceBefore=0))

    # Video meta
    elements.append(p(f"<b>Video:</b>  {job.video_title or 'Untitled'}", "cover_sub"))
    elements.append(p(f"<b>URL:</b>  {job.youtube_url}", "cover_sub"))
    elements.append(p(f"<b>Generated:</b>  {datetime.now().strftime('%B %d, %Y at %H:%M')}", "cover_sub"))
    elements.append(p(f"<b>Job ID:</b>  {job.id}", "cover_meta"))
    elements.append(sp(16))

    # Stats cards
    elements.append(_stat_cards([
        ("Total Comments", str(total),     "default"),
        ("Positive",       pos_pct,        "pos"),
        ("Toxic Comments", str(toxic_cnt), "neg"),
        ("Topics Found",   str(len(topics)),"amber"),
    ]))
    elements.append(sp(20))

    # Donuts
    sent_fig = _donut(
        {"positive": sent["positive"], "neutral": sent["neutral"], "negative": sent["negative"]},
        {"positive": MPL_POS, "neutral": MPL_NEU, "negative": MPL_NEG},
        {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"},
        "Sentiment Distribution",
    )
    lang_fig = _donut(
        {"nepali": lang["nepali"], "english": lang["english"], "neplish": lang["neplish"]},
        MPL_LANG,
        {"nepali": "Nepali", "english": "English", "neplish": "Neplish"},
        "Language Breakdown",
    )
    donut_w = (content_w / 2 - 0.4 * cm)
    sent_img = _fig_to_img(sent_fig, donut_w / cm)
    lang_img = _fig_to_img(lang_fig, donut_w / cm)

    donut_row = Table([[sent_img, lang_img]],
                       colWidths=[donut_w + 0.4*cm, donut_w])
    donut_row.setStyle(TableStyle([
        ("ALIGN",  (0,0),(-1,-1), "CENTER"),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0),(-1,-1), 0),
        ("RIGHTPADDING",(0,0),(-1,-1), 0),
    ]))
    elements.append(donut_row)
    elements.append(sp(24))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=C_GRAY3,
                                spaceAfter=12, spaceBefore=0))

    # Report contents summary
    elements.append(p("Report Contents", _style("rc_hdr", fontSize=9,
                       fontName="Helvetica-Bold", textColor=C_GRAY4, spaceAfter=8)))
    contents = [
        ["Page 2", "Sentiment Analysis",
         "XLM-RoBERTa vs VADER comparison, sentiment over time"],
        ["Page 3", "Topic Analysis",
         "BERTopic clusters with per-topic sentiment distribution"],
        ["Page 4", "Toxicity Analysis",
         "ToxicBERT category breakdown and flagged comment excerpts"],
        ["Page 5", "Comments Sample",
         "30 annotated comments across all sentiment classes"],
    ]
    cont_rows = []
    for pg, title, desc in contents:
        cont_rows.append([
            p(pg,    _style(f"cp_{pg}", fontSize=8, textColor=C_AMBER,
                            fontName="Helvetica-Bold")),
            p(title, _style(f"ct_{pg}", fontSize=8, fontName="Helvetica-Bold",
                            textColor=C_TEXT)),
            p(desc,  _style(f"cd_{pg}", fontSize=8, textColor=C_GRAY4)),
        ])
    cont_tbl = Table(cont_rows, colWidths=[1.6*cm, 4.2*cm, 11.6*cm])
    cont_tbl.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ("RIGHTPADDING",  (0,0),(-1,-1), 6),
        ("LINEBELOW",     (0,0),(-1,-2), 0.4, C_GRAY3),
    ]))
    elements.append(cont_tbl)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — SENTIMENT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    elements.append(PageBreak())
    elements.append(SectionBand(
        "Section 1", "Sentiment Analysis",
        "XLM-RoBERTa primary model  ·  VADER lexicon baseline"
    ))
    elements.append(sp(14))

    elements.append(p("Sentiment Over Time", "subsection"))
    elements.append(sp(4))
    elements.append(_fig_to_img(_timeline_chart(comments), content_w / cm))
    elements.append(sp(16))
    elements.append(hr())

    # Sentiment breakdown table
    elements.append(p("Sentiment Distribution", "subsection"))
    elements.append(sp(4))

    sent_rows = [
        [p("Sentiment", "tbl_hdr"), p("Count", "tbl_hdr"),
         p("Percentage", "tbl_hdr"), p("XLM-RoBERTa Model", "tbl_hdr"),
         p("VADER Baseline", "tbl_hdr")],
    ]
    vader_sent = Counter(c.vader_label or "neutral" for c in comments)
    for label in ("positive", "neutral", "negative"):
        xlm_c   = sent[label]
        vad_c   = vader_sent[label]
        xlm_pct = f"{xlm_c/total*100:.1f}%" if total else "—"
        vad_pct = f"{vad_c/total*100:.1f}%" if total else "—"
        style   = {"positive":"tbl_pos","neutral":"tbl_neu","negative":"tbl_neg"}[label]
        sent_rows.append([
            p(label.capitalize(), style),
            p(str(xlm_c), "tbl_body"),
            p(xlm_pct, "tbl_body"),
            p(f"{xlm_c}  ({xlm_pct})", "tbl_body"),
            p(f"{vad_c}  ({vad_pct})", "tbl_body"),
        ])
    sent_rows.append([
        p("Total", _style("bt", fontSize=8, fontName="Helvetica-Bold")),
        p(str(total), _style("bt", fontSize=8, fontName="Helvetica-Bold")),
        p("100%", _style("bt", fontSize=8, fontName="Helvetica-Bold")),
        p("", "tbl_body"), p("", "tbl_body"),
    ])

    sent_tbl = Table(sent_rows,
                      colWidths=[3.2*cm, 1.8*cm, 2.2*cm, 5.0*cm, 5.2*cm],
                      repeatRows=1)
    sent_tbl.setStyle(HDR_STYLE)
    sent_tbl.setStyle(TableStyle([   # override to keep totals row bold
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, -1), (-1, -1), 6),
        ("BACKGROUND", (0, -1), (-1, -1), C_GRAY2),
    ]))
    elements.append(sent_tbl)
    elements.append(sp(10))
    elements.append(p(
        "Note: XLM-RoBERTa (cardiffnlp/twitter-xlm-roberta-base-sentiment) processes Nepali, "
        "English, and Neplish natively. VADER is an English-focused lexicon baseline — it assigns "
        "a neutral score (0.000) to Devanagari-script comments, which explains higher neutral counts "
        "in the baseline column.", "note"
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — TOPIC ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    elements.append(PageBreak())
    elements.append(SectionBand(
        "Section 2", "Topic Analysis",
        "BERTopic — unsupervised clustering with per-topic sentiment aggregation"
    ))
    elements.append(sp(14))

    if topics:
        elements.append(p("Per-Topic Sentiment Distribution", "subsection"))
        elements.append(sp(4))
        elements.append(_fig_to_img(_topics_bar(topics), content_w / cm))
        elements.append(sp(16))
        elements.append(hr())

        elements.append(p("Topic Details", "subsection"))
        elements.append(sp(4))

        topic_rows = [[
            p("Topic ID", "tbl_hdr"), p("Label",    "tbl_hdr"),
            p("Keywords", "tbl_hdr"), p("Total",    "tbl_hdr"),
            p("Positive", "tbl_hdr"), p("Neutral",  "tbl_hdr"),
            p("Negative", "tbl_hdr"), p("Dominant", "tbl_hdr"),
        ]]
        for t in sorted(topics, key=lambda x: x.comment_count, reverse=True):
            try:   kw = json.loads(t.keywords_json) if t.keywords_json else []
            except: kw = []
            dominant = max(
                [("Positive", t.positive_count),
                 ("Neutral",  t.neutral_count),
                 ("Negative", t.negative_count)],
                key=lambda x: x[1]
            )[0] if t.comment_count else "—"
            dom_style = {"Positive":"tbl_pos","Neutral":"tbl_neu","Negative":"tbl_neg"}.get(dominant,"tbl_body")
            topic_rows.append([
                p(str(t.topic_id), "tbl_body"),
                p(t.label.split(" | ")[0] if t.label else "—", "tbl_body_sm"),
                p(", ".join(kw[:4]), "tbl_body_sm"),
                p(str(t.comment_count),  "tbl_body"),
                p(str(t.positive_count), "tbl_pos"),
                p(str(t.neutral_count),  "tbl_neu"),
                p(str(t.negative_count), "tbl_neg"),
                p(dominant,               dom_style),
            ])

        topic_tbl = Table(topic_rows,
                           colWidths=[1.6*cm, 3.0*cm, 4.8*cm,
                                      1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm],
                           repeatRows=1)
        topic_tbl.setStyle(HDR_STYLE)
        elements.append(topic_tbl)
    else:
        elements.append(sp(30))
        elements.append(p("No topics were discovered for this analysis. "
                           "A larger comment set (≥50 comments) is recommended for BERTopic "
                           "to identify meaningful clusters.", "body"))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — TOXICITY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    elements.append(PageBreak())
    elements.append(SectionBand(
        "Section 3", "Toxicity Analysis",
        "unitary/toxic-bert — multi-label classification across 6 categories"
    ))
    elements.append(sp(14))

    # Big stat
    tox_card = Table([[
        Paragraph(
            f'<font color="#EF4444" size="28"><b>{toxic_cnt}</b></font>'
            f'<font color="#374151" size="11"> / {total}</font>',
            _style("tcard", alignment=TA_CENTER)
        ),
        Paragraph(
            f'<font color="#EF4444" size="28"><b>{tox_pct}</b></font>',
            _style("tcard2", alignment=TA_CENTER)
        ),
        Paragraph(
            f'<b>Threshold</b><br/>0.5 sigmoid',
            _style("tcard3", fontSize=9, textColor=C_GRAY4, alignment=TA_CENTER)
        ),
    ]], colWidths=[content_w/3]*3)
    tox_card.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_GRAY2),
        ("TOPPADDING",    (0,0),(-1,-1), 12),
        ("BOTTOMPADDING", (0,0),(-1,-1), 12),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("LINEAFTER",     (0,0),(-2,-1), 0.5, C_GRAY3),
        ("BOX",           (0,0),(-1,-1), 0.5, C_GRAY3),
    ]))
    elements.append(p("flagged as toxic", _style("tclbl", fontSize=8.5,
                        textColor=C_GRAY4, spaceAfter=6)))
    elements.append(tox_card)
    elements.append(sp(16))
    elements.append(hr())

    elements.append(p("Average Score per Category", "subsection"))
    elements.append(sp(4))
    elements.append(_fig_to_img(_toxicity_bar(comments), content_w / cm))
    elements.append(sp(16))
    elements.append(hr())

    # Flagged comment excerpts
    flagged = [c for c in comments if c.is_toxic]
    if flagged:
        elements.append(p(f"Flagged Comment Excerpts  (showing up to 8 of {len(flagged)})", "subsection"))
        elements.append(sp(4))

        tox_rows = [[
            p("#", "tbl_hdr"),
            p("Comment", "tbl_hdr"),
            p("Sentiment", "tbl_hdr"),
            p("Categories", "tbl_hdr"),
        ]]
        for i, c in enumerate(flagged[:8], 1):
            scores  = _safe_scores(c.toxicity_json)
            flagged_cats = [k.replace("_", " ").title()
                            for k, v in scores.items() if v >= 0.5]
            excerpt = (c.original_text or "")[:90]
            if len(c.original_text or "") > 90: excerpt += "…"
            slbl = c.sentiment_label or "neutral"
            sst  = {"positive":"tbl_pos","negative":"tbl_neg","neutral":"tbl_neu"}.get(slbl, "tbl_body")
            tox_rows.append([
                p(str(i), "tbl_body"),
                p(excerpt, "tbl_body_sm"),
                p(slbl.capitalize(), sst),
                p(", ".join(flagged_cats) or "—", "tbl_body_sm"),
            ])

        tox_ex_tbl = Table(tox_rows,
                            colWidths=[1.0*cm, 9.5*cm, 2.1*cm, 4.8*cm],
                            repeatRows=1)
        tox_ex_tbl.setStyle(HDR_STYLE)
        elements.append(tox_ex_tbl)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5 — COMMENTS SAMPLE
    # ══════════════════════════════════════════════════════════════════════════

    elements.append(PageBreak())
    display_n = min(30, total)
    elements.append(SectionBand(
        "Section 4", "Comments Sample",
        f"Top {display_n} of {total} comments with NLP annotations"
    ))
    elements.append(sp(14))
    elements.append(sp(4))

    # Sort: most positive first, then neutral, then most negative
    sorted_comments = (
        sorted([c for c in comments if c.sentiment_label == "positive"],
               key=lambda c: c.sentiment_score or 0, reverse=True)[:10] +
        sorted([c for c in comments if c.sentiment_label == "neutral"],
               key=lambda c: abs(c.vader_compound or 0))[:10] +
        sorted([c for c in comments if c.sentiment_label == "negative"],
               key=lambda c: c.sentiment_score or 0, reverse=True)[:10]
    )[:display_n]

    comment_rows = [[
        p("#", "tbl_hdr"),
        p("Comment", "tbl_hdr"),
        p("Sentiment", "tbl_hdr"),
        p("VADER", "tbl_hdr"),
        p("Language", "tbl_hdr"),
        p("Toxic", "tbl_hdr"),
    ]]
    lang_lbl = {"nepali": "Nepali", "english": "English", "neplish": "Neplish"}
    for i, c in enumerate(sorted_comments, 1):
        excerpt = (c.original_text or "")[:75]
        if len(c.original_text or "") > 75: excerpt += "…"
        slbl = c.sentiment_label or "neutral"
        sst  = {"positive":"tbl_pos","negative":"tbl_neg","neutral":"tbl_neu"}.get(slbl, "tbl_body")
        vc   = c.vader_compound
        comment_rows.append([
            p(str(i),  "tbl_body"),
            p(excerpt, "tbl_body_sm"),
            p(slbl.capitalize(), sst),
            p(f"{vc:+.3f}" if vc is not None else "—", "tbl_body"),
            p(lang_lbl.get(c.lang or "", c.lang or "—"), "tbl_body"),
            p("Yes" if c.is_toxic else "No",
              _style("tox_y", fontSize=8, textColor=C_TOX, fontName="Helvetica-Bold")
              if c.is_toxic else "tbl_body"),
        ])

    comment_tbl = Table(comment_rows,
                         colWidths=[1.0*cm, 8.4*cm, 2.2*cm, 1.6*cm, 2.4*cm, 1.8*cm],
                         repeatRows=1)
    comment_tbl.setStyle(HDR_STYLE)
    elements.append(comment_tbl)
    elements.append(sp(10))
    elements.append(p(
        f"Showing {len(sorted_comments)} of {total} total comments "
        f"(up to 10 per sentiment class, sorted by confidence). "
        f"Full comment data available via Excel export.", "note"
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(elements, canvasmaker=FooterCanvas)
    buf.seek(0)
    return buf
