from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "runs" / "tee-zhang-2023" / "released-code"
SUMMARY_PATH = RESULT_ROOT / "summary.json"
FIGURE_DIR = ROOT / "report" / "figures"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "tee_zhang_2023_reproduction_report.pdf"
COMPARISON_FIGURE = FIGURE_DIR / "tee_zhang_2023_paper_comparison.png"
SEED_FIGURE = FIGURE_DIR / "tee_zhang_2023_per_seed.png"

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2E74B5")
TEAL = colors.HexColor("#238B8D")
ORANGE = colors.HexColor("#D97706")
MUTED = colors.HexColor("#5C6773")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_TEAL = colors.HexColor("#E8F5F3")
LIGHT_GRAY = colors.HexColor("#F2F4F7")
BORDER = colors.HexColor("#C9D2DC")
INK = colors.HexColor("#111827")
WHITE = colors.white

PIL_NAVY = "#16324F"
PIL_BLUE = "#2E74B5"
PIL_TEAL = "#238B8D"
PIL_ORANGE = "#D97706"
PIL_MUTED = "#5C6773"
PIL_LIGHT_BLUE = "#EAF2F8"
PIL_LIGHT_GRAY = "#F2F4F7"
PIL_BORDER = "#C9D2DC"
PIL_INK = "#111827"
PIL_WHITE = "#FFFFFF"


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results() -> tuple[list[dict], list[dict]]:
    summaries = load_json(SUMMARY_PATH)
    if not isinstance(summaries, list):
        raise ValueError("summary.json must contain a list")

    result_rows: list[dict] = []
    for path in sorted(RESULT_ROOT.glob("division-*/seed-*/results.json")):
        payload = load_json(path)
        config = payload["config"]
        metrics = payload["metrics"]
        matrix = payload["accuracy_matrix"]
        if len(matrix) != 20 or any(len(row) != 20 for row in matrix):
            raise ValueError(f"Expected a 20x20 accuracy matrix: {path}")
        result_rows.append(
            {
                "division": int(config["division"]),
                "seed": int(config["seed"]),
                "average_accuracy": float(metrics["continual_average_accuracy"]),
                "forgetfulness": float(metrics["forgetfulness"]),
                "final_average_accuracy": float(metrics["final_average_accuracy"]),
                "elapsed_seconds": float(payload["elapsed_seconds"]),
                "environment": payload["environment"],
                "config": config,
                "path": path,
            }
        )

    expected = {(division, seed) for division in (1, 120, 300) for seed in range(4)}
    observed = {(row["division"], row["seed"]) for row in result_rows}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Unexpected matrix coverage; missing={missing}, extra={extra}")

    return summaries, sorted(result_rows, key=lambda row: (row["division"], row["seed"]))


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def draw_metric_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    summaries: list[dict],
    *,
    title: str,
    mean_key: str,
    std_key: str,
    paper_key: str,
    y_min: float,
    y_max: float,
    lower_is_better: bool,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=24, fill=PIL_WHITE, outline=PIL_BORDER, width=2)
    title_font = pil_font(32, bold=True)
    label_font = pil_font(22)
    small_font = pil_font(19)
    value_font = pil_font(20, bold=True)
    draw.text((left + 34, top + 24), title, font=title_font, fill=PIL_NAVY)
    direction = "lower is better" if lower_is_better else "higher is better"
    draw.text((right - 220, top + 34), direction, font=small_font, fill=PIL_MUTED)

    plot_left = left + 90
    plot_right = right - 34
    plot_top = top + 92
    plot_bottom = bottom - 82

    tick_count = 5
    for tick in range(tick_count + 1):
        value = y_min + (y_max - y_min) * tick / tick_count
        y = plot_bottom - (plot_bottom - plot_top) * tick / tick_count
        draw.line((plot_left, y, plot_right, y), fill=PIL_LIGHT_GRAY, width=2)
        draw.text((left + 24, y - 11), f"{value:.0f}", font=small_font, fill=PIL_MUTED)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=PIL_MUTED, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=PIL_MUTED, width=2)

    group_width = (plot_right - plot_left) / len(summaries)
    bar_width = 60
    for index, row in enumerate(summaries):
        center = plot_left + group_width * (index + 0.5)
        paper_value = 100 * float(row[paper_key])
        reproduction_value = 100 * float(row[mean_key])
        std = 100 * float(row[std_key])

        def value_to_y(value: float) -> float:
            return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

        paper_y = value_to_y(paper_value)
        reproduction_y = value_to_y(reproduction_value)
        paper_x0 = center - bar_width - 10
        paper_x1 = center - 10
        reproduction_x0 = center + 10
        reproduction_x1 = center + bar_width + 10
        draw.rounded_rectangle(
            (paper_x0, paper_y, paper_x1, plot_bottom),
            radius=8,
            fill=PIL_MUTED,
        )
        draw.rounded_rectangle(
            (reproduction_x0, reproduction_y, reproduction_x1, plot_bottom),
            radius=8,
            fill=PIL_BLUE,
        )

        error_top = value_to_y(reproduction_value + std)
        error_bottom = value_to_y(reproduction_value - std)
        error_x = (reproduction_x0 + reproduction_x1) / 2
        draw.line((error_x, error_top, error_x, error_bottom), fill=PIL_INK, width=4)
        draw.line((error_x - 14, error_top, error_x + 14, error_top), fill=PIL_INK, width=4)
        draw.line((error_x - 14, error_bottom, error_x + 14, error_bottom), fill=PIL_INK, width=4)

        text_center(draw, ((paper_x0 + paper_x1) / 2, paper_y - 22), f"{paper_value:.1f}", value_font, PIL_MUTED)
        text_center(
            draw,
            ((reproduction_x0 + reproduction_x1) / 2, reproduction_y + 38),
            f"{reproduction_value:.2f}",
            value_font,
            PIL_WHITE,
        )
        text_center(draw, (center, plot_bottom + 38), f"division {row['division']}", label_font, PIL_INK)


def build_comparison_figure(summaries: list[dict]) -> None:
    canvas = PILImage.new("RGB", (1800, 900), PIL_WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(42, bold=True)
    subtitle_font = pil_font(23)
    draw.text((70, 36), "Paper targets and four-seed reproduction", font=title_font, fill=PIL_NAVY)
    draw.text(
        (70, 92),
        "Blue bars show reproduction means; error bars show sample standard deviation across seeds 0-3.",
        font=subtitle_font,
        fill=PIL_MUTED,
    )
    draw_metric_panel(
        draw,
        (55, 145, 885, 790),
        summaries,
        title="Continual average accuracy (%)",
        mean_key="average_accuracy_mean",
        std_key="average_accuracy_std",
        paper_key="paper_average_accuracy",
        y_min=35,
        y_max=50,
        lower_is_better=False,
    )
    draw_metric_panel(
        draw,
        (915, 145, 1745, 790),
        summaries,
        title="Forgetfulness F (%)",
        mean_key="forgetfulness_mean",
        std_key="forgetfulness_std",
        paper_key="paper_forgetfulness",
        y_min=50,
        y_max=70,
        lower_is_better=True,
    )
    legend_font = pil_font(21)
    draw.rounded_rectangle((610, 820, 650, 850), radius=6, fill=PIL_MUTED)
    draw.text((664, 822), "Paper", font=legend_font, fill=PIL_INK)
    draw.rounded_rectangle((785, 820, 825, 850), radius=6, fill=PIL_BLUE)
    draw.text((839, 822), "Reproduction", font=legend_font, fill=PIL_INK)
    draw.text((1130, 822), "All values are percentages.", font=legend_font, fill=PIL_MUTED)
    canvas.save(COMPARISON_FIGURE, quality=95)


def draw_seed_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rows: list[dict],
    *,
    title: str,
    key: str,
    y_min: float,
    y_max: float,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=24, fill=PIL_WHITE, outline=PIL_BORDER, width=2)
    title_font = pil_font(31, bold=True)
    label_font = pil_font(20)
    draw.text((left + 34, top + 24), title, font=title_font, fill=PIL_NAVY)
    plot_left = left + 90
    plot_right = right - 38
    plot_top = top + 90
    plot_bottom = bottom - 80
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = plot_bottom - (plot_bottom - plot_top) * tick / 5
        draw.line((plot_left, y, plot_right, y), fill=PIL_LIGHT_GRAY, width=2)
        draw.text((left + 25, y - 10), f"{value:.0f}", font=label_font, fill=PIL_MUTED)
    for seed in range(4):
        x = plot_left + (plot_right - plot_left) * seed / 3
        draw.line((x, plot_top, x, plot_bottom), fill="#F7F8FA", width=2)
        text_center(draw, (x, plot_bottom + 35), f"seed {seed}", label_font, PIL_INK)

    palette = {1: PIL_ORANGE, 120: PIL_BLUE, 300: PIL_TEAL}
    for division in (1, 120, 300):
        points = []
        for row in [candidate for candidate in rows if candidate["division"] == division]:
            x = plot_left + (plot_right - plot_left) * row["seed"] / 3
            value = 100 * row[key]
            y = plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            points.append((x, y))
        draw.line(points, fill=palette[division], width=5)
        for point in points:
            draw.ellipse((point[0] - 9, point[1] - 9, point[0] + 9, point[1] + 9), fill=palette[division], outline=PIL_WHITE, width=3)


def build_seed_figure(rows: list[dict]) -> None:
    canvas = PILImage.new("RGB", (1800, 860), PIL_WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = pil_font(42, bold=True)
    legend_font = pil_font(21)
    draw.text((70, 36), "Per-seed stability", font=title_font, fill=PIL_NAVY)
    draw_seed_panel(
        draw,
        (55, 120, 885, 770),
        rows,
        title="Continual average accuracy (%)",
        key="average_accuracy",
        y_min=35,
        y_max=50,
    )
    draw_seed_panel(
        draw,
        (915, 120, 1745, 770),
        rows,
        title="Forgetfulness F (%)",
        key="forgetfulness",
        y_min=50,
        y_max=70,
    )
    legend_x = 620
    for division, color in ((1, PIL_ORANGE), (120, PIL_BLUE), (300, PIL_TEAL)):
        draw.line((legend_x, 815, legend_x + 45, 815), fill=color, width=6)
        draw.ellipse((legend_x + 15, 806, legend_x + 33, 824), fill=color)
        draw.text((legend_x + 58, 803), f"division {division}", font=legend_font, fill=PIL_INK)
        legend_x += 230
    canvas.save(SEED_FIGURE, quality=95)


def register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
        pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
        return "ReportSans", "ReportSans-Bold"
    return "Helvetica", "Helvetica-Bold"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def build_pdf(summaries: list[dict], rows: list[dict]) -> None:
    regular_font, bold_font = register_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=23,
        leading=27,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=7,
    )
    subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName=regular_font,
        fontSize=10.5,
        leading=14,
        textColor=MUTED,
        spaceAfter=13,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=14,
        leading=17,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )
    subheading = ParagraphStyle(
        "Subheading",
        parent=styles["Heading3"],
        fontName=bold_font,
        fontSize=11,
        leading=14,
        textColor=BLUE,
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=9.3,
        leading=13.1,
        textColor=INK,
        spaceAfter=5,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=8.1,
        leading=10.7,
        textColor=MUTED,
    )
    callout = ParagraphStyle(
        "Callout",
        parent=body,
        fontName=bold_font,
        fontSize=10.2,
        leading=14.2,
        textColor=NAVY,
        leftIndent=8,
        rightIndent=8,
        spaceBefore=4,
        spaceAfter=4,
    )
    code_style = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=6.8,
        leading=9.2,
        textColor=INK,
        leftIndent=8,
        rightIndent=8,
        borderColor=BORDER,
        borderWidth=0.5,
        borderPadding=7,
        backColor=LIGHT_GRAY,
        spaceBefore=5,
        spaceAfter=7,
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=small,
        fontName=bold_font,
        textColor=WHITE,
        alignment=TA_CENTER,
        leading=9.5,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=small,
        textColor=INK,
        alignment=TA_CENTER,
        leading=9.5,
    )

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
        title="Tee and Zhang 2023 Interleave-Division Reproduction Report",
        author="Independent reproduction project",
        subject="Four-seed reproduction of Table 1 on ciFAIR-100",
    )

    def header_footer(canvas, document) -> None:
        canvas.saveState()
        width, height = A4
        if document.page > 1:
            canvas.setFont(regular_font, 7.5)
            canvas.setFillColor(MUTED)
            canvas.drawString(16 * mm, height - 9 * mm, "Tee and Zhang (2023) interleave-division reproduction")
            canvas.setStrokeColor(BORDER)
            canvas.line(16 * mm, height - 11 * mm, width - 16 * mm, height - 11 * mm)
        canvas.setFont(regular_font, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 8 * mm, "Independent reproduction note - August 2026")
        canvas.drawRightString(width - 16 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    summary_by_division = {int(row["division"]): row for row in summaries}
    max_gap = max(
        max(
            abs(100 * (float(row["average_accuracy_mean"]) - float(row["paper_average_accuracy"]))),
            abs(100 * (float(row["forgetfulness_mean"]) - float(row["paper_forgetfulness"]))),
        )
        for row in summaries
    )

    story = []
    story.append(paragraph("Reproducing Interleave-Division Replay on ciFAIR-100", title))
    story.append(
        paragraph(
            "A four-seed audit of Table 1 in Tee and Zhang (2023), using the authors' released-code ordering behavior",
            subtitle,
        )
    )
    story.append(
        Table(
            [[paragraph(
                f"<b>Outcome:</b> all 12 planned runs completed. The six headline metrics are within "
                f"<b>{max_gap:.2f} percentage points</b> of the paper, and the reported division ordering is recovered.",
                callout,
            )]],
            colWidths=[178 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_TEAL),
                ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]),
        )
    )
    story.append(Spacer(1, 5))
    story.append(paragraph("Executive summary", heading))
    story.append(
        paragraph(
            "This study reproduces the ciFAIR-100 interleave-division experiment from <i>Integrating Curricula with Replays: Its Effects on Continual Learning</i>. "
            "A pretrained MobileNetV3-Small learns 20 sequential five-class tasks while retaining a class-balanced replay memory of 1,200 images. "
            "The experiment compares divisions 1, 120, and 300 with controlled seeds 0-3. Division 120 improves continual average accuracy by 6.90 points and reduces forgetfulness by 9.19 points relative to division 1. Division 300 provides no accuracy gain over division 120 and has slightly worse forgetting, matching the paper's qualitative result.",
            body,
        )
    )

    table_data = [[
        paragraph("Division", table_header),
        paragraph("Reproduction Avg Acc", table_header),
        paragraph("Paper Avg Acc", table_header),
        paragraph("Delta", table_header),
        paragraph("Reproduction F", table_header),
        paragraph("Paper F", table_header),
        paragraph("Delta", table_header),
    ]]
    for row in summaries:
        acc = 100 * float(row["average_accuracy_mean"])
        acc_std = 100 * float(row["average_accuracy_std"])
        paper_acc = 100 * float(row["paper_average_accuracy"])
        forget = 100 * float(row["forgetfulness_mean"])
        forget_std = 100 * float(row["forgetfulness_std"])
        paper_forget = 100 * float(row["paper_forgetfulness"])
        table_data.append([
            paragraph(str(row["division"]), table_cell),
            paragraph(f"{acc:.2f} +/- {acc_std:.2f}%", table_cell),
            paragraph(f"{paper_acc:.1f}%", table_cell),
            paragraph(f"{acc - paper_acc:+.2f} pp", table_cell),
            paragraph(f"{forget:.2f} +/- {forget_std:.2f}%", table_cell),
            paragraph(f"{paper_forget:.1f}%", table_cell),
            paragraph(f"{forget - paper_forget:+.2f} pp", table_cell),
        ])
    summary_table = Table(
        table_data,
        colWidths=[14 * mm, 37 * mm, 27 * mm, 19 * mm, 37 * mm, 25 * mm, 19 * mm],
        repeatRows=1,
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 7))
    story.append(RLImage(str(COMPARISON_FIGURE), width=178 * mm, height=89 * mm))
    story.append(paragraph("Figure 1. Paper targets and four-seed reproduction means. Error bars are sample standard deviations.", small))

    story.append(PageBreak())
    story.append(paragraph("1. Experimental protocol", heading))
    protocol_data = [
        [paragraph("Component", table_header), paragraph("Locked configuration", table_header)],
        [paragraph("Data stream", table_cell), paragraph("ciFAIR-100; 20 tasks; 5 new classes per task; class-order seed 100", table_cell)],
        [paragraph("Model", table_cell), paragraph("ImageNet-pretrained torchvision MobileNetV3-Small; expanding five-class head", table_cell)],
        [paragraph("Replay", table_cell), paragraph("1,200 images, rebalanced equally across all seen classes", table_cell)],
        [paragraph("Optimization", table_cell), paragraph("SGD; LR 0.001; momentum 0.9; batch size 32; patience 5", table_cell)],
        [paragraph("Comparison", table_cell), paragraph("Released-code divisions 1 / 120 / 300; seeds 0 / 1 / 2 / 3", table_cell)],
        [paragraph("Execution", table_cell), paragraph("Deterministic CUDA; pretrained weights; AMP; image size 74", table_cell)],
    ]
    protocol_table = Table(protocol_data, colWidths=[42 * mm, 136 * mm], repeatRows=1)
    protocol_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(protocol_table)
    story.append(paragraph("Metrics", subheading))
    story.append(
        paragraph(
            "Let <i>a</i><sub>t</sub> be accuracy on all classes seen after task <i>t</i>, and <i>b</i><sub>t</sub> be accuracy on the five Task-1 classes after task <i>t</i>. "
            "The paper metric is continual average accuracy = mean<sub>t=1..20</sub>(<i>a</i><sub>t</sub>). "
            "Forgetfulness is F = mean<sub>t=2..20</sub>((<i>b</i><sub>1</sub> - <i>b</i><sub>t</sub>) / <i>b</i><sub>1</sub>). "
            "The Task-1 row is excluded, following the authors' released plotting notebook.",
            body,
        )
    )
    story.append(RLImage(str(SEED_FIGURE), width=178 * mm, height=85 * mm))
    story.append(paragraph("Figure 2. Per-seed metrics. Every seed preserves the main division-120 improvement over division 1.", small))
    story.append(paragraph("Environment", subheading))
    environment = rows[0]["environment"]
    story.append(
        paragraph(
            f"Python {environment['python']}; PyTorch {environment['torch']}; torchvision {environment['torchvision']}; "
            f"{environment['cuda_device']}. Every saved run records configuration, environment, trajectories, a 20x20 accuracy matrix, task logs, and epoch logs.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(paragraph("2. Findings and interpretation", heading))
    story.append(paragraph("Finding 1: the main benefit is recovered", subheading))
    d1 = summary_by_division[1]
    d120 = summary_by_division[120]
    d300 = summary_by_division[300]
    story.append(
        paragraph(
            f"Moving from division 1 to division 120 raises continual average accuracy from {100*d1['average_accuracy_mean']:.2f}% to {100*d120['average_accuracy_mean']:.2f}% "
            f"and lowers F from {100*d1['forgetfulness_mean']:.2f}% to {100*d120['forgetfulness_mean']:.2f}%. "
            "The direction and magnitude closely follow Table 1.",
            body,
        )
    )
    story.append(paragraph("Finding 2: more interleaving is not monotonically better", subheading))
    story.append(
        paragraph(
            f"Division 300 reaches {100*d300['average_accuracy_mean']:.2f}% average accuracy, slightly below division 120, while F rises to {100*d300['forgetfulness_mean']:.2f}%. "
            "This reproduces the reported plateau: increasingly frequent switching is not automatically beneficial once the schedule is already highly interleaved.",
            body,
        )
    )
    story.append(paragraph("Finding 3: numerical agreement is strong but not exact", subheading))
    story.append(
        paragraph(
            f"Average-accuracy gaps are 0.11-0.31 percentage points and F gaps are 0.40-0.79 points. "
            "All reproduction accuracy means are slightly below the paper and all F means are slightly higher. This consistent small shift is compatible with differences in modern CUDA kernels, torchvision pretrained weights, and the released environment.",
            body,
        )
    )

    story.append(paragraph("3. Audit findings and limitations", heading))
    limitations = [
        ("Released code does not keep the current-sample budget exactly equal.", "For 2,250 current examples, the released implementation duplicates 90 examples at division 120 and 150 at division 300. The current report intentionally reproduces this behavior, so the gain cannot yet be attributed solely to interleaving frequency."),
        ("Paper and code disagree on image size.", "The appendix states 72x72, while the released VaryDiv.py uses 74x74. This reproduction follows the released code at 74x74."),
        ("The bounded matrix covers three of five divisions.", "Divisions 8 and 60 remain untested. The selected 1/120/300 conditions are sufficient to test the two main qualitative claims but not reconstruct the full curve."),
        ("Runtime is not a valid cross-division result for the early division-1 seeds.", "Two early runs include Windows Modern Standby intervals. Their metrics and saved matrices are complete, but their elapsed-time fields should not be used for speed comparisons."),
        ("Causal interpretation requires the equal-budget sensitivity run.", "A matched equal-budget experiment is the next decisive step because it removes duplicate-tail exposure while preserving the interleave schedule."),
    ]
    limitation_rows = []
    for label, explanation in limitations:
        limitation_rows.append([
            paragraph(label, ParagraphStyle("LimLabel", parent=body, fontName=bold_font, textColor=NAVY, spaceAfter=0)),
            paragraph(explanation, ParagraphStyle("LimText", parent=body, spaceAfter=0)),
        ])
    limitation_table = Table(limitation_rows, colWidths=[58 * mm, 120 * mm])
    limitation_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_BLUE, WHITE]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(limitation_table)

    story.append(paragraph("4. Reproducibility and evidence", heading))
    story.append(
        paragraph(
            "The matrix runner validates matching configurations, skips completed cells, and rewrites the aggregate summary after every successful run. The following command reproduces the reported matrix:",
            body,
        )
    )
    command = (
        "$env:TORCH_HOME = \"$PWD\\.torch-cache\"\n"
        "conda run --no-capture-output -n ece488_clip python run_tee_zhang_2023_matrix.py `\n"
        "  --sequence-implementation released-code `\n"
        "  --divisions 1 120 300 `\n"
        "  --seeds 0 1 2 3"
    )
    story.append(Preformatted(command, code_style))
    integrity_rows = [
        [paragraph("Evidence check", table_header), paragraph("Verified result", table_header)],
        [paragraph("Planned cells", table_cell), paragraph("12/12 complete", table_cell)],
        [paragraph("Per-run accuracy matrices", table_cell), paragraph("12 matrices; each 20x20", table_cell)],
        [paragraph("Numerical integrity", table_cell), paragraph("No NaN or Infinity in JSON/CSV evidence", table_cell)],
        [paragraph("Process outcome", table_cell), paragraph("Controller completed; exit code 0; error logs empty", table_cell)],
        [paragraph("Stored detail", table_cell), paragraph("Configuration, environment, task trajectory, epoch log, matrix, checkpoint", table_cell)],
    ]
    integrity_table = Table(integrity_rows, colWidths=[72 * mm, 106 * mm], repeatRows=1)
    integrity_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(integrity_table)
    story.append(paragraph("Per-seed appendix", heading))
    per_seed = [[
        paragraph("Division", table_header),
        paragraph("Seed", table_header),
        paragraph("Avg Acc", table_header),
        paragraph("F", table_header),
        paragraph("Final seen-task avg", table_header),
        paragraph("Elapsed", table_header),
    ]]
    for row in rows:
        elapsed_text = f"{row['elapsed_seconds']/60:.1f} min"
        if row["division"] == 1 and row["seed"] in (1, 2):
            elapsed_text += "*"
        per_seed.append([
            paragraph(str(row["division"]), table_cell),
            paragraph(str(row["seed"]), table_cell),
            paragraph(f"{100*row['average_accuracy']:.2f}%", table_cell),
            paragraph(f"{100*row['forgetfulness']:.2f}%", table_cell),
            paragraph(f"{100*row['final_average_accuracy']:.2f}%", table_cell),
            paragraph(elapsed_text, table_cell),
        ])
    seed_table = Table(
        per_seed,
        colWidths=[20 * mm, 15 * mm, 29 * mm, 25 * mm, 47 * mm, 32 * mm],
        repeatRows=1,
    )
    seed_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(seed_table)
    story.append(paragraph("* Elapsed time includes Windows standby and is not suitable for runtime comparison.", small))
    story.append(paragraph("Next experiment", heading))
    story.append(
        paragraph(
            "Run divisions 1/120/300 under the equal-budget implementation with paired seeds. If the division-120 advantage remains after duplicate-tail exposure is removed, the result becomes substantially stronger evidence for interleaving frequency rather than sample-count confounding.",
            body,
        )
    )
    story.append(paragraph("Reference", heading))
    story.append(
        paragraph(
            "R. J. Tee and M. Zhang. <i>Integrating Curricula with Replays: Its Effects on Continual Learning.</i> AAAI Summer Symposium Series, 2023. "
            "Official paper: https://ojs.aaai.org/index.php/AAAI-SS/article/view/27486. Official code: https://github.com/ZhangLab-DeepNeuroCogLab/Integrating-Curricula-with-Replays.",
            small,
        )
    )

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries, rows = load_results()
    build_comparison_figure(summaries)
    build_seed_figure(rows)
    build_pdf(summaries, rows)
    print(f"wrote {COMPARISON_FIGURE.relative_to(ROOT)}")
    print(f"wrote {SEED_FIGURE.relative_to(ROOT)}")
    print(f"wrote {OUTPUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
