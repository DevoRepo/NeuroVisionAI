# ------------------------------------------------------------
# Core libraries used throughout this notebook
# ------------------------------------------------------------
import os
import math
import warnings
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Optional

import numpy as np
import cv2                                  # OpenCV -> image loading & classical CV operations
import pandas as pd                         # Tabular summary / final report
import matplotlib.pyplot as plt             # All visualizations (dashboard, charts, gauge)
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from PIL import Image                       # Robust image reading (fallback / format support)
from skimage.metrics import structural_similarity as ssim  # (kept available for future extensions)
from skimage import img_as_float

warnings.filterwarnings("ignore")

# Reproducibility for the synthetic demo image generated later in the notebook
np.random.seed(42)

print("✅ Libraries imported successfully.")
print(f"OpenCV version   : {cv2.__version__}")
print(f"NumPy version    : {np.__version__}")
print(f"Pandas version   : {pd.__version__}")


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_mri_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    '''Load an MRI image from disk and return both the display (RGB) and grayscale versions.

    Args:
        image_path: Path to the MRI image file.

    Returns:
        A tuple ``(rgb_image, gray_image)`` where:
            - ``rgb_image`` is an (H, W, 3) uint8 array suitable for display with matplotlib.
            - ``gray_image`` is an (H, W) uint8 array used for all quality-metric computations.

    Raises:
        FileNotFoundError: If ``image_path`` does not point to an existing file.
        ValueError: If the file extension is unsupported, the file cannot be decoded,
            or the resulting image is empty.
    '''
    # --- 1. Path existence check -------------------------------------------------
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        raise FileNotFoundError(f"❌ Invalid path — no such file: '{image_path}'")

    # --- 2. Extension / format check ----------------------------------------------
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"❌ Unsupported image format '{ext}'. Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    # --- 3. Attempt to decode the image (corrupted-file check) ---------------------
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        # Fall back to PIL in case OpenCV fails on a valid-but-unusual encoding
        try:
            pil_img = Image.open(image_path)
            pil_img.verify()  # raises if corrupted
            raw = np.array(Image.open(image_path).convert("RGB"))
        except Exception as exc:
            raise ValueError(f"❌ Corrupted or unreadable image file: '{image_path}' ({exc})")

    # --- 4. Empty-image check -------------------------------------------------------
    if raw.size == 0 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError(f"❌ Empty image detected: '{image_path}'")

    # --- 5. Normalize channel layout -> produce RGB (for display) and Gray (for metrics)
    if raw.ndim == 2:
        # Already grayscale
        gray = raw
        rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif raw.ndim == 3 and raw.shape[2] == 4:
        # RGBA -> drop alpha
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
    elif raw.ndim == 3 and raw.shape[2] == 3:
        # BGR (OpenCV default) -> RGB for display, standard grayscale for metrics
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"❌ Unsupported image shape: {raw.shape}")

    # --- 6. NaN / dtype safety -------------------------------------------------------
    gray = np.nan_to_num(gray).astype(np.uint8)
    rgb = np.nan_to_num(rgb).astype(np.uint8)

    return rgb, gray


def generate_synthetic_mri_demo(size: int = 256, blur_sigma: float = 1.0,
                                 noise_sigma: float = 5.0) -> np.ndarray:
    '''Generate a synthetic brain-MRI-like grayscale image for demonstration purposes.

    This is used ONLY when a real MRI path is not available in this environment, so that the
    notebook remains fully runnable end-to-end. Replace ``IMAGE_PATH`` below with a real Brain
    MRI slice (e.g. from the Kaggle Brain Tumor MRI Dataset) for genuine results.
    '''
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2

    # Skull (outer ellipse) + brain tissue (inner ellipse) + a bright "lesion" blob
    skull = ((xx - cx) ** 2 / (size * 0.42) ** 2 + (yy - cy) ** 2 / (size * 0.36) ** 2) <= 1
    brain = ((xx - cx) ** 2 / (size * 0.36) ** 2 + (yy - cy) ** 2 / (size * 0.30) ** 2) <= 1
    lesion = ((xx - (cx + size * 0.12)) ** 2 + (yy - (cy - size * 0.05)) ** 2) <= (size * 0.06) ** 2

    img = np.zeros((size, size), dtype=np.float32)
    img[skull] = 60
    img[brain] = 150
    img[lesion] = 210

    img = cv2.GaussianBlur(img, (0, 0), sigmaX=blur_sigma)
    noise = np.random.normal(0, noise_sigma, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


# ------------------------------------------------------------------
# 👉 SET THE PATH TO YOUR BRAIN MRI IMAGE HERE
# ------------------------------------------------------------------
IMAGE_PATH = "sample_brain_mri.png"   # <-- Replace with your own MRI image path

if os.path.isfile(IMAGE_PATH):
    rgb_image, gray_image = load_mri_image(IMAGE_PATH)
    print(f"✅ Loaded real MRI image from: {IMAGE_PATH}")
else:
    print(f"⚠️  '{IMAGE_PATH}' not found — generating a synthetic demo MRI image instead.")
    print("   Replace IMAGE_PATH above with a real Brain MRI slice to analyze your own data.")
    gray_image = generate_synthetic_mri_demo()
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(IMAGE_PATH, gray_image)  # save so future runs can reuse it

print(f"Image shape (grayscale): {gray_image.shape}")
print(f"Image dtype            : {gray_image.dtype}")


# Display the loaded MRI image
plt.figure(figsize=(4.5, 4.5))
plt.imshow(gray_image, cmap="gray")
plt.title("Original Brain MRI (Grayscale)", fontsize=12, fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.show()


def compute_sharpness(gray: np.ndarray, ref_max: float = 1500.0) -> Dict[str, float]:
    '''Compute image sharpness using the Variance-of-Laplacian method.

    Args:
        gray: Grayscale MRI image (uint8, 2D array).
        ref_max: Reference variance value that maps to a normalized score of 1.0.
            (Empirically chosen for typical 256x256 Brain MRI slices; tune per dataset.)

    Returns:
        Dictionary with raw and normalized sharpness plus a human-readable label.
    '''
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    raw_sharpness = float(np.var(laplacian))

    # Guard against divide-by-zero / NaN on a completely flat (blank) image
    if ref_max <= 0 or np.isnan(raw_sharpness):
        normalized = 0.0
    else:
        normalized = float(np.clip(raw_sharpness / ref_max, 0.0, 1.0))

    if normalized >= 0.75:
        label = "Sharp"
    elif normalized >= 0.4:
        label = "Acceptable"
    else:
        label = "Blurry"

    return {"raw": raw_sharpness, "normalized": normalized, "label": label}


sharpness_result = compute_sharpness(gray_image)
print("── Sharpness ──────────────────────────────")
print(f"Raw Sharpness (Variance of Laplacian) : {sharpness_result['raw']:.2f}")
print(f"Normalized Sharpness (0–1)            : {sharpness_result['normalized']:.2f}")
print(f"Quality Label                         : {sharpness_result['label']}")


def compute_brightness(gray: np.ndarray) -> Dict[str, float]:
    '''Compute image brightness as the mean grayscale intensity.

    Args:
        gray: Grayscale MRI image (uint8, 2D array).

    Returns:
        Dictionary with raw brightness (0-255), normalized brightness (0-1),
        and an interpretation label.
    '''
    raw_brightness = float(np.mean(gray))
    normalized = float(np.clip(raw_brightness / 255.0, 0.0, 1.0))

    if normalized < 0.25:
        label = "Too Dark"
    elif normalized > 0.80:
        label = "Too Bright"
    else:
        label = "Normal"

    return {"raw": raw_brightness, "normalized": normalized, "label": label}


brightness_result = compute_brightness(gray_image)
print("── Brightness ─────────────────────────────")
print(f"Raw Brightness (mean intensity)  : {brightness_result['raw']:.2f} / 255")
print(f"Normalized Brightness (0–1)      : {brightness_result['normalized']:.2f}")
print(f"Interpretation                   : {brightness_result['label']}")


def compute_contrast(gray: np.ndarray, ref_max: float = 80.0) -> Dict[str, float]:
    '''Compute image contrast as the standard deviation of grayscale intensity.

    Args:
        gray: Grayscale MRI image (uint8, 2D array).
        ref_max: Reference std value that maps to a normalized score of 1.0.

    Returns:
        Dictionary with raw and normalized contrast plus a human-readable label.
    '''
    raw_contrast = float(np.std(gray))

    if ref_max <= 0 or np.isnan(raw_contrast):
        normalized = 0.0
    else:
        normalized = float(np.clip(raw_contrast / ref_max, 0.0, 1.0))

    if normalized >= 0.7:
        label = "High Contrast"
    elif normalized >= 0.35:
        label = "Moderate Contrast"
    else:
        label = "Low Contrast"

    return {"raw": raw_contrast, "normalized": normalized, "label": label}


contrast_result = compute_contrast(gray_image)
print("── Contrast ────────────────────────────────")
print(f"Raw Contrast (std. deviation)   : {contrast_result['raw']:.2f}")
print(f"Normalized Contrast (0–1)       : {contrast_result['normalized']:.2f}")
print(f"Quality Label                   : {contrast_result['label']}")


def compute_noise(gray: np.ndarray, kernel_size: int = 5, ref_max: float = 25.0) -> Dict[str, float]:
    '''Estimate image noise using the Median-Blur Residual method.

    Args:
        gray: Grayscale MRI image (uint8, 2D array).
        kernel_size: Median filter kernel size (must be odd).
        ref_max: Reference residual std value that maps to a normalized noise score of 1.0
            (i.e. maximally noisy).

    Returns:
        Dictionary with raw noise estimate, normalized noise score (0=clean, 1=very noisy),
        and a human-readable label.
    '''
    if kernel_size % 2 == 0:
        kernel_size += 1  # median filter requires an odd kernel

    median_filtered = cv2.medianBlur(gray, kernel_size)
    residual = gray.astype(np.float32) - median_filtered.astype(np.float32)
    raw_noise = float(np.std(residual))

    if ref_max <= 0 or np.isnan(raw_noise):
        normalized = 0.0
    else:
        normalized = float(np.clip(raw_noise / ref_max, 0.0, 1.0))

    if normalized <= 0.25:
        label = "Low Noise (Clean)"
    elif normalized <= 0.5:
        label = "Moderate Noise"
    else:
        label = "High Noise"

    return {"raw": raw_noise, "normalized": normalized, "label": label}


noise_result = compute_noise(gray_image)
print("── Noise ───────────────────────────────────")
print(f"Raw Noise Estimate (residual std) : {noise_result['raw']:.2f}")
print(f"Normalized Noise Score (0–1)      : {noise_result['normalized']:.2f}  (lower = cleaner)")
print(f"Quality Label                     : {noise_result['label']}")


@dataclass
class QualityWeights:
    '''Weights used to combine normalized metrics into the Overall Quality Score.'''
    sharpness: float = 0.40
    contrast: float = 0.25
    brightness: float = 0.20
    noise: float = 0.15

    def validate(self) -> None:
        total = self.sharpness + self.contrast + self.brightness + self.noise
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"❌ Quality weights must sum to 1.0 (got {total:.4f})")


def compute_overall_score(sharpness_norm: float, brightness_norm: float,
                           contrast_norm: float, noise_norm: float,
                           weights: QualityWeights = QualityWeights()) -> float:
    '''Combine four normalized quality metrics into a single 0-100 Overall Quality Score.

    Args:
        sharpness_norm: Normalized sharpness in [0, 1] (higher = better).
        brightness_norm: Normalized brightness in [0, 1] (higher = better, within reason).
        contrast_norm: Normalized contrast in [0, 1] (higher = better).
        noise_norm: Normalized noise in [0, 1] (higher = *worse*; inverted internally).
        weights: Weighting scheme for the four metrics (must sum to 1.0).

    Returns:
        Overall quality score in the range [0, 100].
    '''
    weights.validate()

    # Defensive clipping in case of any upstream NaN/out-of-range values
    s = float(np.clip(np.nan_to_num(sharpness_norm), 0.0, 1.0))
    b = float(np.clip(np.nan_to_num(brightness_norm), 0.0, 1.0))
    c = float(np.clip(np.nan_to_num(contrast_norm), 0.0, 1.0))
    n = float(np.clip(np.nan_to_num(noise_norm), 0.0, 1.0))

    weighted_sum = (
        weights.sharpness * s +
        weights.contrast * c +
        weights.brightness * b +
        weights.noise * (1.0 - n)   # inverted: lower noise -> higher contribution
    )
    return float(np.clip(weighted_sum * 100.0, 0.0, 100.0))


overall_score = compute_overall_score(
    sharpness_norm=sharpness_result["normalized"],
    brightness_norm=brightness_result["normalized"],
    contrast_norm=contrast_result["normalized"],
    noise_norm=noise_result["normalized"],
)

print(f"🏁 Overall Image Quality Score: {overall_score:.1f} / 100")


def get_quality_grade(score: float) -> str:
    '''Map a 0-100 Overall Quality Score to a categorical quality grade.

    Args:
        score: Overall quality score in [0, 100].

    Returns:
        One of "Excellent", "Good", "Fair", "Poor".
    '''
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Fair"
    else:
        return "Poor"


GRADE_COLORS = {
    "Excellent": "#1E8449",   # deep green
    "Good":      "#27AE60",   # green
    "Fair":      "#F39C12",   # amber
    "Poor":      "#C0392B",   # red
}

quality_grade = get_quality_grade(overall_score)
grade_color = GRADE_COLORS[quality_grade]

print(f"📋 Quality Grade: {quality_grade}")


def plot_quality_dashboard(gray: np.ndarray,
                            sharpness: Dict, brightness: Dict,
                            contrast: Dict, noise: Dict,
                            overall_score: float, grade: str) -> None:
    '''Render the full Image Quality Assessment dashboard using Matplotlib only.'''

    grade_color = GRADE_COLORS[grade]

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.05], hspace=0.45, wspace=0.35)

    fig.suptitle("NeuroVision AI — Image Quality Assessment Report",
                 fontsize=16, fontweight="bold", y=0.98)

    # --- (1) Original MRI Image --------------------------------------------------
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(gray, cmap="gray")
    ax_img.set_title("Original MRI", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    # --- (2) Bar chart of normalized metrics -------------------------------------
    ax_bar = fig.add_subplot(gs[0, 1])
    metric_names = ["Sharpness", "Brightness", "Contrast", "Noise"]
    metric_values = [sharpness["normalized"], brightness["normalized"],
                      contrast["normalized"], noise["normalized"]]
    bar_colors = ["#2E86C1", "#F1C40F", "#8E44AD", "#E74C3C"]

    bars = ax_bar.bar(metric_names, metric_values, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel("Normalized Value (0–1)")
    ax_bar.set_title("Quality Metrics", fontsize=12, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, metric_values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- (3) Donut / Gauge chart of Overall Score --------------------------------
    ax_donut = fig.add_subplot(gs[0, 2])
    frac = overall_score / 100.0
    wedge_sizes = [frac, 1 - frac]
    ax_donut.pie(
        wedge_sizes,
        colors=[grade_color, "#EAECEE"],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.35, edgecolor="white"),
    )
    ax_donut.text(0, 0.15, f"{overall_score:.0f}", ha="center", va="center",
                  fontsize=26, fontweight="bold", color=grade_color)
    ax_donut.text(0, -0.15, "/ 100", ha="center", va="center", fontsize=11, color="gray")
    ax_donut.set_title("Overall Quality Score", fontsize=12, fontweight="bold")

    # --- (4) Color-coded quality card ---------------------------------------------
    ax_card = fig.add_subplot(gs[1, 0])
    ax_card.axis("off")
    card = mpatches.FancyBboxPatch(
        (0.05, 0.15), 0.9, 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=2, edgecolor=grade_color, facecolor=grade_color, alpha=0.15,
        transform=ax_card.transAxes,
    )
    ax_card.add_patch(card)
    ax_card.text(0.5, 0.62, grade.upper(), ha="center", va="center",
                 fontsize=20, fontweight="bold", color=grade_color, transform=ax_card.transAxes)
    ax_card.text(0.5, 0.38, f"Overall Score: {overall_score:.1f}/100", ha="center", va="center",
                 fontsize=11, color="black", transform=ax_card.transAxes)
    ax_card.set_title("Quality Card", fontsize=12, fontweight="bold")

    # --- (5) Final summary table ---------------------------------------------------
    ax_table = fig.add_subplot(gs[1, 1:])
    ax_table.axis("off")
    table_data = [
        ["Metric", "Raw Value", "Normalized (0–1)", "Label"],
        ["Sharpness", f"{sharpness['raw']:.2f}", f"{sharpness['normalized']:.2f}", sharpness["label"]],
        ["Brightness", f"{brightness['raw']:.2f}", f"{brightness['normalized']:.2f}", brightness["label"]],
        ["Contrast", f"{contrast['raw']:.2f}", f"{contrast['normalized']:.2f}", contrast["label"]],
        ["Noise", f"{noise['raw']:.2f}", f"{noise['normalized']:.2f}", noise["label"]],
        ["Overall Score", "-", f"{overall_score:.1f}/100", grade],
    ]
    tbl = ax_table.table(cellText=table_data, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(table_data) - 1:
            cell.set_facecolor(grade_color)
            cell.set_alpha(0.25)
            cell.set_text_props(fontweight="bold")
    ax_table.set_title("Final Summary Table", fontsize=12, fontweight="bold", pad=20)

    plt.show()


plot_quality_dashboard(gray_image, sharpness_result, brightness_result,
                        contrast_result, noise_result, overall_score, quality_grade)


def print_final_report(sharpness: Dict, brightness: Dict, contrast: Dict,
                        noise: Dict, overall_score: float, grade: str) -> None:
    '''Print a neatly formatted, text-based Image Quality Report.'''
    width = 42
    print("─" * width)
    print("IMAGE QUALITY REPORT".center(width))
    print("─" * width)
    print(f"{'Sharpness':<20}{sharpness['normalized']:.2f}")
    print(f"{'Brightness':<20}{brightness['normalized']:.2f}")
    print(f"{'Contrast':<20}{contrast['normalized']:.2f}")
    print(f"{'Noise':<20}{noise['normalized']:.2f}")
    print("─" * width)
    print(f"{'Overall Score':<20}{overall_score:.0f}/100")
    print(f"{'Quality':<20}{grade}")
    print("─" * width)


print_final_report(sharpness_result, brightness_result, contrast_result,
                    noise_result, overall_score, quality_grade)


# Quick self-test of the error-handling paths (does not affect the main pipeline above)
test_cases = {
    "Non-existent path": "this_file_does_not_exist.png",
    "Unsupported extension": "some_scan.gif",
}

for name, path in test_cases.items():
    try:
        load_mri_image(path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[{name}] -> Handled correctly: {e}")


# ------------------------------------------------------------------
# FULL PIPELINE — run end-to-end on the current `gray_image` / `rgb_image`
# ------------------------------------------------------------------

def run_full_quality_assessment(image_path: str) -> pd.DataFrame:
    '''Run the complete Image Quality Assessment pipeline on a given MRI image path.

    Args:
        image_path: Path to a Brain MRI image on disk.

    Returns:
        A one-row pandas DataFrame summarizing every intermediate and final result —
        convenient for logging, batching over multiple images, or feeding into a
        downstream dashboard (e.g. the Streamlit app in the full NeuroVision AI system).
    '''
    rgb, gray = load_mri_image(image_path)

    sharp = compute_sharpness(gray)
    bright = compute_brightness(gray)
    contr = compute_contrast(gray)
    nz = compute_noise(gray)

    score = compute_overall_score(
        sharpness_norm=sharp["normalized"],
        brightness_norm=bright["normalized"],
        contrast_norm=contr["normalized"],
        noise_norm=nz["normalized"],
    )
    grade = get_quality_grade(score)

    print("\n================ STEP-BY-STEP INTERMEDIATE RESULTS ================\n")
    print(f"Image path: {image_path}")
    print(f"Image shape: {gray.shape}\n")

    print_final_report(sharp, bright, contr, nz, score, grade)
    plot_quality_dashboard(gray, sharp, bright, contr, nz, score, grade)

    return pd.DataFrame([{
        "image_path": image_path,
        "sharpness_raw": sharp["raw"], "sharpness_norm": sharp["normalized"],
        "brightness_raw": bright["raw"], "brightness_norm": bright["normalized"],
        "contrast_raw": contr["raw"], "contrast_norm": contr["normalized"],
        "noise_raw": nz["raw"], "noise_norm": nz["normalized"],
        "overall_score": score, "quality_grade": grade,
    }])


results_df = run_full_quality_assessment(IMAGE_PATH)
results_df
