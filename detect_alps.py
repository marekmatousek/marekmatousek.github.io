#!/usr/bin/env python3
"""
Alps visibility detector for webcam.sumava.eu (Bučina).
- Stáhne aktuální obrázek z webkamery
- Vyhodnotí, zda jsou "Alpy vidět"
- Zapíše výsledek do docs/alps/status.json a docs/alps/preview.jpg

Konfigurační proměnné (volitelné) přes ENV:
  CAM_URL        - URL snímku (default: Bučina – příklad níže)
  ROI_X1..ROI_Y2 - ořez ROI v relativních souřadnicích (0..1)
  THR_EDGE       - minimální hustota hran v ROI (0–1)
  THR_LAP        - minimální ostrost (variance Laplacian)
  THR_CONTRAST   - minimální lokální kontrast (stddev 0–255)
"""
import os, io, json, time, math
from datetime import datetime, timezone
import numpy as np
import requests
import cv2

# ---------- Nastavení ----------
DEFAULT_CAM_URL = (
    # Snímková URL – můžeš nahradit přes ENV CAM_URL
    "https://webcam.sumava.eu/snapshot.php?place=2002"
)

# ROI (výchozí: úzký vodorovný pruh uprostřed, kde bývá horizont s Alpami)
ROI_X1 = float(os.getenv("ROI_X1", "0.30"))
ROI_Y1 = float(os.getenv("ROI_Y1", "0.42"))
ROI_X2 = float(os.getenv("ROI_X2", "0.70"))
ROI_Y2 = float(os.getenv("ROI_Y2", "0.58"))

# Prahy (lze doladit podle zkušeností)
THR_EDGE     = float(os.getenv("THR_EDGE", "0.10"))   # 10 % edge pixelů v ROI
THR_LAP      = float(os.getenv("THR_LAP", "25.0"))    # ostrost/struktura
THR_CONTRAST = float(os.getenv("THR_CONTRAST", "18")) # lokální kontrast (stddev)

OUT_DIR = os.path.join("docs", "alps")
STATUS_PATH = os.path.join(OUT_DIR, "status.json")
PREVIEW_PATH = os.path.join(OUT_DIR, "preview.jpg")

# ---------- Pomocné funkce ----------
def fetch_image(cam_url: str) -> np.ndarray:
    """Stáhne obrázek z kamery a vrátí BGR numpy array (OpenCV)."""
    headers = {"User-Agent": "AlpsDetector/1.0 (+https://github.com/)"}
    r = requests.get(cam_url, headers=headers, timeout=20)
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nepodařilo se dekódovat obrazová data.")
    return img

def resize_keep_aspect(img: np.ndarray, target_w: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    scale = target_w / w
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)

def analyze_roi(img_bgr: np.ndarray, roi_rect):
    x1, y1, x2, y2 = roi_rect
    h, w = img_bgr.shape[:2]
    # Ořez
    rx1 = max(0, min(w-1, int(round(x1*w))))
    ry1 = max(0, min(h-1, int(round(y1*h))))
    rx2 = max(rx1+1, min(w,   int(round(x2*w))))
    ry2 = max(ry1+1, min(h,   int(round(y2*h))))
    roi = img_bgr[ry1:ry2, rx1:rx2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Jemné odšumění
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Hrany (Canny) a jejich hustota
    v = np.median(gray_blur)
    lower = int(max(0, 0.66*v))
    upper = int(min(255, 1.33*v))  # adaptivní prahy
    edges = cv2.Canny(gray_blur, lower, upper, L2gradient=True)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    # Ostrost / struktura
    lap_var = float(cv2.Laplacian(gray_blur, cv2.CV_64F).var())

    # Lokální kontrast ve stupních šedi
    contrast = float(gray_blur.std())

    return {
        "edge_density": edge_density,
        "lap_var": lap_var,
        "contrast": contrast,
        "roi_px": (rx1, ry1, rx2, ry2),
        "edges": edges,
    }

def decide_visible(metrics):
    """Heuristické rozhodnutí: Alpy viditelné při dostatečné hranatosti, ostrosti a kontrastu."""
    return (metrics["edge_density"] >= THR_EDGE and
            metrics["lap_var"]      >= THR_LAP and
            metrics["contrast"]     >= THR_CONTRAST)

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def annotate_preview(img, roi_px, verdict_text, metrics):
    x1, y1, x2, y2 = roi_px
    preview = img.copy()
    # Poloprůhledný pruh nahoře pro text
    bar_h = max(36, img.shape[0]//18)
    overlay = preview.copy()
    cv2.rectangle(overlay, (0,0), (preview.shape[1], bar_h), (0,0,0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, preview, 1-alpha, 0, preview)

    # ROI rámeček
    color = (0, 200, 0) if "VIDĚT" in verdict_text else (0, 0, 220)
    cv2.rectangle(preview, (x1,y1), (x2,y2), color, 2)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    t = verdict_text
    cv2.putText(preview, t, (12, bar_h-10), font, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # Malý panel s metrikami
    panel = f"edge={metrics['edge_density']:.3f} | lap={metrics['lap_var']:.1f} | σ={metrics['contrast']:.1f}"
    cv2.putText(preview, panel, (12, bar_h+24), font, 0.6, (255,255,255), 2, cv2.LINE_AA)

    return preview

# ---------- Hlavní běh ----------
def main():
    ensure_outdir()
    cam_url = os.getenv("CAM_URL", DEFAULT_CAM_URL)

    now = datetime.now(timezone.utc)
    iso_now = now.isoformat()

    try:
        img0 = fetch_image(cam_url)
        img = resize_keep_aspect(img0, 1280)

        roi_rect = (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)
        metrics = analyze_roi(img, roi_rect)
        visible = decide_visible(metrics)

        verdict = "ALPY JSOU VIDĚT – běžte ven, je inverze a skvělá dohlednost!" if visible \
                  else "Alpy nejsou vidět – dohlednost slabá / oblačno."

        preview = annotate_preview(
            img,
            metrics["roi_px"],
            verdict_text=verdict,
            metrics=metrics
        )
        # Uložení výstupů
        ok1 = cv2.imwrite(PREVIEW_PATH, preview)
        status = {
            "timestamp_utc": iso_now,
            "alps_visible": bool(visible),
            "message": verdict,
            "metrics": {
                "edge_density": round(metrics["edge_density"], 4),
                "lap_var": round(metrics["lap_var"], 2),
                "contrast": round(metrics["contrast"], 2),
                "thresholds": {
                    "THR_EDGE": THR_EDGE,
                    "THR_LAP": THR_LAP,
                    "THR_CONTRAST": THR_CONTRAST
                }
            },
            "roi": {
                "relative": {"x1": ROI_X1, "y1": ROI_Y1, "x2": ROI_X2, "y2": ROI_Y2},
                "pixels": {
                    "x1": metrics["roi_px"][0], "y1": metrics["roi_px"][1],
                    "x2": metrics["roi_px"][2], "y2": metrics["roi_px"][3],
                }
            },
            "camera_url": cam_url,
            "preview_path": "preview.jpg",
            "ok_preview_saved": bool(ok1),
        }
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

        print(json.dumps({"ok": True, "alps_visible": visible, "ts": iso_now}))
    except Exception as e:
        # Zápis chybového statusu, ať je na stránce vidět, co se stalo
        err = {
            "timestamp_utc": iso_now,
            "alps_visible": False,
            "message": "Chyba při detekci",
            "error": str(e),
            "camera_url": os.getenv("CAM_URL", DEFAULT_CAM_URL),
        }
        ensure_outdir()
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(err, f, ensure_ascii=False, indent=2)
        # vytvoř i placeholder náhled, ať HTML nepadá
        if not os.path.exists(PREVIEW_PATH):
            blank = np.zeros((480, 854, 3), dtype=np.uint8)
            cv2.putText(blank, "Chyba detekce – viz status.json",
                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imwrite(PREVIEW_PATH, blank)
        print(json.dumps({"ok": False, "error": str(e)}))

if __name__ == "__main__":
    main()

