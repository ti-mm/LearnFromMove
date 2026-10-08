#!/usr/bin/env python3
"""Serve a minimal static Interaction rotation CAPTCHA replay page.

The page intentionally avoids the Astro runtime and external browser requests,
but keeps the DOM selectors, canvas drawing formula, slider interaction, and
success status semantics used by ``InteractionCaptchaEnv``.
"""
from __future__ import annotations

import argparse
import mimetypes
import posixpath
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLIC_ROOT = REPO_ROOT / "apps/rotation-captcha-web/public"
CAPTCHA_PATHS = {"/", "/posts/minimind_train_ppo", "/posts/minimind_train_ppo/"}


def resolve_public_asset_path(request_path: str, public_root: Path) -> Path | None:
    """Resolve a URL path to a file under ``public_root`` without traversal."""
    parsed_path = urlparse(request_path).path
    normalized = posixpath.normpath(unquote(parsed_path)).lstrip("/")
    if not normalized or normalized.startswith("../") or "/../" in f"/{normalized}/":
        return None

    root = public_root.resolve()
    candidate = root / normalized
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def build_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interaction CAPTCHA Replay</title>
  <style>
    html, body {
      margin: 0;
      min-height: 100%;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      background: #f1dfbd;
      color: #1c1d20;
    }

    *, *::before, *::after {
      box-sizing: border-box;
    }

    .article-captcha-gate {
      position: relative;
      min-height: 100vh;
    }

    .article-captcha-content {
      position: relative;
      min-height: 100vh;
      display: grid;
      place-items: center;
      opacity: 0.24;
      filter: blur(18px);
      pointer-events: none;
      user-select: none;
    }

    .article-captcha-content::before {
      content: "";
      width: min(980px, calc(100vw - 96px));
      height: min(620px, calc(100vh - 96px));
      border-radius: 18px;
      background:
        linear-gradient(rgba(255, 255, 255, 0.18) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.16) 1px, transparent 1px),
        linear-gradient(135deg, #f6ead1 0%, #efe0bf 44%, #e7d4a8 100%);
      background-size: 22px 22px, 22px 22px, auto;
      border: 1px solid rgba(28, 29, 32, 0.08);
    }

    .article-captcha-overlay {
      --article-captcha-overlay-padding: clamp(12px, 2.8dvh, 24px);
      --article-captcha-canvas-limit: 700px;
      position: fixed;
      inset: 0;
      z-index: 110;
      display: grid;
      place-items: center;
      padding: clamp(12px, 2.8dvh, 24px) 18px;
      overflow-y: auto;
      overscroll-behavior: contain;
    }

    .article-captcha-overlay-backdrop {
      position: absolute;
      inset: 0;
      background:
        linear-gradient(rgba(255, 255, 255, 0.18) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.16) 1px, transparent 1px),
        radial-gradient(circle at top left, rgba(255, 237, 196, 0.92), transparent 42%),
        radial-gradient(circle at right 18%, rgba(214, 132, 68, 0.28), transparent 28%),
        linear-gradient(135deg, #f6ead1 0%, #efe0bf 44%, #e7d4a8 100%);
      background-size: 22px 22px, 22px 22px, auto, auto, auto;
      backdrop-filter: blur(8px);
    }

    .article-captcha-card {
      --article-captcha-paper: #f7f0dc;
      --article-captcha-ink: #1c1d20;
      --article-captcha-accent: #c96a2c;
      position: relative;
      z-index: 1;
      width: fit-content;
      max-width: min(100%, 860px);
      max-height: calc(100dvh - (var(--article-captcha-overlay-padding) * 2));
      padding: clamp(16px, 2.2dvh, 22px);
      border: 1px solid rgba(28, 29, 32, 0.1);
      border-radius: 28px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(255, 248, 235, 0.88)),
        var(--article-captcha-paper);
      box-shadow: 0 24px 60px rgba(39, 30, 14, 0.22);
      backdrop-filter: blur(10px);
      display: grid;
      gap: clamp(12px, 1.8dvh, 18px);
      overflow: hidden;
    }

    .article-captcha-header {
      display: grid;
      gap: 0;
    }

    .article-captcha-title {
      margin: 0;
      font-size: clamp(1.5rem, 2.6vw, 2rem);
      line-height: 1.1;
      color: #1c1d20;
    }

    .article-captcha-sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }

    .article-captcha-canvas-frame {
      width: min(100%, var(--article-captcha-canvas-limit, 700px));
      justify-self: center;
      padding: clamp(8px, 1.2dvh, 12px);
      border-radius: 24px;
      background:
        linear-gradient(145deg, rgba(255, 255, 255, 0.84), rgba(255, 243, 219, 0.92));
      border: 1px solid rgba(28, 29, 32, 0.08);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }

    .article-captcha-canvas {
      display: block;
      width: 100%;
      max-width: 100%;
      height: auto;
      margin-inline: auto;
      border-radius: 18px;
      background: #f2e7cf;
    }

    /*
     * The legacy 1920x1080 replay deliberately keeps the historical layout
     * above.  Paper-bound 720p runs need a second constraint: several audited
     * challenges have portrait intrinsic canvases (up to 760x1352), so a
     * width-only fit clips the puzzle below the viewport.  At short viewports,
     * let the replaced canvas preserve its intrinsic aspect ratio while also
     * fitting the available vertical space.  The 900px media boundary keeps
     * the legacy 1080p rendering byte-for-byte/layout compatible.
     */
    @media (max-height: 899px) {
      .article-captcha-canvas {
        width: auto;
        max-height: calc(100dvh - 164px);
      }
    }

    .article-captcha-controls {
      position: absolute;
      top: var(--article-captcha-controls-top, 18px);
      left: var(--article-captcha-controls-left, 18px);
      z-index: 2;
      width: 360px;
      display: grid;
      gap: 10px;
      padding: 14px;
      border-radius: 24px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(255, 245, 226, 0.86)),
        rgba(247, 240, 220, 0.92);
      border: 1px solid rgba(28, 29, 32, 0.08);
      box-shadow: 0 20px 44px rgba(39, 30, 14, 0.2);
      backdrop-filter: blur(10px);
      opacity: 0;
      pointer-events: none;
      transition: opacity 180ms ease;
    }

    .article-captcha-controls[data-positioned="true"] {
      opacity: 1;
      pointer-events: auto;
    }

    .article-captcha-slider-panel {
      padding: 0;
      height: 20px;
    }

    .article-captcha-slider {
      width: 100%;
      margin: 0;
      appearance: none;
      height: 20px;
      border-radius: 999px;
      border: 1px solid rgba(28, 29, 32, 0.08);
      background:
        linear-gradient(
          90deg,
          #c96a2c 0%,
          #f1a955 var(--range-progress, 0%),
          rgba(28, 29, 32, 0.1) var(--range-progress, 0%),
          rgba(28, 29, 32, 0.1) 100%
        );
      box-shadow: inset 0 1px 3px rgba(28, 29, 32, 0.12);
    }

    .article-captcha-slider:disabled {
      opacity: 0.72;
      cursor: not-allowed;
    }

    .article-captcha-slider::-webkit-slider-thumb {
      appearance: none;
      width: 28px;
      height: 28px;
      border: 0;
      border-radius: 50%;
      background:
        radial-gradient(circle at 32% 32%, #fff7e8 0%, #fbe6b7 34%, #d97832 36%, #8e4318 100%);
      box-shadow:
        0 8px 20px rgba(72, 30, 6, 0.24),
        0 0 0 3px rgba(255, 255, 255, 0.5);
      cursor: grab;
    }

    .article-captcha-slider::-moz-range-thumb {
      width: 28px;
      height: 28px;
      border: 0;
      border-radius: 50%;
      background:
        radial-gradient(circle at 32% 32%, #fff7e8 0%, #fbe6b7 34%, #d97832 36%, #8e4318 100%);
      box-shadow:
        0 8px 20px rgba(72, 30, 6, 0.24),
        0 0 0 3px rgba(255, 255, 255, 0.5);
      cursor: grab;
    }

    .article-captcha-status-panel {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: flex-start;
    }

    .article-captcha-status {
      margin: 0;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(28, 29, 32, 0.08);
      color: #1c1d20;
      font-weight: 700;
    }

    .article-captcha-status[data-state="success"] {
      background: rgba(43, 142, 104, 0.16);
      color: #1d6b4c;
    }

    .article-captcha-status[data-state="error"] {
      background: rgba(196, 78, 38, 0.16);
      color: #9b3b18;
    }

    .article-captcha-status[data-state="loading"] {
      background: rgba(58, 96, 154, 0.16);
      color: #274f83;
    }
  </style>
</head>
<body class="article-captcha-locked">
  <div
    data-article-captcha-gate
    class="article-captcha-gate"
    data-storage-key="site-captcha:passed"
    data-background-image-url=""
    data-current-background-image-url=""
    data-gate-state="locked"
  >
    <div class="article-captcha-content" data-article-captcha-content></div>
    <div
      class="article-captcha-overlay"
      data-article-captcha-overlay
      role="dialog"
      aria-modal="true"
      aria-labelledby="article-captcha-title"
      aria-describedby="article-captcha-description"
    >
      <div class="article-captcha-overlay-backdrop"></div>
      <div class="article-captcha-card" data-article-captcha-card>
        <div class="article-captcha-header">
          <h2 id="article-captcha-title" class="article-captcha-title">Security Check</h2>
          <p id="article-captcha-description" class="article-captcha-sr-only">
            Drag the slider to complete verification.
          </p>
        </div>
        <div class="article-captcha-canvas-frame" data-article-captcha-canvas-frame>
          <canvas
            class="article-captcha-canvas"
            data-article-captcha-canvas
            width="760"
            height="508"
            aria-label="Rotation CAPTCHA canvas"
          ></canvas>
        </div>
      </div>
      <div
        class="article-captcha-controls"
        data-article-captcha-controls
        data-positioned="false"
      >
        <div class="article-captcha-slider-panel">
          <label class="article-captcha-sr-only" for="article-captcha-slider">
            Drag the slider to complete verification
          </label>
          <input
            id="article-captcha-slider"
            class="article-captcha-slider"
            data-article-captcha-slider
            type="range"
            min="0"
            max="100"
            value="0"
            step="0.01"
          >
        </div>
        <div class="article-captcha-status-panel">
          <p
            class="article-captcha-status"
            data-article-captcha-status
            data-state="loading"
            role="status"
            aria-live="polite"
          >
            Loading verification...
          </p>
        </div>
      </div>
    </div>
  </div>
  <script>
    (() => {
      const FULL_ROTATION_DEG = 360;
      const CAPTCHA_INSTRUCTION_TEXT = "Drag the slider to complete verification";
      const CAPTCHA_SUCCESS_TEXT = "Verification passed, opening the page...";
      const CAPTCHA_RETRY_TEXT = "Verification failed, please try again";
      const CAPTCHA_LOAD_ERROR_TEXT = "Verification failed to load. Refresh the page and try again.";
      const ROTATION_REGION_OUTER = "outer";

      const root = document.querySelector("[data-article-captcha-gate]");
      const overlay = document.querySelector("[data-article-captcha-overlay]");
      const content = document.querySelector("[data-article-captcha-content]");
      const canvasFrame = document.querySelector("[data-article-captcha-canvas-frame]");
      const controls = document.querySelector("[data-article-captcha-controls]");
      const canvas = document.querySelector("[data-article-captcha-canvas]");
      const slider = document.querySelector("[data-article-captcha-slider]");
      const status = document.querySelector("[data-article-captcha-status]");
      const context = canvas.getContext("2d");
      const encodedSpec = new URLSearchParams(window.location.search).get("replaySpec");
      let querySpec = {};
      if (encodedSpec) {
        try {
          querySpec = JSON.parse(encodedSpec);
        } catch (_error) {
          querySpec = {};
        }
      }
      const spec = window.__guiAgentCaptchaReplaySpec || querySpec;
      const challenge = spec.challenge || {};
      const rotationRegion =
        challenge.rotationRegion === ROTATION_REGION_OUTER ? ROTATION_REGION_OUTER : "center";
      let backgroundImage = null;
      let currentRotationDeg = Number(challenge.startRotationDeg || 0);
      let sliderValue = Number(challenge.startSliderValue || challenge.sliderMinValue || 0);
      let pointerActive = false;
      let isLocked = false;

      function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
      }

      function normalizeRotationDeg(value) {
        const normalized = value % FULL_ROTATION_DEG;
        return normalized < 0 ? normalized + FULL_ROTATION_DEG : normalized;
      }

      function getRotationDeltaDeg(fromDeg, toDeg) {
        const from = normalizeRotationDeg(fromDeg);
        const to = normalizeRotationDeg(toDeg);
        let delta = to - from;
        if (delta > 180) {
          delta -= FULL_ROTATION_DEG;
        }
        if (delta <= -180) {
          delta += FULL_ROTATION_DEG;
        }
        return delta;
      }

      function getShortestDistanceDeg(firstDeg, secondDeg) {
        return Math.abs(getRotationDeltaDeg(firstDeg, secondDeg));
      }

      function sliderValueToRotation(value) {
        const nextSliderValue = clamp(
          Number(value),
          Number(challenge.sliderMinValue || 0),
          Number(challenge.sliderMaxValue || 100)
        );
        return normalizeRotationDeg(
          Number(challenge.targetRotationDeg || 0) +
          (nextSliderValue - Number(challenge.targetSliderValue || 0)) *
            Number(challenge.degreesPerSliderUnit || 0) *
            Number(challenge.sensitivityScale ?? 1) *
            Number(challenge.rotationDirection ?? 1)
        );
      }

      function updateSliderVisual(value) {
        const min = Number(slider.min || 0);
        const max = Number(slider.max || 1);
        const span = Math.max(max - min, 1);
        const percentage = ((Number(value) - min) / span) * 100;
        slider.style.setProperty("--range-progress", `${percentage}%`);
      }

      function setStatus(state, message) {
        status.dataset.state = state;
        status.textContent = message;
      }

      function setGateLocked(locked) {
        root.dataset.gateState = locked ? "locked" : "passed";
        overlay.hidden = !locked;
        content.inert = locked;
        overlay.setAttribute("aria-hidden", locked ? "false" : "true");
        document.body.classList.toggle("article-captcha-locked", locked);
      }

      function setCanvasSizeFromChallenge() {
        const center = challenge.circleCenter || {};
        const width = Math.max(1, Math.round(Number(center.x || 380) * 2));
        const height = Math.max(1, Math.round(Number(center.y || 254) * 2));
        canvas.width = width;
        canvas.height = height;
        overlay.style.setProperty("--article-captcha-canvas-limit", `${width}px`);
      }

      function drawRotatedPiece(circlePath, centerX, centerY, radius) {
        context.save();
        context.translate(centerX, centerY);
        context.rotate((currentRotationDeg * Math.PI) / 180);
        context.translate(-centerX, -centerY);
        context.clip(circlePath);
        context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);

        const highlight = context.createLinearGradient(
          centerX - 24,
          centerY - 24,
          centerX + 24,
          centerY + 24
        );
        highlight.addColorStop(0, "rgba(255, 255, 255, 0.08)");
        highlight.addColorStop(0.55, "rgba(255, 255, 255, 0.03)");
        highlight.addColorStop(1, "rgba(28, 29, 32, 0.02)");
        context.fillStyle = highlight;
        context.fill(circlePath);
        context.restore();

        const seamColor =
          status.dataset.state === "success"
            ? "rgba(29, 107, 76, 0.16)"
            : status.dataset.state === "error"
              ? "rgba(155, 59, 24, 0.16)"
              : "rgba(20, 30, 42, 0.12)";
        context.save();
        context.strokeStyle = seamColor;
        context.lineWidth = 1.25;
        context.shadowColor = "rgba(255, 255, 255, 0.12)";
        context.shadowBlur = 8;
        context.stroke(circlePath);
        context.restore();
      }

      function drawRotatedOuter(circlePath, centerX, centerY, radius) {
        context.save();
        context.fillStyle = "#f2e7cf";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.beginPath();
        context.rect(0, 0, canvas.width, canvas.height);
        context.arc(centerX, centerY, radius, 0, Math.PI * 2);
        context.clip("evenodd");
        context.translate(centerX, centerY);
        context.rotate((currentRotationDeg * Math.PI) / 180);
        context.translate(-centerX, -centerY);
        context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);
        context.restore();

        context.save();
        context.clip(circlePath);
        context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);
        const highlight = context.createLinearGradient(
          centerX - 24,
          centerY - 24,
          centerX + 24,
          centerY + 24
        );
        highlight.addColorStop(0, "rgba(255, 255, 255, 0.08)");
        highlight.addColorStop(0.55, "rgba(255, 255, 255, 0.03)");
        highlight.addColorStop(1, "rgba(28, 29, 32, 0.02)");
        context.fillStyle = highlight;
        context.fill(circlePath);
        context.restore();

        const seamColor =
          status.dataset.state === "success"
            ? "rgba(29, 107, 76, 0.16)"
            : status.dataset.state === "error"
              ? "rgba(155, 59, 24, 0.16)"
              : "rgba(20, 30, 42, 0.12)";
        context.save();
        context.strokeStyle = seamColor;
        context.lineWidth = 1.25;
        context.shadowColor = "rgba(255, 255, 255, 0.12)";
        context.shadowBlur = 8;
        context.stroke(circlePath);
        context.restore();
      }

      function drawPairedRelativeScene(circlePath, centerX, centerY, radius) {
        const canonicalInitialRelativeDeg = Number(
          challenge.canonicalInitialRelativeDeg ?? challenge.startRotationDeg ?? 0
        );
        const outerRotationDeg = rotationRegion === ROTATION_REGION_OUTER
          ? canonicalInitialRelativeDeg - currentRotationDeg
          : 0;
        const centerRotationDeg = rotationRegion === ROTATION_REGION_OUTER
          ? canonicalInitialRelativeDeg
          : currentRotationDeg;

        context.save();
        context.fillStyle = "#f2e7cf";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.beginPath();
        context.rect(0, 0, canvas.width, canvas.height);
        context.arc(centerX, centerY, radius, 0, Math.PI * 2);
        context.clip("evenodd");
        context.translate(centerX, centerY);
        context.rotate((outerRotationDeg * Math.PI) / 180);
        context.translate(-centerX, -centerY);
        context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);
        context.restore();

        context.save();
        context.translate(centerX, centerY);
        context.rotate((centerRotationDeg * Math.PI) / 180);
        context.translate(-centerX, -centerY);
        context.clip(circlePath);
        context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);
        context.restore();

        const highlight = context.createLinearGradient(
          centerX - 24,
          centerY - 24,
          centerX + 24,
          centerY + 24
        );
        highlight.addColorStop(0, "rgba(255, 255, 255, 0.08)");
        highlight.addColorStop(0.55, "rgba(255, 255, 255, 0.03)");
        highlight.addColorStop(1, "rgba(28, 29, 32, 0.02)");
        context.save();
        context.fillStyle = highlight;
        context.fill(circlePath);
        context.strokeStyle = "rgba(20, 30, 42, 0.12)";
        context.lineWidth = 1.25;
        context.stroke(circlePath);
        context.restore();
      }

      function drawScene() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        if (!backgroundImage) {
          return;
        }
        const center = challenge.circleCenter || {};
        const centerX = Number(center.x || canvas.width / 2);
        const centerY = Number(center.y || canvas.height / 2);
        const radius = Number(challenge.circleRadius || Math.min(canvas.width, canvas.height) * 0.18);
        const circlePath = new Path2D();
        circlePath.arc(centerX, centerY, radius, 0, Math.PI * 2);

        if (challenge.pairedRelativeRotation === true) {
          drawPairedRelativeScene(circlePath, centerX, centerY, radius);
        } else if (rotationRegion === ROTATION_REGION_OUTER) {
          drawRotatedOuter(circlePath, centerX, centerY, radius);
        } else {
          context.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);
          drawRotatedPiece(circlePath, centerX, centerY, radius);
        }
      }

      function syncRotation(value) {
        sliderValue = clamp(
          Number(value),
          Number(challenge.sliderMinValue || 0),
          Number(challenge.sliderMaxValue || 100)
        );
        currentRotationDeg = sliderValueToRotation(sliderValue);
        slider.value = String(sliderValue);
        updateSliderVisual(sliderValue);
        drawScene();
      }

      function configureSlider() {
        slider.min = String(challenge.sliderMinValue ?? 0);
        slider.max = String(challenge.sliderMaxValue ?? 100);
        slider.step = "0.01";
        slider.disabled = false;
        syncRotation(challenge.startSliderValue ?? slider.min);
      }

      function configureControlsPosition() {
        const sliderBox = spec.sliderBox || {};
        const sliderX = Number(sliderBox.x);
        const sliderY = Number(sliderBox.y);
        const sliderWidth = Number(sliderBox.width);
        if (!Number.isFinite(sliderX) || !Number.isFinite(sliderY)) {
          return;
        }
        const panelPadding = 14;
        controls.style.setProperty(
          "--article-captcha-controls-left",
          `${Math.max(0, sliderX - panelPadding)}px`
        );
        controls.style.setProperty(
          "--article-captcha-controls-top",
          `${Math.max(0, sliderY - panelPadding)}px`
        );
        if (Number.isFinite(sliderWidth) && sliderWidth > 0) {
          controls.style.width = `${sliderWidth + panelPadding * 2}px`;
        }
      }

      function releasePointer() {
        if (!pointerActive || isLocked) {
          return;
        }
        pointerActive = false;
        syncRotation(Number(slider.value));
        const deltaDeg = getShortestDistanceDeg(
          currentRotationDeg,
          Number(challenge.targetRotationDeg || 0)
        );
        const MAX_ROTATION_TOLERANCE_DEG = 5;
        const rawToleranceDeg = Number(challenge.rotationToleranceDeg ?? MAX_ROTATION_TOLERANCE_DEG);
        const toleranceDeg = Math.min(
          Number.isFinite(rawToleranceDeg) ? rawToleranceDeg : MAX_ROTATION_TOLERANCE_DEG,
          MAX_ROTATION_TOLERANCE_DEG
        );
        window.__guiAgentCaptchaLastAttempt = {
          rotationRegion,
          sliderValue: Number(slider.value),
          currentRotationDeg,
          targetRotationDeg: Number(challenge.targetRotationDeg || 0),
          toleranceDeg,
          deltaDeg,
          success: deltaDeg <= toleranceDeg,
        };
        if (deltaDeg <= toleranceDeg) {
          isLocked = true;
          slider.disabled = true;
          setStatus("success", CAPTCHA_SUCCESS_TEXT);
          root.dataset.gateState = "passed";
          drawScene();
          return;
        }
        setStatus("error", CAPTCHA_RETRY_TEXT);
        drawScene();
      }

      function initialize() {
        setGateLocked(true);
        root.dataset.rotationRegion = rotationRegion;
        if (challenge.pairedRelativeRotation === true) {
          controls.style.transition = "none";
        }
        setCanvasSizeFromChallenge();
        configureControlsPosition();
        configureSlider();
        setStatus("loading", "Loading verification...");
        const imageUrl = String(spec.backgroundImageUrl || root.dataset.backgroundImageUrl || "");
        root.dataset.backgroundImageUrl = imageUrl;
        root.dataset.currentBackgroundImageUrl = imageUrl;
        if (!imageUrl || !spec.challenge) {
          slider.disabled = true;
          setStatus("error", CAPTCHA_LOAD_ERROR_TEXT);
          return;
        }
        const image = new Image();
        image.onload = () => {
          backgroundImage = image;
          setStatus("idle", CAPTCHA_INSTRUCTION_TEXT);
          drawScene();
          controls.dataset.positioned = "true";
          slider.focus({ preventScroll: true });
        };
        image.onerror = () => {
          slider.disabled = true;
          setStatus("error", CAPTCHA_LOAD_ERROR_TEXT);
        };
        image.src = imageUrl;
      }

      slider.addEventListener("input", () => {
        if (isLocked) {
          return;
        }
        setStatus("idle", CAPTCHA_INSTRUCTION_TEXT);
        syncRotation(Number(slider.value));
      });
      slider.addEventListener("pointerdown", () => {
        pointerActive = true;
      });
      slider.addEventListener("mousedown", () => {
        pointerActive = true;
      });
      slider.addEventListener("touchstart", () => {
        pointerActive = true;
      }, { passive: true });
      document.addEventListener("pointerup", releasePointer);
      document.addEventListener("mouseup", releasePointer);
      document.addEventListener("touchend", releasePointer, { passive: true });
      document.addEventListener("touchcancel", releasePointer, { passive: true });

      window.__guiAgentCaptchaStaticReplayReady = false;
      const markReady = () => {
        window.__guiAgentCaptchaStaticReplayReady = true;
      };
      const observer = new MutationObserver(() => {
        if (status.dataset.state === "idle" || status.dataset.state === "error") {
          markReady();
          observer.disconnect();
        }
      });
      observer.observe(status, { attributes: true, attributeFilter: ["data-state"] });
      initialize();
    })();
  </script>
</body>
</html>
"""


class StaticReplayHandler(BaseHTTPRequestHandler):
    public_root: Path = DEFAULT_PUBLIC_ROOT

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path in CAPTCHA_PATHS:
            self._send_bytes(
                build_page().encode("utf-8"),
                content_type="text/html; charset=utf-8",
            )
            return

        asset_path = resolve_public_asset_path(self.path, self.public_root)
        if asset_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
        self._send_bytes(asset_path.read_bytes(), content_type=content_type)

    def _send_bytes(self, body: bytes, *, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def make_handler(public_root: Path) -> type[StaticReplayHandler]:
    class Handler(StaticReplayHandler):
        pass

    Handler.public_root = Path(public_root)
    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a static Interaction rotation CAPTCHA replay page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4321)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    args = parser.parse_args()

    handler = make_handler(args.public_root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Starting static Interaction CAPTCHA replay server at "
        f"http://{args.host}:{args.port}/posts/minimind_train_ppo/",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
