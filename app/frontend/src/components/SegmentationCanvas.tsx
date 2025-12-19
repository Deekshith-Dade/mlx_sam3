"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import type { SegmentationResult } from "@/lib/api";

interface Props {
  imageUrl: string | null;
  imageWidth: number;
  imageHeight: number;
  result: SegmentationResult | null;
  boxMode: "positive" | "negative";
  onBoxDrawn: (box: number[]) => void;
  isLoading: boolean;
}

// Color palette for masks
const COLORS = [
  [59, 235, 161], // Emerald
  [96, 165, 250], // Blue
  [251, 191, 36], // Amber
  [248, 113, 113], // Red
  [167, 139, 250], // Violet
  [52, 211, 153], // Green
  [251, 146, 60], // Orange
  [147, 197, 253], // Light Blue
];

export function SegmentationCanvas({
  imageUrl,
  imageWidth,
  imageHeight,
  result,
  boxMode,
  onBoxDrawn,
  isLoading,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(
    null
  );
  const [currentPoint, setCurrentPoint] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [displayScale, setDisplayScale] = useState(1);

  // Calculate display scale to fit image in container
  useEffect(() => {
    if (!containerRef.current || !imageWidth || !imageHeight) return;

    const containerWidth = containerRef.current.clientWidth;
    const maxHeight = window.innerHeight * 0.7;

    const scaleX = containerWidth / imageWidth;
    const scaleY = maxHeight / imageHeight;
    const scale = Math.min(scaleX, scaleY, 1);

    setDisplayScale(scale);
  }, [imageWidth, imageHeight]);

  // Load and draw image
  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imageRef.current = img;
      drawCanvas();
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Redraw when result changes
  const drawCanvas = useCallback(() => {
    if (!canvasRef.current || !imageRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Use integer dimensions for canvas
    const displayWidth = Math.floor(imageWidth * displayScale);
    const displayHeight = Math.floor(imageHeight * displayScale);

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    // Clear canvas
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    // Draw image
    ctx.drawImage(imageRef.current, 0, 0, displayWidth, displayHeight);

    // Draw masks with semi-transparency
    if (result?.masks && result.masks.length > 0) {
      for (let i = 0; i < result.masks.length; i++) {
        const mask = result.masks[i];
        const box = result.boxes?.[i];
        const score = result.scores?.[i] ?? 0;
        const color = COLORS[i % COLORS.length];

        // Draw mask overlay
        if (mask && Array.isArray(mask) && mask.length > 0) {
          // Handle both [1, H, W] and [H, W] formats
          let maskData: number[][];
          if (Array.isArray(mask[0]) && Array.isArray(mask[0][0])) {
            // Shape is [1, H, W] - take first channel
            maskData = mask[0] as number[][];
          } else if (Array.isArray(mask[0])) {
            // Shape is [H, W] - use directly
            maskData = mask as unknown as number[][];
          } else {
            continue;
          }

          const maskH = maskData.length;
          const maskW = maskData[0]?.length || 0;
          if (maskH === 0 || maskW === 0) continue;

          // Create overlay with semi-transparent mask color
          const imgData = ctx.getImageData(0, 0, displayWidth, displayHeight);
          const data = imgData.data;

          // Scale factors from mask coordinates to display coordinates
          const scaleX = displayWidth / maskW;
          const scaleY = displayHeight / maskH;

          for (let my = 0; my < maskH; my++) {
            const row = maskData[my];
            if (!row) continue;
            for (let mx = 0; mx < row.length; mx++) {
              if (row[mx] > 0) {
                // Map mask pixel to display pixel(s)
                const dx = Math.floor(mx * scaleX);
                const dy = Math.floor(my * scaleY);

                if (
                  dx >= 0 &&
                  dx < displayWidth &&
                  dy >= 0 &&
                  dy < displayHeight
                ) {
                  const idx = (dy * displayWidth + dx) * 4;
                  // Blend with existing pixel (50% opacity)
                  data[idx] = Math.round(data[idx] * 0.5 + color[0] * 0.5);
                  data[idx + 1] = Math.round(
                    data[idx + 1] * 0.5 + color[1] * 0.5
                  );
                  data[idx + 2] = Math.round(
                    data[idx + 2] * 0.5 + color[2] * 0.5
                  );
                }
              }
            }
          }

          ctx.putImageData(imgData, 0, 0);
        }

        // Draw bounding box
        if (box) {
          const [x0, y0, x1, y1] = box;
          ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
          ctx.lineWidth = 2;
          ctx.strokeRect(
            x0 * displayScale,
            y0 * displayScale,
            (x1 - x0) * displayScale,
            (y1 - y0) * displayScale
          );

          // Draw score label
          ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
          ctx.fillRect(x0 * displayScale, y0 * displayScale - 24, 50, 20);
          ctx.fillStyle = "#000";
          ctx.font = "bold 12px JetBrains Mono, monospace";
          ctx.fillText(
            `${(score * 100).toFixed(0)}%`,
            x0 * displayScale + 4,
            y0 * displayScale - 8
          );
        }
      }
    }

    // Draw prompted boxes
    if (result?.prompted_boxes) {
      for (const promptedBox of result.prompted_boxes) {
        const [x0, y0, x1, y1] = promptedBox.box;
        ctx.strokeStyle = promptedBox.label ? "#3beba1" : "#f87171";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(
          x0 * displayScale,
          y0 * displayScale,
          (x1 - x0) * displayScale,
          (y1 - y0) * displayScale
        );
        ctx.setLineDash([]);
      }
    }

    // Draw current drawing box
    if (isDrawing && startPoint && currentPoint) {
      const x = Math.min(startPoint.x, currentPoint.x);
      const y = Math.min(startPoint.y, currentPoint.y);
      const width = Math.abs(currentPoint.x - startPoint.x);
      const height = Math.abs(currentPoint.y - startPoint.y);

      ctx.strokeStyle = boxMode === "positive" ? "#3beba1" : "#f87171";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, width, height);
      ctx.setLineDash([]);
    }
  }, [
    imageWidth,
    imageHeight,
    displayScale,
    result,
    isDrawing,
    startPoint,
    currentPoint,
    boxMode,
  ]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isLoading) return;
    const coords = getCanvasCoordinates(e);
    if (coords) {
      setIsDrawing(true);
      setStartPoint(coords);
      setCurrentPoint(coords);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const coords = getCanvasCoordinates(e);
    if (coords) {
      setCurrentPoint(coords);
    }
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPoint) {
      setIsDrawing(false);
      return;
    }

    const coords = getCanvasCoordinates(e);
    if (!coords) {
      setIsDrawing(false);
      return;
    }

    // Calculate box in original image coordinates
    const x0 = Math.min(startPoint.x, coords.x) / displayScale;
    const y0 = Math.min(startPoint.y, coords.y) / displayScale;
    const x1 = Math.max(startPoint.x, coords.x) / displayScale;
    const y1 = Math.max(startPoint.y, coords.y) / displayScale;

    // Minimum box size check
    if (Math.abs(x1 - x0) < 10 || Math.abs(y1 - y0) < 10) {
      setIsDrawing(false);
      setStartPoint(null);
      setCurrentPoint(null);
      return;
    }

    // Convert to normalized center x, center y, width, height format
    const centerX = (x0 + x1) / 2 / imageWidth;
    const centerY = (y0 + y1) / 2 / imageHeight;
    const width = (x1 - x0) / imageWidth;
    const height = (y1 - y0) / imageHeight;

    onBoxDrawn([centerX, centerY, width, height]);

    setIsDrawing(false);
    setStartPoint(null);
    setCurrentPoint(null);
  };

  const handleMouseLeave = () => {
    if (isDrawing) {
      setIsDrawing(false);
      setStartPoint(null);
      setCurrentPoint(null);
    }
  };

  if (!imageUrl) {
    return (
      <div
        ref={containerRef}
        className="flex items-center justify-center h-96 border-2 border-dashed border-border rounded-xl bg-card/50"
      >
        <p className="text-muted-foreground text-sm">
          Upload an image to begin segmentation
        </p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative">
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        className={`rounded-lg shadow-xl ${
          isLoading ? "opacity-50 pointer-events-none" : ""
        }`}
        style={{ cursor: isLoading ? "wait" : "crosshair" }}
      />
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex items-center gap-3 bg-card/90 backdrop-blur-sm px-4 py-2 rounded-lg border border-border">
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            <span className="text-sm">Processing...</span>
          </div>
        </div>
      )}
    </div>
  );
}
