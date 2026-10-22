"use client";;
import { useParentSize } from "@visx/responsive";

/**
 * Drop-in for visx `<ParentSize>` without its `overflow: hidden` measuring
 * layer, so tooltips, markers, and strokes at the chart edge aren't clipped.
 */
export function ParentSize({
  children,
  className,
  debounceTime,
  style
}) {
  const { parentRef, width, height } = useParentSize({ debounceTime });
  return (
    <div
      className={className}
      style={{ position: "relative", width: "100%", height: "100%", ...style }}>
      <div ref={parentRef} style={{ position: "absolute", inset: 0 }}>
        {children({ width, height })}
      </div>
    </div>
  );
}
