import { Children, cloneElement, createElement, Fragment, isValidElement } from "react";

/** Marker on wrapper components whose single child should inherit clip classification. */
export const CHART_CLIP_PASSTHROUGH = "__chartClipPassthrough";

export function isChartClipPassthrough(type) {
  return (typeof type === "function" && (type)[CHART_CLIP_PASSTHROUGH] ===
    true);
}

const REACT_LAZY_TYPE = Symbol.for("react.lazy");

function isLazyComponentType(type) {
  return (typeof type === "object" &&
  type !== null &&
  (type).$$typeof === REACT_LAZY_TYPE && typeof (type)._init === "function");
}

/**
 * Unwrap a `React.lazy` element type to the component it resolves to.
 *
 * Chart children written in a Server Component (e.g. MDX docs) reach the
 * client chart as lazy client references, which hide `displayName` and the
 * `__isChartMarkers` / `__isPostOverlay` flags. Without unwrapping, markers,
 * axes and grids are classified as series and end up inside the reveal clip,
 * which cuts off marker count badges above the plot.
 */
export function resolveLazyChartChild(child) {
  const type = child.type;
  if (!isLazyComponentType(type)) {
    return child;
  }
  let resolved;
  try {
    resolved = type._init(type._payload);
  } catch {
    // Module still loading (lazy init throws a thenable) — keep the lazy element.
    return child;
  }
  if (
    typeof resolved !== "function" &&
    !(typeof resolved === "object" && resolved !== null)
  ) {
    return child;
  }
  return createElement(resolved, {
    ...(child.props),
    key: child.key,
  });
}

/** Unwrap visibility wrappers so `Grid` / axes stay outside the series clip. */
export function resolveChartChildElement(child) {
  const element = resolveLazyChartChild(child);
  if (isChartClipPassthrough(element.type)) {
    const inner = (element.props).children;
    if (isValidElement(inner)) {
      return resolveChartChildElement(inner);
    }
  }
  return element;
}

/** Walk chart children, flattening React fragments (studio often groups layers in `<>...</>`). */
export function forEachChartChild(
  children,
  callback
) {
  let index = 0;
  const visit = (nodes) => {
    Children.forEach(nodes, (child) => {
      if (!isValidElement(child)) {
        return;
      }
      if (child.type === Fragment) {
        visit((child.props).children);
        return;
      }
      callback(resolveLazyChartChild(child), index);
      index += 1;
    });
  };
  visit(children);
}

const CLIP_EXCLUDED_COMPONENT_NAMES = new Set([
  "Background",
  "Grid",
  "XAxis",
  "YAxis",
  "BarXAxis",
  "BarYAxis",
  "LiveXAxis",
  "LiveYAxis",
]);

const UNDERLAY_COMPONENT_NAMES = new Set([
  "ReferenceArea",
  "BarColumnTrack",
  "Weekends",
  "Deadline",
]);

/** Markers render after the interaction overlay so they stay clickable. */
export function isPostOverlayComponent(child) {
  const childType = child.type;

  if (childType.__isChartMarkers || childType.__isPostOverlay) {
    return true;
  }

  const componentName =
    typeof child.type === "function"
      ? childType.displayName || childType.name || ""
      : "";

  return (
    componentName === "ChartMarkers" ||
    componentName === "MarkerGroup" ||
    componentName === "ChartBrush"
  );
}

/** Renders above grid/axes but below series; excluded from grow-clip reveal. */
export function isUnderlayComponent(child) {
  const childType = child.type;
  const componentName =
    typeof child.type === "function"
      ? childType.displayName || childType.name || ""
      : "";
  return UNDERLAY_COMPONENT_NAMES.has(componentName);
}

/** Grid and axes stay visible during series clip reveal (e.g. loading → ready). */
export function isClipExcludedComponent(child) {
  const childType = child.type;
  const componentName =
    typeof child.type === "function"
      ? childType.displayName || childType.name || ""
      : "";
  return CLIP_EXCLUDED_COMPONENT_NAMES.has(componentName);
}

/** SVG layer lists from chart shells need stable keys when rendered as arrays. */
export function renderKeyedChartLayers(children) {
  return children.map((child, index) =>
    cloneElement(child, { key: child.key ?? `chart-layer-${index}` }));
}
