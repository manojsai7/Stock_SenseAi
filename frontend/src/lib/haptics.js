/**
 * Trigger gentle tactile feedback for explicit user interactions on supported touch devices.
 * Graceful no-op when the Vibration API is unsupported or disabled.
 */
export function triggerHaptic(pattern = 'selection') {
  if (typeof navigator === 'undefined' || typeof navigator.vibrate !== 'function') {
    return;
  }
  const duration = pattern === 'success' ? [8, 24, 8] : pattern === 'error' ? [18, 28, 18] : 8;
  try {
    navigator.vibrate(duration);
  } catch {
    /* graceful no-op */
  }
}
