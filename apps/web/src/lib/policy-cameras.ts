/** A policy trained on several camera views needs one camera of this computer for each. */

/** "observation.images.camera_4" -> "camera_4": the name the recording gave the view. */
export function viewName(key: string) {
  return key.replace(/^observation\.images?\.?/, '') || key;
}

/**
 * Which camera feeds each view. The first view follows the camera selected on the page.
 * Every other view keeps the user's choice while that camera is free; otherwise it takes
 * the camera number in its name (a recording calls its second view after its camera, so on
 * the recording computer this is right) or the next unused camera. Returns null when this
 * computer has fewer cameras than the policy has views.
 */
export function policyCameras(
  keys: string[],
  primary: string,
  selected: number,
  available: number[],
  chosen: Record<string, number> = {},
): Record<string, number> | null {
  if (!keys.includes(primary) || !available.includes(selected)) return null;
  const result: Record<string, number> = { [primary]: selected };
  const used = new Set([selected]);
  for (const key of keys) {
    if (key === primary) continue;
    const named = /camera_(\d+)$/.exec(key);
    const index = [chosen[key], named ? Number(named[1]) : undefined, ...available].find(
      (value): value is number => value !== undefined && available.includes(value) && !used.has(value),
    );
    if (index === undefined) return null;
    used.add(index);
    result[key] = index;
  }
  return Object.fromEntries(keys.map((key) => [key, result[key]]));
}

export function sameCameras(left?: Record<string, number> | null, right?: Record<string, number> | null) {
  const a = Object.entries(left ?? {}),
    b = right ?? {};
  return a.length === Object.keys(b).length && a.every(([key, value]) => b[key] === value);
}
