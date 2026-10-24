import type { ReactNode } from 'react';

export type IconName =
  | 'search'
  | 'arrow'
  | 'plus'
  | 'database'
  | 'folder'
  | 'check'
  | 'clock'
  | 'spark'
  | 'external'
  | 'sliders'
  | 'layers'
  | 'compress'
  | 'chart'
  | 'play'
  | 'book'
  | 'help'
  | 'robot'
  | 'home'
  | 'record'
  | 'funnel'
  | 'branch'
  | 'box'
  | 'cloud'
  | 'gear';

export function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    home: (
      <>
        <path d="m3 11 9-7 9 7" />
        <path d="M5 10v10h5v-6h4v6h5V10" />
      </>
    ),
    record: (
      <>
        <circle cx="12" cy="12" r="9" />
        <circle cx="12" cy="12" r="3.5" fill="currentColor" stroke="none" />
      </>
    ),
    funnel: <path d="M3 4h18l-7 8.5V19l-4 2v-8.5L3 4Z" />,
    branch: (
      <>
        <circle cx="6" cy="5" r="2" />
        <circle cx="6" cy="19" r="2" />
        <circle cx="18" cy="7" r="2" />
        <path d="M6 7v10M18 9c0 5-6 4-11.2 8.6" />
      </>
    ),
    box: (
      <>
        <path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z" />
        <path d="m4 7.5 8 4.5 8-4.5M12 12v9" />
      </>
    ),
    cloud: <path d="M7 18a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 18 9a4.5 4.5 0 0 1-.5 9H7Z" />,
    gear: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M12 2.5v2.6M12 18.9v2.6M4.4 4.4l1.9 1.9M17.7 17.7l1.9 1.9M2.5 12h2.6M18.9 12h2.6M4.4 19.6l1.9-1.9M17.7 6.3l1.9-1.9" />
        <circle cx="12" cy="12" r="6.5" />
      </>
    ),
    robot: (
      <>
        <path d="M4 21h16M7 21v-3l4-3M11 15l-4-5 4-6 5 3-3 6 4 3v5" />
        <circle cx="11" cy="15" r="2" />
        <circle cx="7" cy="10" r="2" />
        <path d="m16 7 3-2 2 3-3 2" />
      </>
    ),
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="m16 16 5 5" />
      </>
    ),
    arrow: (
      <>
        <path d="M5 12h14M13 6l6 6-6 6" />
      </>
    ),
    plus: (
      <>
        <path d="M12 5v14M5 12h14" />
      </>
    ),
    database: (
      <>
        <ellipse cx="12" cy="5" rx="8" ry="3" />
        <path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />
      </>
    ),
    folder: (
      <>
        <path d="M3 7V5a1 1 0 0 1 1-1h5l2 3h9a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7Z" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    help: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M9.5 9a2.5 2.5 0 1 1 4.2 1.8c-1.1.6-1.7 1.1-1.7 2.2" />
        <circle cx="12" cy="16.5" r=".7" fill="currentColor" stroke="none" />
      </>
    ),
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    spark: (
      <>
        <path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z" />
      </>
    ),
    external: (
      <>
        <path d="M14 3h7v7M21 3l-9 9M10 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5" />
      </>
    ),
    sliders: (
      <>
        <path d="M4 7h7m4 0h5M4 17h3m4 0h9" />
        <circle cx="13" cy="7" r="2" />
        <circle cx="9" cy="17" r="2" />
      </>
    ),
    layers: (
      <>
        <path d="m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5" />
      </>
    ),
    compress: (
      <>
        <path d="M4 9h5V4M20 9h-5V4M4 15h5v5M20 15h-5v5M9 9 3 3M15 9l6-6M9 15l-6 6M15 15l6 6" />
      </>
    ),
    chart: (
      <>
        <path d="M4 3v17h17M8 15v-4M13 15V7M18 15V5" />
      </>
    ),
    play: <path d="m8 4 12 8-12 8V4Z" />,
    book: (
      <>
        <path d="M12 5v16M3 3h5a4 4 0 0 1 4 2 4 4 0 0 1 4-2h5v16h-5a4 4 0 0 0-4 2 4 4 0 0 0-4-2H3V3Z" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
