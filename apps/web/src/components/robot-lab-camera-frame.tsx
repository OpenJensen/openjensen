'use client';

import { useEffect, useRef, useState } from 'react';
import { robotApi } from '@/lib/robot-api';

export function RobotLabCameraFrame({
  index,
  indices,
  stamp,
  large,
  alt,
  onError,
}: {
  index: number;
  indices: number[];
  stamp: number;
  large: boolean;
  alt: string;
  onError: () => void;
}) {
  const [sequence, setSequence] = useState(0);
  const [frameState, setFrameState] = useState<'loading' | 'live' | 'delayed'>('loading');
  const lastFrame = useRef(0);
  const nextFrame = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestedAt = useRef(0);
  const interval = useRef(large ? 1000 / 30 : 1000 / 15);
  useEffect(() => {
    interval.current = large ? 1000 / 30 : 1000 / 15;
  }, [large]);
  useEffect(() => {
    const started = Date.now();
    const timer = setInterval(() => {
      if (Date.now() - (lastFrame.current || started) > 2500) setFrameState('delayed');
    }, 500);
    return () => {
      clearInterval(timer);
      if (nextFrame.current) clearTimeout(nextFrame.current);
    };
  }, []);
  return (
    <>
      <img
        alt={alt}
        src={robotApi.cameraUrl(index, stamp + sequence, indices)}
        onLoad={() => {
          lastFrame.current = Date.now();
          setFrameState('live');
          if (nextFrame.current) clearTimeout(nextFrame.current);
          // One request at a time, paced from its start rather than adding a
          // fixed pause after every response. Slow responses never build a queue.
          const wait = Math.max(0, interval.current - (performance.now() - requestedAt.current));
          nextFrame.current = setTimeout(() => {
            requestedAt.current = performance.now();
            setSequence((value) => value + 1);
          }, wait);
        }}
        onError={() => {
          if (nextFrame.current) clearTimeout(nextFrame.current);
          onError();
        }}
      />
      <span className={frameState === 'live' ? 'robot-lab-camera-live' : 'robot-lab-camera-delayed'} role="status">
        {frameState === 'live'
          ? 'LIVE'
          : frameState === 'delayed'
            ? 'Feed delayed — waiting for a new image'
            : 'Loading camera…'}
      </span>
    </>
  );
}
