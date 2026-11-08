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
  const delay = useRef(large ? 100 : 700);
  useEffect(() => {
    delay.current = large ? 100 : 700;
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
          nextFrame.current = setTimeout(() => setSequence((value) => value + 1), delay.current);
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
