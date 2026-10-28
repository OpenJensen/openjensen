import { expect, test } from '@playwright/test';
import { describeHardwareError } from '../../apps/web/src/lib/hardware-error';

const stalled =
  'Hardware session stopped: Motor 1 read Present_Position failed: [TxRxResult] There is no status packet!. Stop attempted but motor acknowledgement failed. Cut motor power if needed. Motor 2 write Torque_Enable acknowledgement unverified';

test('unresponsive motors become one sentence, the next step and the raw driver output', () => {
  expect(describeHardwareError(stalled)).toEqual({
    title: 'The motors stopped responding.',
    next: 'If the arm is still holding position, switch off motor power. Then check the USB cable and power supply.',
    details: stalled,
  });
  expect(describeHardwareError('[Errno 5] Input/output error').next).toBe(
    'Check the USB cable and motor power, then refresh devices.',
  );
});

test('short messages stay as written and long ones keep their first sentence', () => {
  const short = 'Stop attempted but motor acknowledgement failed. Keep supporting the arm.';
  expect(describeHardwareError(short)).toEqual({ title: short });
  const long = `Calibration could not be saved. ${'The profile write was refused. '.repeat(8)}`;
  expect(describeHardwareError(long)).toEqual({ title: 'Calibration could not be saved.', details: long });
});

test('corrupt replies explain serial communication separately from the device name', () => {
  const message =
    'Motor safety check failed: Corrupt motor replies: disconnect the other motors and check cabling/power';
  const described = describeHardwareError(message);
  expect(described.title).toBe('A motor reply was corrupted.');
  expect(described.next).toContain('check for duplicate motor IDs');
  expect(described.next).not.toContain('disconnect the other motors');
  expect(described.details).toBe(message);
});
