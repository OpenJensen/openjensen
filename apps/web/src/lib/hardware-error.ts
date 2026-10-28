export type DescribedError = { title: string; next?: string; details?: string };

const NO_RESPONSE =
  /no status packet|\[TxRxResult\]|Input\/output error|\[Errno 5\]|port is closed|device disconnected/i;

/**
 * Turn a hardware companion error into one readable sentence and a next step.
 * The original text stays available as `details` for diagnosis.
 */
export function describeHardwareError(message: string): DescribedError {
  const motor = message.match(/Motor (\d+)/i)?.[1];
  const motorNames = ['shoulder pan', 'shoulder lift', 'elbow', 'wrist flex', 'wrist roll', 'gripper'];
  if (/OverEle error|overcurrent/i.test(message)) {
    return {
      title: motor
        ? `Motor ${motor} (${motorNames[Number(motor) - 1] ?? 'servo'}) reports overcurrent.`
        : 'A motor reports overcurrent.',
      next: 'Support the arm and switch off motor power. Check for binding, obstructions and tight cables before powering on and scanning again.',
      details: message,
    };
  }
  const range = message.match(/([A-Za-z ]+) is outside its saved range \((\d+) ticks; allowed (\d+)[–-](\d+)\)/);
  if (range) {
    const joint = range[1].trim(),
      value = Number(range[2]),
      low = Number(range[3]),
      high = Number(range[4]);
    const distance = value < low ? low - value : value - high;
    return {
      title: `${joint} is outside its calibrated range.`,
      next: `The reading is ${distance} tick${distance === 1 ? '' : 's'} ${value < low ? 'below' : 'above'} the saved range (${low}–${high}). Support the arm and reposition inside the range with motor power off, or recalibrate its full travel.`,
      details: message,
    };
  }
  if (/corrupt motor repl(?:y|ies)|COMM_RX_CORRUPT/i.test(message)) {
    return {
      title: 'A motor reply was corrupted.',
      next: 'Check motor power and USB/servo cables, then scan again. If this repeats, check for duplicate motor IDs.',
      details: message,
    };
  }
  if (NO_RESPONSE.test(message)) {
    return {
      title: 'The motors stopped responding.',
      next: /cut motor power/i.test(message)
        ? 'If the arm is still holding position, switch off motor power. Then check the USB cable and power supply.'
        : 'Check the USB cable and motor power, then refresh devices.',
      details: message,
    };
  }
  if (message.length <= 200) return { title: message };
  const first = message.match(/^.*?[.!?](?=\s|$)/)?.[0] ?? message.slice(0, 160);
  return { title: first, details: message };
}
