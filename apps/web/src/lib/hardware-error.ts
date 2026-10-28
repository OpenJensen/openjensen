export type DescribedError = { title: string; next?: string; details?: string };

const NO_RESPONSE =
  /no status packet|\[TxRxResult\]|Input\/output error|\[Errno 5\]|port is closed|device disconnected/i;

/**
 * Turn a hardware companion error into one readable sentence and a next step.
 * The original text stays available as `details` for diagnosis.
 */
export function describeHardwareError(message: string): DescribedError {
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
