import js from '@eslint/js';
import reactHooks from 'eslint-plugin-react-hooks';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: [
      'apps/web/src/lib/api.generated.ts',
    ],
  },
  {
    files: ['apps/web/src/**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended, reactHooks.configs.flat.recommended],
    rules: {
      // Input validation deliberately rejects control characters with [\x00-\x1f].
      'no-control-regex': 'off',
      // React Compiler rules. The app does not use the compiler, and several panels
      // intentionally reset state in effects and read refs to fence stale async work,
      // so these are reported rather than enforced; `refs` flags every such read.
      'react-hooks/refs': 'off',
      'react-hooks/set-state-in-effect': 'warn',
      'react-hooks/purity': 'warn',
      'react-hooks/immutability': 'warn',
    },
  },
);
