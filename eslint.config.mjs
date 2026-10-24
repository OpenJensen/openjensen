import js from '@eslint/js';
import reactHooks from 'eslint-plugin-react-hooks';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: [
      'apps/web/src/lib/api.generated.ts',
      // SO-101 Robot Lab sources are under active development on another branch;
      // lint them once that work merges, as with .prettierignore.
      'apps/web/src/components/robot-lab.*',
      'apps/web/src/components/so101-*',
      'apps/web/src/lib/robot-api.ts',
      'apps/web/src/lib/so101-*',
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
