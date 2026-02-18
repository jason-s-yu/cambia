import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactPlugin from 'eslint-plugin-react';
import hooksPlugin from 'eslint-plugin-react-hooks';
import refreshPlugin from 'eslint-plugin-react-refresh';
import stylisticTs from '@stylistic/eslint-plugin-ts';
import stylisticPlugin from '@stylistic/eslint-plugin';
import globals from 'globals';
import prettierConfig from 'eslint-config-prettier'; // Needs to be last in extends/plugins that override styles

export default tseslint.config(
  // Global ignores
  {
    ignores: ['dist', 'node_modules', '.env', '*.config.js', '*.config.ts']
  },

  // Base ESLint recommended config
  eslint.configs.recommended,

  // TypeScript specific configs
  ...tseslint.configs.recommended, // Or recommendedTypeChecked if using type-aware rules extensively
  // ...tseslint.configs.stylistic, // Consider adding stylistic rules from typescript-eslint if desired

  // React specific configs
  {
    files: ['**/*.{ts,tsx}'], // Target TS and TSX files for React rules
    plugins: {
      react: reactPlugin,
      'react-hooks': hooksPlugin,
      'react-refresh': refreshPlugin
    },
    languageOptions: {
      parserOptions: {
        ecmaFeatures: {
          jsx: true
        }
      },
      globals: {
        ...globals.browser // Define browser environment globals
      }
    },
    settings: {
      react: {
        version: 'detect' // Automatically detect React version
      }
    },
    rules: {
      ...reactPlugin.configs.recommended.rules, // Recommended React rules
      ...reactPlugin.configs['jsx-runtime'].rules, // Rules for new JSX transform
      ...hooksPlugin.configs.recommended.rules, // Recommended React Hooks rules
      'react-refresh/only-export-components': [ // Rule for Fast Refresh
        'warn',
        { allowConstantExport: true }
      ],
      'react/prop-types': 'off', // Disable prop-types as we use TypeScript
      'react/react-in-jsx-scope': 'off' // Not needed with new JSX transform
      // Add any other React specific rule overrides here
    }
  },

  // Stylistic rules using @stylistic plugin
  {
    files: ['**/*.{js,ts,jsx,tsx}'],
    plugins: {
      '@stylistic': stylisticPlugin,
      '@stylistic/ts': stylisticTs // Use the specific TS plugin for TS stylistic rules
    },
    rules: {
      // Enforce specific stylistic rules from the spec
      '@stylistic/ts/quotes': ['error', 'single', { avoidEscape: true, allowTemplateLiterals: true }],
      '@stylistic/ts/semi': ['error', 'always'],
      '@stylistic/ts/comma-dangle': ['error', 'never'],

      // Add other stylistic rules as desired (examples)
      '@stylistic/ts/indent': ['error', 2], // Example: 2 spaces indentation
      '@stylistic/ts/object-curly-spacing': ['error', 'always'], // Example: space inside braces
      '@stylistic/ts/member-delimiter-style': ['error', {
        multiline: { delimiter: 'semi', requireLast: true },
        singleline: { delimiter: 'semi', requireLast: false }
      }],
      '@stylistic/ts/type-annotation-spacing': 'error'

      // You can also use pre-defined stylistic configs, e.g.:
      // ...stylisticPlugin.configs['recommended-flat'].rules, // Or recommended-extends
    }
  },

  // Prettier config to disable conflicting ESLint style rules
  // IMPORTANT: Must be the LAST configuration object in the array
  prettierConfig
);