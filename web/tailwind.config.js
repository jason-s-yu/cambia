/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}' // Scan these files for Tailwind classes
  ],
  darkMode: 'class', // Use the 'class' strategy for manual dark mode toggling
  theme: {
    extend: {
      // Add custom theme extensions here if needed
      // Example:
      // colors: {
      //   primary: '#ff0000',
      // },
    }
  },
  plugins: [
    // Add official Tailwind plugins here if needed (install them first)
    // Example: require('@tailwindcss/forms'),
    // require('@tailwindcss/typography'),
  ]
};