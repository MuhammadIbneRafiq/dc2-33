/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#1d4ed8",
        secondary: "#10b981",
        danger: "#ef4444",
        warning: "#f59e0b",
        dark: "#1f2937",
      },
    },
  },
  plugins: [],
} 