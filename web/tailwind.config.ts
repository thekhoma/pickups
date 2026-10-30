import type { Config } from "tailwindcss";

// "The cutting room" (design v2.1): warm black, ivory type, the footage is the colour.
// Tally red is for dots, the record line and Delete only. No gold, orange, amber, lime or purple anywhere.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0D0C0B",
        panel: "#151412",
        raised: "#1D1B18",
        line: "#2A2724",
        line2: "#3A3631",
        ivory: "rgb(238 233 224 / <alpha-value>)",
        /** Body text: ivory at 85% (100% is reserved for headlines and the primary pill). */
        ink: "rgb(238 233 224 / 0.85)",
        ink2: "#A39D93",
        ink3: "#857F77",
        /** Disabled only. */
        ink4: "#6E6962",
        tally: "#F5301E",
        unlit: "#3A2A28",
        borrowed: "#3DD6F5",
        ai: "#FF4FD8",
      },
      fontFamily: {
        serif: ['"Instrument Serif"', "Georgia", "serif"],
        sans: ['"Instrument Sans"', "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ['"Geist Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      borderRadius: {
        card: "6px",
      },
      transitionTimingFunction: {
        out: "cubic-bezier(0.2, 0.7, 0.2, 1)",
      },
      keyframes: {
        shimmer: { "0%": { backgroundPosition: "-400px 0" }, "100%": { backgroundPosition: "400px 0" } },
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
        "rise-in": { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "none" } },
        "slide-in": { from: { transform: "translateX(-24px)", opacity: "0" }, to: { transform: "none", opacity: "1" } },
        draw: { from: { strokeDashoffset: "24" }, to: { strokeDashoffset: "0" } },
        probe: { "0%, 100%": { opacity: "1" }, "50%": { opacity: "0.25" } },
      },
      animation: {
        shimmer: "shimmer 1.6s linear infinite",
        "fade-in": "fade-in 160ms ease-out both",
        "rise-in": "rise-in 160ms cubic-bezier(0.2, 0.7, 0.2, 1) both",
        "slide-in": "slide-in 180ms cubic-bezier(0.2, 0.7, 0.2, 1) both",
        draw: "draw 200ms cubic-bezier(0.2, 0.7, 0.2, 1) both",
        probe: "probe 1.2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
