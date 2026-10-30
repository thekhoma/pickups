import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const outDir = fileURLToPath(new URL("../src/pickups/web/dist", import.meta.url));
const BACKEND = "http://127.0.0.1:7788";

// Mock mode ships in every build (a lazy chunk plus the small synthetic clips in public/mock): `?mock=1`,
// `?mock=fresh` and `?tour=1` work from the backend's own page too.
export default defineConfig(({ mode }) => {
  return {
    base: "/",
    plugins: [react()],
    define: mode === "mock" ? { "import.meta.env.VITE_MOCK": JSON.stringify("1") } : {},
    build: { outDir, emptyOutDir: true, chunkSizeWarningLimit: 800 },
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": { target: BACKEND, changeOrigin: false },
        "/media": { target: BACKEND, changeOrigin: false },
        "/examples": { target: BACKEND, changeOrigin: false },
      },
    },
  };
});
