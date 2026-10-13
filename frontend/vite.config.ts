import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development, /api is proxied to the FastAPI backend so the browser talks to a
// single origin. Override the target with VITE_API_PROXY_TARGET.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
