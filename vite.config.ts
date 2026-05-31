import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? "/pusoy-alpha/" : "/",
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 8080,
    allowedHosts: ["vibecheck.local"]
  },
  preview: {
    host: "0.0.0.0",
    port: 8080
  }
});
