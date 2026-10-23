import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// getUserMedia only works on https or localhost. To test on a phone, serve the dev server over
// the LAN through a tunnel or a local certificate, and point the proxy at the FastAPI app.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
