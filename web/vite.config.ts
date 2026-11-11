import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/",
  build: {
    target: "es2022",
    assetsInlineLimit: 2048,
  },
  server: {
    host: true,
    port: 5173,
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
