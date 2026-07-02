import vue from "@vitejs/plugin-vue";
import { resolve } from "path";
import { defineConfig, ConfigEnv } from "vite";
import viteCompression from "vite-plugin-compression";
import { buildConfig } from "./src/utils/build";
import EnvironmentPlugin from "vite-plugin-environment";

const pathResolve = (dir: string) => {
  return resolve(__dirname, ".", dir);
};

const alias: Record<string, string> = {
  "@": pathResolve("./src/"),
  "/@": pathResolve("./src/"),
  "/images": pathResolve("./src/assets/images"),
  "vue-i18n": "vue-i18n/dist/vue-i18n.cjs.js",
};

const envValue = (key: string, defaultValue: string) => process.env[key] ?? defaultValue;

const envBoolean = (key: string, defaultValue = false) => {
  const value = process.env[key];
  if (value === undefined) return defaultValue;
  try {
    return JSON.parse(value) === true;
  } catch {
    return defaultValue;
  }
};

const viteConfig = defineConfig((mode: ConfigEnv) => {
  const openCdn = envBoolean("VITE_OPEN_CDN", false);
  const openBrowser = envBoolean("VITE_OPEN", false);
  const publicPath = envValue("VITE_PUBLIC_PATH", "./");
  const backendAddress = envValue("VITE_BACKEND_ADDRESS", "http://127.0.0.1:8000");
  const port = Number(envValue("VITE_PORT", "8888"));

  return {
    plugins: [
      vue(),
      viteCompression(),
      openCdn ? buildConfig.cdn() : null,
      EnvironmentPlugin({
        VITE_PORT: String(port),
        VITE_OPEN: String(openBrowser),
        VITE_OPEN_CDN: String(openCdn),
        VITE_PUBLIC_PATH: publicPath,
        VITE_BACKEND_ADDRESS: backendAddress,
      }),
    ],
    root: process.cwd(),
    resolve: { alias },
    base: mode.command === "serve" ? "./" : publicPath,
    optimizeDeps: { exclude: ["vue-demi"] },
    server: {
      host: "0.0.0.0",
      port,
      open: openBrowser,
      hmr: true,
      proxy: {
        "/gitee": {
          target: "https://gitee.com",
          ws: true,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/gitee/, ""),
        },
        "/api": {
          target: backendAddress,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
    build: {
      outDir: "dist",
      chunkSizeWarningLimit: 1500,
      rollupOptions: {
        output: {
          chunkFileNames: "assets/js/[name]-[hash].js",
          entryFileNames: "assets/js/[name]-[hash].js",
          assetFileNames: "assets/[ext]/[name]-[hash].[ext]",
          manualChunks(id) {
            if (id.includes("node_modules")) {
              return (
                id
                  .toString()
                  .match(/\/node_modules\/(?!.pnpm)(?<moduleName>[^\/]*)\//)
                  ?.groups!.moduleName ?? "vender"
              );
            }
          },
        },
        ...(openCdn ? { external: buildConfig.external } : {}),
      },
    },
    css: { preprocessorOptions: { css: { charset: false } } },
    define: {
      __VUE_I18N_LEGACY_API__: JSON.stringify(false),
      __VUE_I18N_FULL_INSTALL__: JSON.stringify(false),
      __INTLIFY_PROD_DEVTOOLS__: JSON.stringify(false),
      __NEXT_VERSION__: JSON.stringify(process.env.npm_package_version),
      __NEXT_NAME__: JSON.stringify(process.env.npm_package_name),
    },
  };
});

export default viteConfig;
