import vue from "@vitejs/plugin-vue";
import { resolve } from "path";
import { defineConfig, ConfigEnv } from "vite";
// import vueSetupExtend from "vite-plugin-vue-setup-extend-plus";
import viteCompression from "vite-plugin-compression";
import { buildConfig } from "./src/utils/build";
import EnvironmentPlugin from "vite-plugin-environment";

const pathResolve = (dir: string) => {
  return resolve(__dirname, ".", dir);
};

const alias: Record<string, string> = {
  "/@": pathResolve("./src/"),
  '/images': pathResolve("./src/assets/images"),
  "vue-i18n": "vue-i18n/dist/vue-i18n.cjs.js",
};

// 之前使用loadEnv获取环境变量 现在使用node自带的process对象配置环境变量
const viteConfig = defineConfig((mode: ConfigEnv) => {

  return {
    plugins: [
      vue(),
      // vueSetupExtend(),
      viteCompression(),
      JSON.parse(String(process.env.VITE_OPEN_CDN)) ? buildConfig.cdn() : null,
      EnvironmentPlugin([
        "VITE_PORT",
        "VITE_OPEN",
        "VITE_OPEN_CDN",
        "VITE_PUBLIC_PATH",
        "VITE_BACKEND_ADDRESS",
      ]),
    ],
    root: process.cwd(),
    resolve: { alias },
    base: mode.command === "serve" ? "./" : process.env.VITE_PUBLIC_PATH,
    optimizeDeps: { exclude: ["vue-demi"] },
    server: {
      host: "0.0.0.0",
      // port: env.VITE_PORT as unknown as number,
      port: process.env.VITE_PORT as unknown as number,
      open: JSON.parse(String(process.env.VITE_OPEN)),
      hmr: true,
      proxy: {
        "/gitee": {
          target: "https://gitee.com",
          ws: true,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/gitee/, ""),
        },
        "/api": {
          target: process.env.VITE_BACKEND_ADDRESS,
          // target: "http://127.0.0.1:5000",
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
        ...(JSON.parse(String(process.env.VITE_OPEN_CDN))
          ? { external: buildConfig.external }
          : {}),
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
