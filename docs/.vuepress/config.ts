import { defineUserConfig } from "vuepress";
import type { DefaultThemeOptions } from "vuepress";

export default defineUserConfig<DefaultThemeOptions>({
  lang: "en-US",
  title: "CraNet",
  description: "A tiny library for Deep Learning",
  base: "/cranet/",
  themeConfig: {},
  bundler: "@vuepress/bundler-vite",
  markdown: {
    anchor: {},
    extractHeaders: {
      level: [2, 3],
    },
    importCode: {
      handleImportPath: (str) => str,
    },
    links: {
      internalTag: "RouterLink",
      externalAttrs: { target: "_blank", rel: "noopener noreferrer" },
      externalIcon: true,
    },
    toc: {
      pattern: /^\[\[toc\]\]$/i,
      level: [2, 3],
    },
  },
});
