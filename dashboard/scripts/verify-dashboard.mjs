import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { chromium } from "playwright-core";
import { PNG } from "pngjs";

const browserPath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const outputDir = path.resolve(import.meta.dirname, "..", "..", "outputs", "dashboard");
fs.mkdirSync(outputDir, { recursive: true });

const server = spawn(process.execPath, ["scripts/serve.mjs"], {
  cwd: path.resolve(import.meta.dirname, ".."),
  stdio: "ignore",
});
for (let attempt = 0; attempt < 40; attempt += 1) {
  try {
    const response = await fetch("http://127.0.0.1:4173");
    if (response.ok) break;
  } catch {
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
}

const browser = await chromium.launch({
  executablePath: browserPath,
  headless: true,
  args: ["--use-angle=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"],
});

const errors = [];
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  deviceScaleFactor: 1,
});
const page = await context.newPage();
page.on("console", (message) => {
  if (message.type() === "error") errors.push(`console: ${message.text()}`);
});
page.on("pageerror", (error) => errors.push(`page: ${error.message}`));

await page.goto("http://127.0.0.1:4173", { waitUntil: "networkidle" });
await page.getByText("Metric model ready").waitFor({ timeout: 60000 });
await page.waitForTimeout(1500);

const canvasBuffer = await page.locator("canvas").screenshot();
const png = PNG.sync.read(canvasBuffer);
let min = 255;
let max = 0;
let nonBackground = 0;
const corner = [png.data[0], png.data[1], png.data[2]];
for (let index = 0; index < png.data.length; index += 4) {
  const r = png.data[index];
  const g = png.data[index + 1];
  const b = png.data[index + 2];
  const value = Math.round((r + g + b) / 3);
  min = Math.min(min, value);
  max = Math.max(max, value);
  if (
    Math.abs(r - corner[0]) +
      Math.abs(g - corner[1]) +
      Math.abs(b - corner[2]) >
    24
  ) {
    nonBackground += 1;
  }
}
const canvasRatio = nonBackground / (png.width * png.height);
if (max - min < 20 || canvasRatio < 0.01) {
  errors.push(`canvas appears blank: range=${max - min}, foreground=${canvasRatio}`);
}

await page.screenshot({
  path: path.join(outputDir, "dashboard-desktop.png"),
  fullPage: true,
});

await page.getByRole("button", { name: "Evaluation" }).click();
await page.getByText("Capture quality changed the map").waitFor();
await page.screenshot({
  path: path.join(outputDir, "dashboard-evaluation.png"),
  fullPage: true,
});

await page.getByRole("button", { name: "System" }).click();
await page.getByText("System execution trace").waitFor();
await page.screenshot({
  path: path.join(outputDir, "dashboard-system.png"),
  fullPage: true,
});

await page.setViewportSize({ width: 390, height: 844 });
await page.getByRole("button", { name: "Reconstruction" }).click();
await page.getByText("Metric model ready").waitFor();
await page.waitForTimeout(500);
const overflow = await page.evaluate(
  () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
);
if (overflow) {
  const details = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    offenders: [...document.querySelectorAll("*")]
      .map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          className: element.className,
          tag: element.tagName,
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          width: Math.round(rect.width),
        };
      })
      .filter((item) => item.left < -1 || item.right > document.documentElement.clientWidth + 1)
      .slice(0, 12),
  }));
  errors.push(`mobile layout has horizontal overflow: ${JSON.stringify(details)}`);
}
await page.screenshot({
  path: path.join(outputDir, "dashboard-mobile.png"),
  fullPage: true,
});

console.log(
  JSON.stringify(
    {
      canvas: {
        width: png.width,
        height: png.height,
        luminanceRange: max - min,
        foregroundRatio: Number(canvasRatio.toFixed(4)),
      },
      errors,
      screenshots: outputDir,
    },
    null,
    2,
  ),
);

await browser.close();
server.kill();
if (errors.length) process.exitCode = 1;
