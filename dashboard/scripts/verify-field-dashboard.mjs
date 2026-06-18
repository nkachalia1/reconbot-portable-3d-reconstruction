import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright-core";

const browserPath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const target = process.env.RECONBOT_DASHBOARD_URL || "http://127.0.0.1:5000";
const outputDir = path.resolve(import.meta.dirname, "..", "..", "outputs", "dashboard");
fs.mkdirSync(outputDir, { recursive: true });

const initialStatus = await fetch(`${target}/api/field/status`).then((response) => response.json());
if (initialStatus.session?.recording) {
  await fetch(`${target}/api/field/video/stop`, { method: "POST" });
}
if (initialStatus.session?.active) {
  await fetch(`${target}/api/field/session/stop`, { method: "POST" });
}

const browser = await chromium.launch({
  executablePath: browserPath,
  headless: true,
});
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
const page = await context.newPage();
const errors = [];
page.on("console", (message) => {
  if (message.type() === "error") errors.push(`console: ${message.text()}`);
});
page.on("pageerror", (error) => errors.push(`page: ${error.message}`));

await page.goto(target, { waitUntil: "networkidle" });
await page.getByRole("button", { name: "Field" }).click();
await page.getByText("Camera online").waitFor({ timeout: 30000 });
await page.getByLabel("Session ID").fill(`browser_validation_${Date.now()}`);
await page.getByLabel("Arc direction").selectOption("right");
await page.getByRole("button", { name: "Start session" }).click();
await page.getByRole("button", { name: "Start recording" }).waitFor();
await page.getByAltText("Live laptop webcam preview").waitFor();

const startResponse = page.waitForResponse(
  (response) =>
    response.url().endsWith("/api/field/video/start") &&
    response.request().method() === "POST",
  { timeout: 30000 },
);
await page.getByRole("button", { name: "Start recording" }).click();
if (!(await startResponse).ok()) throw new Error("Video start request failed");
await page.getByText("Recording in progress").waitFor();
await page.waitForTimeout(3500);
await page.screenshot({
  path: path.join(outputDir, "dashboard-field-recording.png"),
  fullPage: true,
});

const stopResponse = page.waitForResponse(
  (response) =>
    response.url().endsWith("/api/field/video/stop") &&
    response.request().method() === "POST",
  { timeout: 60000 },
);
await page.getByRole("button", { name: "Stop recording" }).click();
if (!(await stopResponse).ok()) throw new Error("Video stop request failed");
await page.getByRole("link", { name: "Download video" }).waitFor({ timeout: 30000 });
const video = page.locator(".live-frame video");
await video.waitFor();
await video.evaluate(async (element) => {
  element.muted = true;
  await element.play();
});
await page.waitForFunction(
  () => {
    const element = document.querySelector(".live-frame video");
    return element instanceof HTMLVideoElement && element.currentTime > 0.5;
  },
  undefined,
  { timeout: 15000 },
);
await page.screenshot({
  path: path.join(outputDir, "dashboard-field-live.png"),
  fullPage: true,
});

await page.setViewportSize({ width: 390, height: 844 });
const overflow = await page.evaluate(
  () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
);
if (overflow) errors.push("field dashboard has horizontal overflow on mobile");
await page.screenshot({
  path: path.join(outputDir, "dashboard-field-mobile.png"),
  fullPage: true,
});

await page.getByRole("button", { name: "Finish session" }).click();
await page.getByRole("button", { name: "Start session" }).waitFor();

const finalStatus = await fetch(`${target}/api/field/status`).then((response) => response.json());
const videoResponse = await fetch(`${target}/api/field/video.mp4`);
const videoBytes = (await videoResponse.arrayBuffer()).byteLength;
if (!videoResponse.ok || videoBytes < 1000) errors.push("recorded MP4 is missing or empty");
const summary = {
  target,
  durationSeconds: finalStatus.session?.video_metadata?.duration_s,
  frames: finalStatus.session?.video_metadata?.frames,
  videoBytes,
  errors,
  screenshots: outputDir,
};
console.log(JSON.stringify(summary, null, 2));

await browser.close();
if (errors.length) process.exitCode = 1;
