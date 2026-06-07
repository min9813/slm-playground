const { test, expect } = require("@playwright/test");

test("generates audio from the browser UI", async ({ page }) => {
  await page.goto("http://127.0.0.1:8000", { waitUntil: "networkidle" });
  await page.fill("#textInput", "UIから生成しています。");
  await page.fill("#tokenNumber", "64");

  await Promise.all([
    page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/tts") &&
        response.request().method() === "POST",
      { timeout: 60_000 },
    ),
    page.click("#generateButton"),
  ]);

  await expect(page.locator("#audioPlayer")).toHaveAttribute("src", /\/outputs\//);
  await expect(page.locator("#totalTime")).not.toHaveText("0.00s");
  await expect(page.locator("#generationTime")).not.toHaveText("0.00s");
});
