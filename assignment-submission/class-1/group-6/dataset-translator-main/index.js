#!/usr/bin/env node

import fs from "fs";
import prompts from "prompts";
import ora from "ora";
import csv from "csv-parser";
import chalk from "chalk";
import { createObjectCsvWriter } from "csv-writer";
import { GoogleGenerativeAI } from "@google/generative-ai";

const MAX_BATCH_TOKENS = 6000;
const REQUEST_DELAY = 1500;
const RETRY_LIMIT = 2;

const log = {
  info: (msg) => console.log(chalk.blue("[INFO]"), msg),
  success: (msg) => console.log(chalk.green("[SUCCESS]"), msg),
  warn: (msg) => console.log(chalk.yellow("[WARNING]"), msg),
  error: (msg) => console.log(chalk.red("[ERROR]"), msg),
  done: (msg) => console.log(chalk.cyan("[DONE]"), msg),
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

async function readCSV(filePath) {
  const rows = [];

  return new Promise((resolve, reject) => {
    fs.createReadStream(filePath)
      .pipe(csv())
      .on("data", (data) => rows.push(data))
      .on("end", () => resolve(rows))
      .on("error", reject);
  });
}

async function translateBatch(texts, apiKey) {
  const genAI = new GoogleGenerativeAI(apiKey);

  const model = genAI.getGenerativeModel({
    model: "models/gemini-2.5-flash",
  });

  const numbered = texts.map((t, i) => `${i + 1}. ${t}`).join("\n");

  const prompt = `
Translate the following English sentences into Burmese.

Rules:
- Translate only the sentence
- Do not add explanation
- Return EXACTLY the same number of lines
- Preserve numbering format

Sentences:
${numbered}
`;

  const result = await model.generateContent(prompt);

  const output = result.response.text();

  const translated = output
    .split("\n")
    .map((line) => line.replace(/^\d+\.\s*/, "").trim())
    .filter((t) => t.length > 0);

  return translated;
}

async function translateWithRetry(texts, apiKey, retries = RETRY_LIMIT) {
  try {
    return await translateBatch(texts, apiKey);
  } catch (err) {
    const status = err?.status;

    // Stop immediately on quota exceeded
    if (status === 429) {
      log.error("Gemini API daily quota exceeded.");
      throw new Error("QUOTA_EXCEEDED");
    }

    if (retries > 0) {
      log.warn("Temporary API error. Retrying...");
      await sleep(3000);
      return translateWithRetry(texts, apiKey, retries - 1);
    }

    throw err;
  }
}

async function saveCSV(rows, start, end, prefix) {
  const file = `${prefix}_${start}-${end}.csv`;

  const writer = createObjectCsvWriter({
    path: file,
    header: [
      { id: "text", title: "text" },
      { id: "label", title: "label" },
    ],
  });

  await writer.writeRecords(rows);

  log.success(`Saved file: ${file}`);
}

async function main() {
  console.log(chalk.bold("\nBurmese Dataset Translator\n"));

  const response = await prompts([
    {
      type: "text",
      name: "file",
      message: "Enter CSV dataset path",
    },
    {
      type: "password",
      name: "apiKey",
      message: "Enter Gemini API Key",
    },
  ]);

  const { file, apiKey } = response;

  if (!file || !fs.existsSync(file)) {
    log.error("Dataset file not found.");
    process.exit(1);
  }

  if (!apiKey) {
    log.error("API Key is required.");
    process.exit(1);
  }

  const spinner = ora("Reading dataset...").start();

  let rows;

  try {
    rows = await readCSV(file);
  } catch {
    spinner.fail("Failed to read CSV file.");
    process.exit(1);
  }

  spinner.stop();

  log.success(`Dataset loaded (${rows.length} rows)`);

  if (rows.length === 0) {
    log.error("Dataset is empty.");
    process.exit(1);
  }

  if (!rows[0].text || rows[0].label === undefined) {
    log.error("CSV must contain 'text' and 'label' columns.");
    process.exit(1);
  }

  let batch = [];
  let batchTokens = 0;
  let translatedCount = 0;
  let part = 1;

  const genSpinner = ora("Starting translation...").start();

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const tokens = estimateTokens(row.text);

    if (batchTokens + tokens > MAX_BATCH_TOKENS) {
      genSpinner.text = `Translating batch ${part} (${batch.length} rows)...`;

      const texts = batch.map((r) => r.text);

      let translated;

      try {
        translated = await translateWithRetry(texts, apiKey);
      } catch (err) {
        genSpinner.fail("Translation stopped.");

        if (err.message === "QUOTA_EXCEEDED") {
          const remaining = rows.slice(translatedCount);

          await saveCSV(
            remaining,
            translatedCount,
            rows.length,
            "untranslated_dataset"
          );

          log.info(`Translated rows: ${translatedCount}`);
          log.warn("Remaining rows saved as untranslated dataset");

          process.exit(0);
        }

        log.error("Gemini API error.");
        process.exit(1);
      }

      if (translated.length !== batch.length) {
        genSpinner.fail("API response truncated.");

        log.warn(
          `Requested ${batch.length} rows but received ${translated.length}`
        );

        const translatedRows = batch
          .slice(0, translated.length)
          .map((row, i) => ({
            text: translated[i],
            label: row.label,
          }));

        await saveCSV(
          translatedRows,
          translatedCount,
          translatedCount + translatedRows.length,
          "burmese_dataset"
        );

        const remaining = rows.slice(translatedCount + translatedRows.length);

        await saveCSV(
          remaining,
          translatedCount + translatedRows.length,
          rows.length,
          "untranslated_dataset"
        );

        process.exit(0);
      }

      const outputRows = batch.map((row, i) => ({
        text: translated[i],
        label: row.label,
      }));

      const start = translatedCount;
      const end = translatedCount + outputRows.length;

      // Stop spinner before saving so [SUCCESS] prints on its own line
      genSpinner.stop();

      await saveCSV(outputRows, start, end, "burmese_dataset");

      translatedCount += outputRows.length;
      part++;

      batch = [];
      batchTokens = 0;

      // Restart spinner for next batch
      genSpinner.start();

      await sleep(REQUEST_DELAY);
    }

    batch.push(row);
    batchTokens += tokens;
  }

  if (batch.length > 0) {
    genSpinner.text = `Translating final batch (${batch.length} rows)...`;

    const texts = batch.map((r) => r.text);

    try {
      const translated = await translateWithRetry(texts, apiKey);

      const outputRows = batch.map((row, i) => ({
        text: translated[i],
        label: row.label,
      }));

      const start = translatedCount;
      const end = translatedCount + outputRows.length;

      // Stop spinner before saving so [SUCCESS] prints on its own line
      genSpinner.stop();

      await saveCSV(outputRows, start, end, "burmese_dataset");

      translatedCount += outputRows.length;
    } catch (err) {
      genSpinner.fail("Translation stopped.");

      if (err.message === "QUOTA_EXCEEDED") {
        const remaining = rows.slice(translatedCount);

        await saveCSV(
          remaining,
          translatedCount,
          rows.length,
          "untranslated_dataset"
        );

        log.info(`Translated rows: ${translatedCount}`);
        log.warn("Remaining rows saved as untranslated dataset");

        process.exit(0);
      }

      log.error("Gemini API error.");
      process.exit(1);
    }
  }

  log.done("Translation completed.");
  log.info(`Total translated rows: ${translatedCount}`);
}

/* Prevent Node crash */
process.on("unhandledRejection", (err) => {
  log.error("Unhandled error occurred.");
  console.error(err.message);
  process.exit(1);
});

main();