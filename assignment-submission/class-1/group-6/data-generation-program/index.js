#!/usr/bin/env node

import prompts from "prompts";
import fs from "fs";
import ora from "ora";
import chalk from "chalk";
import { performance } from "perf_hooks";
import { GoogleGenerativeAI } from "@google/generative-ai";

const emotions = {
  sadness: { name: "Sadness", id: 0 },
  joy: { name: "Joy", id: 1 },
  love: { name: "Love", id: 2 },
  anger: { name: "Anger", id: 3 },
  fear: { name: "Fear", id: 4 },
  surprise: { name: "Surprise", id: 5 },
};

async function generateBatch(model, emotionName, batchSize) {
  const prompt = `
Generate ${batchSize} UNIQUE Burmese quotes expressing "${emotionName}".

Rules:
- Burmese language only
- One quote per line
- No numbering
- No explanation
- No emojis
- Each quote must be different
- Each quote should be 1–2 sentences
- Quotes should be expressive and slightly longer (not too short, not paragraphs)
- Avoid repeating phrases or sentence patterns
`;

  const result = await model.generateContent({
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: {
      temperature: 1.1,
      topP: 0.95,
      topK: 50,
    },
  });

  const text = result.response.text();

  return text
    .split("\n")
    .map((q) => q.trim())
    .filter((q) => q.length > 0);
}

function formatTime(ms) {
  const seconds = (ms / 1000).toFixed(2);
  return `${seconds}s`;
}

async function main() {
  let spinner;

  try {
    console.log(
      chalk.cyan.bold("\nBurmese Quote Generator (Gemini)\n")
    );

    const args = process.argv.slice(2);

    let emotionArg = args[0];
    let limitArg = args[1];
    let csvFlag = args.includes("--csv");

    let apiKey;
    let emotionName;
    let emotionId;
    let limit;
    let format;

    if (emotionArg && limitArg) {
      const keyPrompt = await prompts({
        type: "password",
        name: "value",
        message: "Enter Gemini API Key",
      });

      apiKey = keyPrompt.value;

      const emotionData = emotions[emotionArg?.toLowerCase()];

      if (!apiKey) {
        console.log(chalk.red("API Key is required."));
        process.exit(1);
      }

      if (!emotionData) {
        console.log(chalk.red("Invalid emotion."));
        console.log("Valid: sadness joy love anger fear surprise");
        process.exit(1);
      }

      emotionName = emotionData.name;
      emotionId = emotionData.id;

      limit = parseInt(limitArg);

      if (isNaN(limit) || limit <= 0) {
        console.log(chalk.red("Limit must be positive."));
        process.exit(1);
      }

      format = csvFlag ? "csv" : "txt";
    } else {
      const response = await prompts([
        {
          type: "password",
          name: "apiKey",
          message: "Enter Gemini API Key",
        },
        {
          type: "select",
          name: "emotion",
          message: "Select Emotion",
          choices: Object.keys(emotions).map((key) => ({
            title: emotions[key].name,
            value: key,
          })),
        },
        {
          type: "number",
          name: "limit",
          message: "Number of quotes",
        },
        {
          type: "select",
          name: "format",
          message: "Output format",
          choices: [
            { title: "TXT", value: "txt" },
            { title: "CSV", value: "csv" },
          ],
        },
      ]);

      apiKey = response.apiKey;

      const emotionData = emotions[response.emotion];

      emotionName = emotionData.name;
      emotionId = emotionData.id;

      limit = response.limit;
      format = response.format;

      if (!apiKey) {
        console.log(chalk.red("API Key is required."));
        process.exit(1);
      }
    }

    const startTime = performance.now();

    spinner = ora("Initializing Gemini...").start();

    const genAI = new GoogleGenerativeAI(apiKey);

    const model = genAI.getGenerativeModel({
      model: "models/gemini-2.5-flash",
    });

    const quotesSet = new Set();

    const batchSize = 30;
    const maxAttempts = 50;

    let attempts = 0;

    spinner.text = "Generating quotes...";

    while (quotesSet.size < limit && attempts < maxAttempts) {
      attempts++;

      const batch = await generateBatch(model, emotionName, batchSize);

      for (const q of batch) {
        if (quotesSet.size < limit) {
          quotesSet.add(q);
        }
      }

      spinner.text = `Generating quotes (${quotesSet.size}/${limit})`;
    }

    spinner.succeed("Generation complete");

    const quotes = Array.from(quotesSet).slice(0, limit);

    if (quotes.length === 0) {
      console.log(chalk.red("No quotes generated."));
      process.exit(1);
    }

    let output;

    if (format === "csv") {
      const header = "text,label";

      const rows = quotes.map((q) => {
        const escaped = q.replace(/"/g, '""');
        return `"${escaped}",${emotionId}`;
      });

      output = [header, ...rows].join("\n");
    } else {
      output = quotes.join("\n");
    }

    const filename = `burmese_quotes_${emotionName}_${limit}.${format}`;

    fs.writeFileSync(filename, output);

    const endTime = performance.now();

    const duration = formatTime(endTime - startTime);

    console.log("");
    console.log(chalk.green("File saved:"), chalk.white(filename));
    console.log(chalk.green("Quotes generated:"), quotes.length);
    console.log(chalk.green("Time taken:"), duration);
    console.log("");
  } catch (error) {
    if (spinner) spinner.fail("Generation failed");

    const status = error?.status || error?.response?.status;

    console.log("");

    if (status === 401) {
      console.log(chalk.red("Invalid Gemini API key."));
    } else if (status === 404) {
      console.log(chalk.red("Gemini model not available."));
    } else if (status === 429) {
      console.log(chalk.red("Rate limit exceeded."));
    } else if (error.code === "ENOTFOUND") {
      console.log(chalk.red("Network error."));
    } else {
      console.log(chalk.red("Unexpected error."));
      console.log(error.message);
    }

    process.exit(1);
  }
}

main();