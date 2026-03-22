# Burmese Dataset Translator

A command-line tool for translating English text datasets into Burmese using the Gemini API.

This tool is designed for AI/ML dataset preparation and performs safe batch translation with automatic error handling and API limit protection.

https://www.npmjs.com/package/dataset-translator

---

## Features

* Batch translation to reduce API requests
* Token-aware batching for reliable processing
* Automatic retry for temporary API errors
* Safe stop when API quota is exceeded
* Saves translated and untranslated rows separately
* Clean command-line output
* Works with CSV datasets

---

## Installation

You can run the tool without cloning the repository.

### Install Globally

Install the CLI globally using npm:

```bash
npm install -g dataset-translator
```

After installation, run the tool from anywhere:

```bash
dataset-translator
```

---

### Run Without Installing (npx)

You can also run the tool directly using npx:

```bash
npx dataset-translator
```

---

## Requirements

* Node.js 18 or later
* Gemini API key

The translator uses the model:

* Gemini 2.5 Flash

---

## Dataset Format

Your CSV file must contain the following columns:

| Column | Description                   |
| ------ | ----------------------------- |
| text   | English sentence to translate |
| label  | Classification label          |

Example dataset:

```csv
text,label
I love programming.,2
This is very sad.,0
The movie was amazing.,1
```

---

## Usage

Run the command:

```bash
dataset-translator
```

The CLI will ask for:

1. CSV dataset path
2. Gemini API key

Example:

```
Burmese Dataset Translator

Enter CSV dataset path: example_dataset.csv
Enter Gemini API Key: **********************
```

---

## Output

The tool creates translated files in batches.

Example output:

```
burmese_dataset_0-457.csv
burmese_dataset_457-855.csv
untranslated_dataset_855-953.csv
```

### File Types

**Translated files**

```
burmese_dataset_*.csv
```

Contain translated Burmese text.

**Untranslated files**

```
untranslated_dataset_*.csv
```

Contain rows that could not be translated due to API limits.

---

## Error Handling

The translator includes automatic protection against API issues.

### Temporary API Errors

The tool retries automatically.

### API Quota Exceeded

If the Gemini API quota is reached:

* Translation stops safely
* Remaining rows are saved to `untranslated_dataset`
* No data is lost

---

## Typical Workflow

1. Prepare dataset CSV file
2. Run the translator
3. Wait for batch translation
4. Use translated dataset for training or analysis

---

## Use Cases

This tool is useful for:

* Sentiment analysis datasets
* NLP dataset localization
* Multilingual AI training
* Dataset augmentation

---

## License

MIT License
