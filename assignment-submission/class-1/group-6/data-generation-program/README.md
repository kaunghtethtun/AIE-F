# Burmese Quote Generator

CLI tool to generate Burmese quotes based on emotions using Google's Gemini API.

This tool helps developers and AI engineers quickly generate Burmese text datasets for sentiment analysis, NLP training, or content generation.

https://www.npmjs.com/package/burmese-quote-generator

---

## Features

* Generate Burmese quotes by emotion
* Supports multiple emotions
* Export quotes as **TXT** or **CSV**
* Interactive CLI mode
* Fast generation using Gemini API

---

## Installation

Run directly using **npx**:

```
npx burmese-quote-generator
```

Or install globally:

```
npm install -g burmese-quote-generator
```

---

## Usage

### Interactive Mode

```
burmese-quote-generator
```

You will be prompted for:

* Gemini API Key
* Emotion
* Number of quotes
* Output format

---

### CLI Mode

```
burmese-quote-generator <emotion> <limit>
```

Example:

```
burmese-quote-generator sadness 100
```

Generate CSV file:

```
burmese-quote-generator joy 200 --csv
```

---

## Supported Emotions

* sadness
* joy
* love
* anger
* fear
* surprise

---

## Output

The generated quotes will be saved as a file:

```
burmese_quotes_<Emotion>_<Limit>.txt
```

or

```
burmese_quotes_<Emotion>_<Limit>.csv
```

---

## Requirements

* Node.js 18 or higher
* Gemini API key

---

## Get Gemini API Key

Create an API key from:

https://ai.google.dev

---

## Example Output

```
အမှတ်တရတွေက စိတ်ထဲမှာ နေရာယူထားတယ်။
တိတ်တိတ်လေး လွမ်းနေတတ်တာက စိတ်ရဲ့အကျင့်ပါ။
မေ့ချင်လို့ မေ့လို့မရတဲ့ အချိန်တွေရှိတယ်။
```

---

## License

MIT License
