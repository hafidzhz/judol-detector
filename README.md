# judol-detector

This repository contains Python scripts to scrape YouTube comments from Indonesian videos and train a machine learning model for detecting gambling-related content. It's designed to support research or moderation efforts by collecting relevant data and training a classifier.

## Features

- Scrapes comments using the YouTube Data API
- Focus on Indonesian-language videos
- Text preprocessing: cleaning and Unicode normalization
- Train a machine learning model for classification
- Streamlit UI for easy interaction

## Requirements

- Python 3.7+
- YouTube Data API v3 key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/judol-detector.git
   cd judol-detector
   ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## API Key Setup
Before running the scraper, you must add your YouTube Data API key.
