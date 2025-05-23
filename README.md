﻿# Gradio Gemini Classification

This is a Gradio app that make use of Generative AI like Google Gemini to execute function and classify loan approval status based on user personal details. The program works for the latest version of Gemini `gemini-2.5-flash-preview-04-17` and also the previous version `gemini-2.0-flash`. 

When user completed their personal details the details will be sent to Gemini for compilation and called for a classification function. Then the returned result will then be passed to Gemini and then to the Gradio app for display.

For detail set up please refer to the notebook folder `/notebook` as the code in the app can be a bit messy to think through.

## Demonstration

![alt text](image.png)

## Use Thie Repo

To use the repo:
```bash
git clone https://github.com/keanteng/gradio-chat-classification
```

Make sure you have the required libraries installed. You can do this by running:
```bash
pip install -r requirements.txt
```

## Run The App

To run the non-hot-reload version of the app, run the following command:
```bash
python app.py
```

To run the hot-reload version of the app, run the following command. Note that you need to update the file path:
```bash
./run.sh
``` 
