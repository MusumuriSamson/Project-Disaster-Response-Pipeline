# Disaster Response Pipeline Project

### Contents
1. [Introduction](#introduction)
2. [File Descriptions](#files)
3. [Instructions](#instructions)
4. [License](#license)
5. [Screenshots](#screenshots)

<a name="introduction"></a>

## Introduction 
This project is part of Udacity's Data Scientist Nanodegree Program in conjunction with [Figure Eight](https://www.figure-eight.com/).

The pre-labeled disaster messages will be use in this project to construct a disaster response model that can categorize messages received in real time during a catastrophe event, allowing communications to be routed to the appropriate disaster response agency.

This project provides a web application that allows disaster response workers to input received messages and obtain categorization results.

<a name="files"></a>

## File Descriptions 
### 1. 'data' Folder
**disaster_messages.csv** - actual disaster messages (supplied by Figure Eight)br/>
**disaster_categories.csv** - message categoriesbr/>
**process_data.py** - ETL pipeline for loading, cleaning, extracting features, and storing data in SQLite databasebr/>
**process_data.ipynb**  Jupyter Notebook is being used to prepare the data i.e. Extract Transform Loadbr/>
**DisasterResponse.db** - Clean data is saved in a SQlite database.

### 2. 'models' Folder
**train_classifier.py** - Python script - ML pipeline for loading cleaned data, training the model, and saving the learned model as a pickle (.pkl) file for later usebr/>
**classifier.pkl** is a pickle file that holds a trained modelbr/>
**mlpipeline.ipynb** Jupyter Notebook was used to set up the ML pipeline.

### 3 'app' Folder 
**run.py** - python script to launch web app.<br/>
### 3.1 'templates' Folder 
Contains web dependent files (go.html and master.html) necessary to launch the web application.


<a name="introduction"></a>

## Instructions 
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3000/

<a name="license"></a>

## License 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="screenshots"></a>

## Screenshots 
1. Main Webpage
![alt text]<img src="https://github.com/MusumuriSamson/Project-Disaster-Response-Pipeline/blob/9d36ca550fd87f45a81dd77f89bf4512a7cf6557/img/Web%20main%202.jpg" width="100" height="100">
