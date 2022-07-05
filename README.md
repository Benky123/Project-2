# Disaster Response Pipeline Project

### Table of Contents

1. [Overview of the Disaster Response Pipeline Project](#Overview)
2. [Data Prepocessing](#Preprocess)
3. [Machine Learning](#Learning)
4. [Website](#Website)
5. [How to run the project](#Intrucdition)

## Overview of the Disaster Response Pipeline Project <a name="Overview"></a>

In the Disaster Response Pipeline Project, I have created a data preprocessing pipeline and a machine learning pipeline to analyze the relationship amoung message and the classification results on the other 36 categories in the dataset.

## Data Prepocessing<a name="Preprocess"></a>

Preprocessing is done via ```data/process_data.py```  containing an ETL pipeline. 

## Machine Learning <a name="Learning"></a>

Machine Learning is done via ```models/train_classifier.py```  containing an ML pipeline.

## Website<a name="Website"></a>

Connecting the front-end and back-end of the website is done via ```app/run.py``` file containing the Flask.

## How to run the project<a name="Instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
