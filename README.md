# ENVS3 DIFUSE Module
## James Busch & William Chen
This repo contains data and code for a web app designed for the visualization and analysis of geospatial data for the ENVS3 course. 

### Folder Structure
* Text files, Procfile, app.py, and static/images: code and files referenced by Heroku for building the web application.
* Module Data: contains .csv files with the raw parish-level data used in the assignment. The current versions of the applications use parish_data_v6.csv
  * The git_shapefiles subfolder contains .geojson files used for the map visualizations. The current versions of the applications use parish_data_v6.json
* Module Applications contains the following subfolders:
  * Colab Notebook: contains a single .ipynb file for the Colab Notebook application. This was used as a mockup for the standalone Heroku web application.
  * Heroku App: contains all of the necessary code and files for the Heroku web app (duplicate of the files contained in the root directory). Heroku references each file when building the web app from this repo. The app.py file contains all of the python code that is used for the creation of the Dash user interface and the data visualizations.
* Module Assignment and Instructions: contains two .docx files, one that is a current version of the ENVS3 assignment and another which provides instructions for how to use each web application

