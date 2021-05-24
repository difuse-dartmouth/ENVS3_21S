# ENVS3 DIFUSE Module
## James Busch & William Chen
This repo contains data and code for a web app designed for the visualization and analysis of geospatial data for the ENVS3 course. 

### ENVS3 Heroku Web Application
* The web application developed for this project can be accessed at https://envs3-web-app.herokuapp.com/

### Folder Structure
* Text files, Procfile, app.py, and static/images: code and files referenced by Heroku for building the web application.
* Module Data: contains .csv files with the raw parish-level data used in the assignment. The current versions of the applications use parish_data_v6.csv, the v4.csv file contains additional variables not used in the final version.
  * The git_shapefiles subfolder contains .geojson files used for the map visualizations. The current versions of the applications use parish_data_v6.json
* Module Applications contains the following subfolders:
  * Colab Notebook: contains a single .ipynb file for the Colab Notebook application. This was used as a mockup for the standalone Heroku web application.
  * Heroku App: contains all of the necessary code and files for the Heroku web app (duplicate of the files contained in the root directory). Heroku references each file when building the web app from this repo. The app.py file contains all of the python code that is used for the creation of the Dash user interface and the data visualizations.
* Module Assignment and Instructions: contains three .docx files, 1) short answer version of the ENVS3 assignment 2) instructions for how to use the web application/Colab notebook and 3) supplemental information on the analyses included in the web application (e.g. what a correlation coefficient means)

