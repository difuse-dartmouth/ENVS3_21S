# ENVS3 DIFUSE Module
## James Busch & William Chen
This repo contains data and code for a web app designed for the visualization and analysis of geospatial data for the ENVS3 course. 

### Folder Structure
* Module Data: contains .csv files with the raw parish-level data used in the assignment. The current versions of the applications use parish_data_v6.csv
  * The git_shapefiles subfolder contains .geojson files used for the map visualizations. The current versions of the applications use parish_data_v6.json
* Module Applications contains the following subfolders:
  * Colab Notebook: contains a single .ipynb file for the Colab Notebook application. This was used as a mockup for the standalone Netlify web application.
  * Heroku Apps: contains three subfolders, each containing code referenced by Heroku to create apps corresponding to the three data visualizations in the Netlify web application.
  * Netlify App: contains code that is referenced by Netlify for generating the web application
* Module Assignment and Instructions: contains two .docx files, one that is a current version of the ENVS3 assignment and another which provides instructions for how to use each web application

