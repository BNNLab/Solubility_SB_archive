Website README

1. Running the website
- python flask module is used to build the website
- the website is run on the local server by running "form_download.py" in python i.e.
python form_download.py
- it may be convenient to run the python script in the background using nohup, and example is shown in "run.sh" with the output directed to "nohup.out"
- tested in Ubuntu and Windows, optimal performance in Chrome browser

2. Python
- the current set up uses python 3.8 and an anaconda virtual environment
- the envirnment is stored in "environment.yml" and can be recreated thus:
conda env create -f environment.yml

3. Files and folders
form_download.py - the "master" python file which runs the website, the comments within this file explain how the website functions and is built
input_files.py, models.py, and models_PM6.py - python methods to generate input files and run models, these are called within form_download.py
run.sh - example of a nohup script to run the website in the background of a server
nohup.out - the output of run.sh
environment.yml - the conda environment containing the python and modules required
Input_files/ - The Gaussian 09 input files are generated here before the user downloads them
Logging/ - Information about numbers of visitors and solubility predictions made are stored here
model_data/ - The solubility datasets and saved models for ET, SVM, GP, and ANN are stored here
static/ - Any file, such as images and css, needing to be displayed are stored here
templates/ - The html source code of the webpages to be displayed
Uploads/ - The .log files for solubility are uploaded here and the solubility prediction results

4. Structure of website
solubility/home - home page
solubility/input - generate input files for Gaussian 09
solubility/models - upload the .log files from Gaussian 09
solubility/success - download successful solubility prediction