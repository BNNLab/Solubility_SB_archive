'''
This python script runs the website using flask. Details on how to run the application can be found in README.txt.
Flask defines what happens at every url and renders templates (in templates/ folder) to display on the webpage. This allows python to happen in the background.
'''
###THIS AREA IS FOR IMPORTING MODULES###
# importing flask and other modules 
# python modules
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
from datetime import datetime
import os, uuid, time, csv
import pandas as pd
from zipfile import ZipFile
from werkzeug.utils import secure_filename
# own python methods
from models import get_final_pred
from models_PM6 import get_final_pred_PM6
from input_files import get_input_file_gas,get_input_file_sol
###THIS AREA IS FOR IMPORTING MODULES###END

###THIS IS THE OBJECT FOR CONTROLLING THE PREFIX TO THE WEBPAGES###
class PrefixMiddleware(object):
    def __init__(self, app, prefix=''):
        self.app = app
        self.prefix = prefix
    def __call__(self, environ, start_response):
        if environ['PATH_INFO'].startswith(self.prefix):
            environ['PATH_INFO'] = environ['PATH_INFO'][len(self.prefix):]
            environ['SCRIPT_NAME'] = self.prefix
            return self.app(environ, start_response)
        else:
            start_response('404', [('Content-Type', 'text/plain')])
            return ["This url does not belong to the app.".encode()]
###THIS IS THE OBJECT FOR CONTROLLING THE PREFIX TO THE WEBPAGES###END

###THIS AREA IS FOR CREATING APP###
# Flask constructor 
app = Flask(__name__)
#maximum file size upload size for part 2 (currently 100MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1000 * 1000
###THIS AREA IS FOR CREATING APP###END

###THIS AREA IS FOR DEFINING WEBPAGE CONTENT###
# A decorator used to tell the application which URL is associated function
# GET means nothing has been uploads, POST means the user has enter information
##the logic is based around "if POST" (do something if you've clicked a button, otherwise display page)
# home page
@app.route('/home', methods =["GET", "POST"]) 
def home():
    # define a counter to track visits
    counter=[]
    # get current count
    f=open('Logging/counter.txt','r')
    for line in f:
        counter.append(line)
    f.close()
    # add 1
    counter=int(counter[0])+1
    # save
    f=open('Logging/counter.txt','w')
    f.write(str(counter))
    f.close()
    # render correct html and pass counter information to it
    return render_template("home.html",counter=[counter])

# input page to get Gaussian 09 input files
@app.route('/input', methods =["GET", "POST"]) 
def gfg():
    # if user has posted information
    if request.method == "POST":
        ##remove old files if they exist
        ##get time and delete files older than 1 day
        current_time=time.time()
        for files in os.listdir('Input_files'):
            if files[-4:]==".com":
                creation_time = os.path.getctime('Input_files/' + files)
                if (current_time - creation_time) // (24 * 3600) >= 1:
                    os.remove('Input_files/' + files)
            if files[-4:]==".zip":
                creation_time = os.path.getctime('Input_files/' + files)
                if (current_time - creation_time) // (24 * 3600) >= 1:
                    os.remove('Input_files/' + files)
        # getting input with name = SMILES in HTML form
        SMILES = request.form['SMILES']
        # get solvent, ESM (DFT or PM6)
        solvent = request.form.get("Solvent")
        ESM = request.form["ESM"]
        # get input files or displey error page
        input_file_gas=get_input_file_gas(SMILES,ESM)
        if input_file_gas == ["Error in SMILES"]:
            return render_template("error.html")
        input_file_sol=get_input_file_sol(SMILES,solvent,ESM)
        if input_file_sol == ["Error in SMILES"]:
            return render_template("error.html")
        # get a random name for the input files
        inp_name=uuid.uuid4().hex + ".com"
        gas_name="gas_" + inp_name
        sol_name="sol_" + inp_name
        # write these files
        f = open("Input_files/" + gas_name,"a")
        for line in input_file_gas:
            f.write(line)
            f.write("\n")
        f.close()
        f = open("Input_files/" + sol_name,"a")
        for line in input_file_sol:
            f.write(line)
            f.write("\n")
        f.close()
        # zip up with random name
        zip_name=uuid.uuid4().hex + ".zip"
        with ZipFile('Input_files/' + zip_name,'w') as zip:
            zip.write('Input_files/' + gas_name)
            zip.write('Input_files/' + sol_name)
        # log input file creation
        with open('Logging/input_log.csv','a') as csvfile:
            writer = csv.writer(csvfile)
            now=datetime.now()
            writer.writerow([now.strftime("%d%m%Y %H:%M:%S"),SMILES,solvent,ESM])
        csvfile.close()
        ##send files to user
        return send_file("Input_files/" + zip_name, as_attachment=True, attachment_filename="Input_files.zip")
    # otherwise display the page
    return render_template("form_BN.html")###if nothing pressed display page


# models page to upload log files and get prediction
@app.route('/models', methods =["GET", "POST"]) 
def models():
    if request.method == "POST":
        ##delete previous results if exists
        ##get time and delete files older than 1 day
        current_time=time.time()
        for files in os.listdir('Uploads'):
            if files[-4:]==".csv":
                creation_time = os.path.getctime('Uploads/' + files)
                if (current_time - creation_time) // (24 * 3600) >= 1:
                    os.remove('Uploads/' + files)
            if files[-4:]==".log":
                creation_time = os.path.getctime('Uploads/' + files)
                if (current_time - creation_time) // (24 * 3600) >= 1:
                    os.remove('Uploads/' + files)
        ##check/get gas file
        file_gas = request.files['gas_file']
        filename_gas = secure_filename(file_gas.filename)
        #if no file reload page
        if filename_gas == "":
            flash('Choose files to upload')
            return render_template("models.html")
        ##check/get sol file
        file_sol = request.files['sol_file']
        filename_sol = secure_filename(file_sol.filename)
        #if no file reload page
        if filename_sol == "":
            flash('Choose files to upload')
            return render_template("models.html")
        #get MP
        MP = request.form.get("MP")
        if MP != "":
            try:
                MP=float(MP)
            except:
                flash('Please enter a valid melting point or leave blank to not use melting point in model')
                return render_template("models.html")
        #get ESM
        ESM = request.form.get("ESM")
		#save files
        ##random for gas and sol files, save and rename
        g_name = uuid.uuid4().hex + ".log"
        file_gas.save(filename_gas)
        os.rename(filename_gas, "Uploads/" + g_name)
        s_name = uuid.uuid4().hex + ".log"
        file_sol.save(filename_sol)
        os.rename(filename_sol, "Uploads/" + s_name)
        ##do prediction	if possible otherwise display error page	
        path="Uploads"
        if ESM == "DFT":
            try:
                solvent,new_pred=get_final_pred(path,g_name,s_name,MP)
            except:
                try:
                    os.remove("Uploads/" + g_name)
                except:
                    pass
                try:
                    os.remove("Uploads/" + s_name)
                except:
                    pass
                try:
                    os.remove("Uploads/" + s_name.replace(".log",".xyz"))
                except:
                    pass
                return render_template("error_model.html")
        if ESM == "PM6":
            try:
                solvent,new_pred=get_final_pred_PM6(path,g_name,s_name,MP)
            except:
                try:
                    os.remove("Uploads/" + g_name)
                except:
                    pass
                try:
                    os.remove("Uploads/" + s_name)
                except:
                    pass
                try:
                    os.remove("Uploads/" + s_name.replace(".log",".xyz"))
                except:
                    pass
                return render_template("error_model.html")
        # random name for results .csv file
        csv_name=uuid.uuid4().hex + ".csv"
        new_pred.to_csv("Uploads/" + csv_name,index=False)
        ##remove uploaded files
        os.remove("Uploads/" + g_name)
        os.remove("Uploads/" + s_name)
        os.remove("Uploads/" + s_name.replace(".log",".xyz"))
        ##log input file creation
        with open('Logging/models_log.csv','a') as csvfile:
            writer = csv.writer(csvfile)
            now=datetime.now()
            writer.writerow([now.strftime("%d%m%Y %H:%M:%S"),solvent,ESM,MP])
        csvfile.close()
        ##send to success page
        return redirect(url_for('success',csv_name=csv_name))
    return render_template("models.html")

# success page to download predictions
@app.route('/success', methods =["GET", "POST"])
def success():
    if request.method == "POST":
        ##download button pressed
        csv_name=request.args["csv_name"]
        return send_file("Uploads/" + csv_name, as_attachment=True, attachment_filename="solubility_results.csv")
    return render_template("success.html")
###THIS AREA IS FOR DEFINING WEBPAGE CONTENT###END

# make every page have the prefix "solubility" in the url
app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix = '/solubility')

###THIS RUNS THE APP###
if __name__=='__main__':
    app.secret_key = 'super secret key'
    app.run() 
###THIS RUNS THE APP###END

