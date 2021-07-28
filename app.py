# -*- coding: utf-8 -*-

from __future__ import division, print_function
# coding=utf-8
#import sys
import os
import glob
import re
#import flask
import numpy as np
import tensorflow as tf
tf.config.experimental.list_physical_devices('CPU')
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_inception.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="abhay_deol"
    elif preds==1:
        preds="adil_hussain"
    elif preds==2:
        preds="ajay_devgn"
    elif preds==3:
        preds="akshay_kumar"
    elif preds==4:
        preds="akshaye_khanna"
    elif preds==5:
        preds="amitabh_bachchan"
    elif preds==6:
        preds="amjad_khan"
    elif preds==7:
        preds="amol_palekar"
    elif preds==8:
        preds="amole_gupte"
    elif preds==9:
        preds="amrish_puri"
    elif preds==10:
        preds="anil_kapoor"
    elif preds==11:
        preds="annu_kapoor"
    elif preds==12:
        preds="anupam_kher"
    elif preds==13:
        preds="anushka_shetty"
    elif preds==14:
        preds="arshad_warsi"
    elif preds==15:
        preds="aruna_irani"
    elif preds==16:
        preds="ashish_vidyarthi"
    elif preds==17:
        preds="asrani"
    elif preds==18:
        preds="atul_kulkarni"
    elif preds==19:
        preds="ayushmann_khurrana"
    elif preds==20:
        preds="boman_irani"
    elif preds==21:
        preds="chiranjeevi"
    elif preds==22:
        preds="chunky_panday"
    elif preds==23:
        preds="danny_denzongpa"
    elif preds==24:
        preds="darsheel_safary"
    elif preds==25:
        preds="deepika_padukone"
    elif preds==26:
        preds="deepti_naval"
    elif preds==27:
        preds="dev_anand"
    elif preds==28:
        preds="dharmendra"
    elif preds==29:
        preds="dilip_kumar"
    elif preds==30:
        preds="dimple_kapadia"
    elif preds==31:
        preds="farhan_akhtar"
    elif preds==32:
        preds="farida_jalal"
    elif preds==33:
        preds="farooq_shaikh"
    elif preds==34:
        preds="girish_karnad"
    elif preds==35:
        preds="govinda"
    elif preds==36:
        preds="gulshan_grover"
    elif preds==37:
        preds="hrithik_roshan"
    elif preds==38:
        preds="huma_qureshi"
    elif preds==39:
        preds="irrfan_khan"
    elif preds==40:
        preds="jaspal_bhatti"
    elif preds==41:
        preds="jeetendra"
    elif preds==42:
        preds="jimmy_sheirgill"
    elif preds==43:
        preds="johnny_lever"
    elif preds==44:
        preds="kader_khan"
    elif preds==45:
        preds="kajol"
    elif preds==46:
        preds="kalki_koechlin"
    elif preds==47:
        preds="kamal_haasan"
    elif preds==48:
        preds="kangana_ranaut"
    elif preds==49:
        preds="kay_kay_menon"
    elif preds==50:
        preds="konkona_sen_sharma"
    elif preds==51:
        preds="kulbhushan_kharbanda"
    elif preds==52:
        preds="lara_dutta"
    elif preds==53:
        preds="madhavan"
    elif preds==54:
        preds="madhuri_dixit"
    elif preds==55:
        preds="mammootty"
    elif preds==56:
        preds="manoj_bajpayee"
    elif preds==57:
        preds="manoj_pahwa"
    elif preds==58:
        preds="mehmood"
    elif preds==59:
        preds="mita_vashisht"
    elif preds==60:
        preds="mithun_chakraborty"
    elif preds==61:
        preds="mohanlal"
    elif preds==62:
        preds="mohnish_bahl"
    elif preds==63:
        preds="mukesh_khanna"
    elif preds==64:
        preds="mukul_dev"
    elif preds==65:
        preds="nagarjuna_akkineni"
    elif preds==66:
        preds="nana_patekar"
    elif preds==67:
        preds="nandita_das"
    elif preds==68:
        preds="nargis"
    elif preds==69:
        preds="naseeruddin_shah"
    elif preds==70:
        preds="navin_nischol"
    elif preds==71:
        preds="nawazuddin_siddiqui"
    elif preds==72:
        preds="neeraj_kabi"
    elif preds==72:
        preds="nirupa_roy"
    elif preds==74:
        preds="om_puri"
    elif preds==75:
        preds="pankaj_kapur"
    elif preds==76:
        preds="pankaj_tripathi"
    elif preds==77:
        preds="paresh_rawal"
    elif preds==78:
        preds="pawan_malhotra"
    elif preds==79:
        preds="pooja_bhatt"
    elif preds==80:
        preds="prabhas"
    elif preds==81:
        preds="prabhu_deva"
    elif preds==82:
        preds="prakash_raj"
    elif preds==83:
        preds="pran"
    elif preds==84:
        preds="prem_chopra"
    elif preds==85:
        preds="priyanka_chopra"
    elif preds==86:
        preds="raaj_kumar"
    elif preds==87:
        preds="radhika_apte"
    elif preds==88:
        preds="rahul_bose"
    elif preds==89:
        preds="raj_babbar"
    elif preds==90:
        preds="raj_kapoor"
    elif preds==91:
        preds="rajat_kapoor"
    elif preds==92:
        preds="gulshan_grover"
    elif preds==93:
        preds="rajesh_khanna"
    elif preds==94:
        preds="rajinikanth"
    elif preds==95:
        preds="rajit_kapoor"
    elif preds==96:
        preds="rajkummar_rao"
    elif preds==97:
        preds="rajpal_yadav"
    elif preds==98:
        preds="rakhee_gulzar"
    elif preds==99:
        preds="ramya_krishnan"
    elif preds==100:
        preds="ranbir_kapoor"
    elif preds==101:
        preds="randeep_hooda"
    elif preds==102:
        preds="rani_mukerji"
    elif preds==103:
        preds="ranveer_singh"
    elif preds==104:
        preds="ranvir_shorey"
    elif preds==105:
        preds="ratna_pathak_shah"
    elif preds==106:
        preds="rekha"
    elif preds==107:
        preds="richa_chadha"
    elif preds==108:
        preds="rishi_kapoor"
    elif preds==109:
        preds="riteish_deshmukh"
    elif preds==110:
        preds="sachin_khedekar"
    elif preds==111:
        preds="saeed_jaffrey"
    elif preds==112:
        preds="saif_ali_khan"
    elif preds==113:
        preds="salman_khan"
    elif preds==114:
        preds="sanjay_dutt"
    elif preds==115:
        preds="sanjay_mishra"
    elif preds==116:
        preds="shabana_azmi"
    elif preds==117:
        preds="shah_rukh_khan"
    elif preds==118:
        preds="sharman_joshi"
    elif preds==119:
        preds="sharmila_tagore"
    elif preds==120:
        preds="shashi_kapoor"
    elif preds==121:
        preds="shreyas_talpade"
    elif preds==122:
        preds="soumitra_chatterjee"
    elif preds==123:
        preds="sridevi"
    elif preds==124:
        preds="sunil_shetty"
    elif preds==125:
        preds="sunny_deol"
    elif preds==126:
        preds="tabu"
    elif preds==127:
        preds="tinnu_anand"
    elif preds==128:
        preds="utpal_dutt"
    elif preds==129:
        preds="varun_dhawan"
    elif preds==130:
        preds="vidya_balan"
    elif preds==131:
        preds="vinod_khanna"
    elif preds==132:
        preds="waheeda_rehman"
    elif preds==133:
        preds="zarina_wahab"
    elif preds==134:
        preds="zeenat_aman"
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
