 from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    sl= float(request.values['SL'])
    sw= float(request.values['SW'])
    pl= float(request.values['PL'])
    pw= float(request.values['PW'])
    pr=np.array([[sl,sw,pl,pw]])
    output=model.predict(pr)
    def ot(output): 
        if (output==0):
            return 'Irix-Setosa'
        if(output==1):
            return 'Irix-Versicolor'
        else:
            return 'Irix-Verginica'
    return render_template ('result.html',prediction_text="This Iris Data belongs to {} Category".format(ot(output)))
if __name__=='__main__':
    app.run(port=8000)
    
