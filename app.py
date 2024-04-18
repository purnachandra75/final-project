from flask import Flask,render_template,request

from resfile import predicton,yeildpreduction
app=Flask(__name__)
@app.route("/")
def index():
    return(render_template("index.html"))
@app.route("/result",methods=["POST"])

def result():
    l=[]
    
    if request.method=="POST":
        l.append(int(request.form["n"]))
        l.append(int(request.form["p"]))
        l.append(int(request.form["k"]))
        l.append(int(request.form["temp"]))
        l.append(int(request.form["humidity"]))
        l.append(int(request.form["ph"]))
        l.append(int(request.form["rainfall"]))
        l.append(int(request.form['season']))
        print(l)
        res=predicton(l)
        print(res)
        if res=="Rice":
            return(render_template('rice.html',res1=res))
        if res=="watermelon":
            return(render_template('result.html',res1=res))
        if res=="banana":
            return(render_template('banana.html',res1=res))
        if res=="garlic":
            return(render_template('garlic.html',res1=res))
        if res=="grapes":
            return(render_template('grapes.html',res1=res))
        if res=="mango":
            return(render_template('mango.html',res1=res))
        if res=="Onion":
            return(render_template('onion.html',res1=res))
        if res=="orange":
            return(render_template('orange.html',res1=res))
        if res=="papaya":
            return(render_template('papaya.html',res1=res))
        if res=="pomegranate":
            return(render_template('pomegranate.html',res1=res))
        if res=="Sugarcane":
            return(render_template('sugarcane.html',res1=res))
        if res=="Sunflower":
            return(render_template('sunflower.html',res1=res))
        if res=="sweetcorn":
            return(render_template('sweet corn.html',res1=res))
        if res=="apple":
            return(render_template('apple.html',res1=res))
        if res=="Barley":
            return(render_template('barley.html',res1=res))
        if res=="blackgram":
            return(render_template('black gram.html',res1=res))
        if res=="chickpea":
            return(render_template('chickpea.html',res1=res))
        if res=="Chillies":
            return(render_template('chilli.html',res1=res))
        if res=="coconut":
            return(render_template('coconut.html',res1=res))
        if res=="coffee":
            return(render_template('coffee.html',res1=res))
        if res=="Coriander":
            return(render_template('Coriander.html',res1=res))
        if res=="Cowpea(lobia)":
            return(render_template('cowpea.html',res1=res))
        if res=="Moong(Green Gram)":
            return(render_template('green gram.html',res1=res))
        if res=="kidneybeans":
            return(render_template('kidney beans.html',res1=res))
        if res=="lentil":
            return(render_template('lentil.html',res1=res))
        
        if res=="Linseed":
            return(render_template('linseed.html',res1=res))
        if res=="mothbeans":
            return(render_template('mothbeans.html',res1=res))
        if res=="muskmelon":
            return(render_template('muskmelon.html',res1=res))
        if res=="Potato":
            return(render_template('potato.html',res1=res))
        if res=="Turmeric":
            return(render_template('turmeric.html',res1=res))
        if res=="Wheat":
            return(render_template('wheat.html',res1=res))
        if res=="Safflower":
            return(render_template('saffron.html',res1=res))
        if res=="pigeonpeas":
            return(render_template('pigeonpea.html',res1=res))
        if res=="mungbean":
            return(render_template('mung beans.html',res1=res))
        if res=="cotton":
            return(render_template('cotton.html',res1=res))
        if res=="jute":
            return(render_template('jute.html',res1=res))
        if res=="Urad":
            return(render_template('urad.html',res1=res))
        if res=="Groundnut":
            return(render_template('groundnut.html',res1=res))
        if res=="Arhar/Tur":
            return(render_template('arhar.html',res1=res))
        if res=="Bajra":
            return(render_template('bajra.html',res1=res))
        if res=="Peas & beans (Pulses)":
            return(render_template('peas.html',res1=res))
        if res=="Masoor":
            return(render_template('masoar.html',res1=res))
        if res=="Ragi":
            return(render_template('ragi.html',res1=res))
        if res=="Jowar":
            return(render_template('Jowar.html',res1=res))
@app.route("/yield1",methods=["POST"])
def yieldfun1():
    l1=[]
    if request.method=="POST":
        l1.append(int(request.form['crop']))
        l1.append(float(request.form['area']))
        l1.append(int(request.form['state']))
        l1.append(int(request.form['season']))
        l1.append(int(request.form['year']))
        resl=yeildpreduction(l1)
        return(render_template('yeild_prediction.html',res=resl))

@app.route("/yield")
def yieldfun():
    return(render_template('yeild_prediction.html'))

'''@app.route("/result1",methods=["POST"])
def result1():
    l1=[]
    if request.method=="POST":
        l1.append(int(request.form['crop']))
        
        l1.append(float(request.form['area']))
        l1.append(int(request.form['state']))
        l1.append(int(request.form['season']))
        resl=yeildpreduction(l1)
        
        return(render_template('result.html',res11=resl))
        print(resl)'''
if __name__=="__main__":
    app.run(debug=True)
