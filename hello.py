from flask import Flask, redirect, url_for, request,render_template
import numpy as np
import KNN as m1
import Linear as m2

app = Flask(__name__,static_url_path='/static')

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/')
def log():
   return render_template('Algo.html')

@app.route('/knn',methods = ['POST'])
def Knn():
    k = request.form['k']
    data1 = request.files['dataset1']
    data2 = request.files['dataset2']
    a = np.loadtxt(data1)
    b = np.loadtxt(data2)
    graph1,graph2,accuracy,cm = m1.model(k,a,b)
    print(k)
    accuracy = 100*accuracy
    #print(cm)
    print(accuracy)
    return render_template('knngraph.html', graph1 = graph1, graph2 = graph2,accuracy=accuracy,cm=cm)  

@app.route('/Linear',methods = ['POST'])
def Linear():
    data1 = request.files['dataset1']
    a = np.loadtxt(data1)
    graph1,graph2,accuracy = m2.model(a)
    accuracy = 100*accuracy
    return render_template('Lineargraph.html', graph1 = graph1, graph2 = graph2,accuracy=accuracy)
@app.route('/algo',methods = ['POST'])
def algo():
   k = request.form['algo']
   print(k)
   if k=="KNN":
      return render_template('knn.html')
   else:
      return render_template('Linear.html')
     
if __name__ == '__main__':
   app.run(debug = True,threaded=True)