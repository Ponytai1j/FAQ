# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def hello():
#     return "Hello World123!"

# if __name__ == '__main__':
#     app.debug = False
#     app.run(host='localhost', port=5000)

from flask import Flask, redirect, url_for, request, render_template
from FAQ import FAQ
from tqdm import tqdm
import os
app = Flask(__name__)
faq = FAQ()
data_path = './Dataset/xxwx/FAQ_parent_education1-297.xls'
faq.init(datapath=data_path, indexpath='./config/')
faq.load_all()
@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/welcome/<name>')
def welcome(name):
    return f'welcome {name}'

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        query = request.form['username'] #拿到前端传输的表单数据
        faq_res = faq.search(query)
        res = faq_res[0]
        query = faq_res[2]
        sensitive_res = faq_res[1]
        context = {
        'q1': res[0][0],
        'a1': res[0][1],
        'p1': res[0][2],
        'q2': res[1][0],
        'a2': res[1][1],
        'p2': res[1][2],
        'q3': res[2][0],
        'a3': res[2][1],
        'p3': res[2][2],
        'q4': res[3][0],
        'a4': res[3][1],
        'p4': res[3][2],
        'q5': res[4][0],
        'a5': res[4][1],
        'p5': res[4][2],
        'query': query,
        'sensitive_detect1': query,
        'sensitive_detect2': sensitive_res,
        }
        return render_template("result.html",context=context)
        
    else:
        user = request.args.get('username')  # GET方法获取数据，args是包含表单参数对及其对应值对的列表的字典对象。
        return redirect(url_for('welcome', context=context))

if __name__ == '__main__':
    #app.run(debug=True,port=int(os.getenv('PORT', 4444)))
    app.run(debug=True,port=5000)
