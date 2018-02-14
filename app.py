from flask import Flask, render_template
 
app = Flask(__name__)
str1='Wazzzzzupp'
 
@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('index.html', str1=str1)
 
if __name__ == '__main__':
    app.run()
    

