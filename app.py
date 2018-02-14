from flask import Flask, render_template, request
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES
 
app = Flask(__name__)
str1='Wazzzzzupp'

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        return filename
    return render_template('index.html')
 
@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('index.html', str1=str1)
 
if __name__ == '__main__':
    app.run()
    

