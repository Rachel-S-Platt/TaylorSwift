from flask import Flask, render_template, request
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES
import os

music_dir = './static/music'

app = Flask(__name__)
str1='Wazzzzzupp'

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    print(os.listdir(music_dir))
    song_to_play = -1
    music_files = [f for f in os.listdir(music_dir) if f.endswith('mp3')]
    music_files_number = len(music_files)
    print(music_files)
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        return filename
    return render_template('index.html', music_files = music_files, music_files_number = music_files_number, song_to_play=song_to_play)

@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('index.html', str1=str1)

if __name__ == '__main__':
    app.run()
