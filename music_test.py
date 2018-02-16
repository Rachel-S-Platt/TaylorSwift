import os

music_dir = './static/music'

print(os.listdir(music_dir))
music_files = [f for f in os.listdir(music_dir) if f.endswith('mp3')]
print(music_files)
