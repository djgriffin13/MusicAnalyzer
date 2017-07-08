import subprocess

for i in range(1,1001):
    subprocess.call(['ffmpeg', '-i', 'clips_45sec.tar/clips_45sec/clips_45seconds/'+str(i)+'.mp3',\
    'Samples/'+str(i)+'.wav'])