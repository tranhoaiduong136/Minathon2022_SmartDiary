import pyrebase
import tensorflow as tf
import numpy
from flask import Flask,render_template,Response,request,redirect,url_for
import cv2

# config = {
# 	"apiKey": "AIzaSyD2_FySPjBcalHhI5_whQKTOECGhBbm8S8",
# 	"authDomain": "moodify-a067c.firebaseapp.com",
# 	"projectId": "moodify-a067c",
# 	"databaseURL" : "https://moodify-a067c-default-rtdb.asia-southeast1.firebasedatabase.app",
# 	"storageBucket": "moodify-a067c.appspot.com",
# 	"messagingSenderId": "805380866668",
# 	"appId": "1:805380866668:web:9a2367c9bd583f672c4e49",
# 	"measurementId": "G-RPXX9LERM0"
# }

# firebase = pyrebase.initialize_app(config)
# database = firebase.database()
# storage = firebase.storage()

#Get model
#storage.child("final.h5").download("final.h5","final.h5")
model = tf.keras.models.load_model("final.h5")

app=Flask(__name__)
flagCam = 0
state = 0
import time
def generate_frames():
    global flagCam,state
    camera = cv2.VideoCapture(0)
    time = 0
    ans = [0,0,0,0,0,0,0]
    while True:
        time = time + 1
        ## read the camera frame
        success,frame=camera.read()
        if not success: 
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            temp = frame
            frame = cv2.resize(frame, (1280, 720))
            print(temp)
            img = temp
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = numpy.expand_dims(numpy.expand_dims(cv2.resize(img,(48,48)),-1),0)
            result = model.predict(img)
            print(result)
            index =  int(numpy.argmax(result))
            ans[index]  = ans[index] +1
            frame = buffer.tobytes()
            if flagCam == 1:
                flagCam = 0
                print(ans)
                max = ans[0] 
                state = 0
                for i in range(len(ans)):
                    if(ans[i] > max):
                        max = ans[i]
                        state = i
                camera.release()
                cv2.destroyAllWindows() 
                break
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/',methods=['GET','POST'])
def index1():
    global flagCam,state
    if request.method == 'POST':
            flagCam = 1
            if state == 0: 
                return render_template('angry.html')
            elif state == 1:
                state = 0
                return render_template('disgusted.html')
            elif state == 2:
                state = 0
                return render_template('fearful.html')
            elif state == 3:
                state = 0
                return render_template('happy.html')
            elif state == 4:
                state = 0
                return render_template('neutral.html')
            elif state == 5:
                state = 0
                return render_template('sad.html')
            elif state == 6:
                state = 0
                return render_template('suprised.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logo.png')
def image1():
    return render_template("logo.png")

@app.route('/hero-img.png')
def image2():
    return render_template("hero-img.png")

@app.route('/logo-page.png')
def image3():
    return render_template("hero-page.png")


if __name__=="__main__":
    app.run(debug=True)

