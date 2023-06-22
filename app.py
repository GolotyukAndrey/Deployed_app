
import cv2
from flask import Flask, render_template, request, Response
from werkzeug.utils import send_from_directory
import os
from ultralytics import YOLO
from pytube import YouTube


#Get frame function from saved .mp4 file, returning stream of images with detected objects
def get_frame():
    video_path = filepath
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    directory = 'runs/detect/video/' + f.filename

    if not os.path.exists(directory):
        os.makedirs(directory)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('runs/detect/video/' + f.filename + '/result.mp4', fourcc, 30.0, (frame_width, frame_height))            

    model = YOLO('yolov8n.pt')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        res_plotted = results[0].plot()
        out.write(res_plotted)
        _, results = cv2.imencode('.jpg', res_plotted)
        yield(b'--frame\r\n'
                b'Content0Type: image/jpeg\r\n\r\n' + results.tobytes() + b'\r\n\r\n')
    print(f'Result saved in runs/detect/video/{f.filename}/result.mp4')

#Sending returned data from get_frame function    
def video_feed():
    print('Function called')
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#Downloading youtube video function, returning stream of images with detected objects
def get_video():
    try:
        yt = YouTube(video_link)
        print(f'Title: {yt.title}')
        print(f'Views: {yt.views}')
        print(f'Length: {yt.length}')
        ys = yt.streams.get_highest_resolution()

        set = {'>', '<', ':', '"', '/', '\\', '|', '*', '.', ',', '='}
        flag = 0
        for i in range(len(yt.title)):
            if yt.title[i] in set:
                flag = 1
                title = yt.title.replace(yt.title[i], ' ')  
        if flag == 0:
            title = yt.title
        print(f'Directory name: {title}')

        directory_uploads = 'uploads/' + title
        try:
            if not os.path.exists(directory_uploads):
                os.makedirs(directory_uploads)
        except Exception:
            print('Invalid video name')

        try:
            ys.download(directory_uploads, 'youtube_video.mp4')
            print('Download finished!')
        except Exception:
            print('Unable to download video')        

        video_path = directory_uploads + '/youtube_video.mp4'
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        directory = 'runs/detect/video/' + title

        if not os.path.exists(directory):
            os.makedirs(directory)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(directory + '/result.mp4', fourcc, 30.0, (frame_width, frame_height))            

        model = YOLO('yolov8n.pt')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)
            res_plotted = results[0].plot()
            out.write(res_plotted)
            _, results = cv2.imencode('.jpg', res_plotted)
            yield(b'--frame\r\n'
                    b'Content0Type: image/jpeg\r\n\r\n' + results.tobytes() + b'\r\n\r\n')
        print(f'Result saved in {directory}/result.mp4')
    except Exception:
        print('Wrong URL, please enter youtube link')

#Object detection function which use YOLOv8 frame by frame from web-camera and return stream of images with detected objects
def camera():
    model = YOLO('yolov8n.pt')
    video = cv2.VideoCapture(0)
    while True:
        success, image = video.read()
        if not success:
            break
        results = model(image)
        cv2.waitKey(1)
        res_plotted = results[0].plot()
        ret, results = cv2.imencode('.jpg', res_plotted)
        yield(b'--frame\r\n'
              b'Content0Type: image/jpeg\r\n\r\n' + results.tobytes() + b'\r\n\r\n')

#Display function which is returning the latest saved .jpg image with predictions from 'runs/detect' folder        
def display(filename):
    folder_path = 'runs/detect'
    subfolder = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolder, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path + '/' + latest_subfolder
    print('Printing directory', directory)
    files = os.listdir(directory)
    latest_file = files[0]

    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)
    file_extension = filename.rsplit('.', 1)[1].lower()
    environ = request.environ
    if file_extension == 'jpg':
        return send_from_directory(directory, latest_file, environ)
    else:
        return 'Invalid file format'


app = Flask(__name__)

@app.route('/')
def start_page():
    return render_template('index.html')

@app.route('/ObjectDetection_browse')
def ObjectDetection_browse():
    return render_template('ObjectDetectionBrowse.html')

@app.route('/ObjectDetection_browse', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        global f
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        global filepath
        filepath = os.path.join(basepath, 'uploads', f.filename)
        print('Upload folder is ', filepath)
        f.save(filepath)
        file_extension = f.filename.rsplit('.', 1)[1].lower()

        if file_extension == 'jpg':
            model = YOLO('yolov8n.pt')
            img = cv2.imread(filepath)
            model.predict(img, name='image', save=True)

            return display(f.filename)
        
        elif file_extension == 'mp4':    

            return video_feed()
                
@app.route('/ContactUs')
def ContactUs():
    return render_template('ContactUs.html')

@app.route('/ObjectDetection_videolink')
def ObjectDetection_videolink():
    return render_template('ObjectDetectionVideoLink.html')

@app.route('/ObjectDetection_videolink_detection',methods=['GET','POST'])
def youtube_detect():
    print('Function youtube_detect called')
    global video_link
    video_link = request.form.get('videolink')
    print(video_link)
    return Response(get_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ObjectDetection_camera')
def ObjectDetection_camera():
    return render_template('ObjectDetectionCamera.html')

@app.route('/ObjectDetection_camera_detection')
def camera_detect():
    print('Function camera_detect called')
    return Response(camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=='__main__':
    app.run(debug=False)