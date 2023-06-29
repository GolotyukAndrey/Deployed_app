
import cv2
import os
import requests
from flask import Flask, render_template, request, Response
from werkzeug.utils import send_from_directory
from ultralytics import YOLO
from tqdm import tqdm


#Get frame function from saved .mp4 file, returning stream of images with detected objects
def get_frame(filepath, filename):
    cap = cv2.VideoCapture(filepath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    directory = 'runs/detect/video/' + filename

    if not os.path.exists(directory):
        os.makedirs(directory)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('runs/detect/video/' + filename + '/result.mp4', fourcc, 30.0, (frame_width, frame_height))            

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
    print(f'Result saved in runs/detect/video/{filename}/result.mp4')

#Sending returned data from get_frame function    
def video_feed(filepath, filename):
    print('Function called')
    return Response(get_frame(filepath, filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#Downloading youtube video function, returning stream of images with detected objects
def get_video(video_link):

    print(video_link)
    folder_counter = str(sum([len(folder) for r, d, folder in os.walk('uploads/youtube_video/')]))

    if '&list' in video_link:
        video_link = video_link.split("&")[0].split("=")[-1]
    else:
        video_link = video_link.split("=")[-1]

    directory_name = 'uploads/youtube_video/' + folder_counter
    headers = {
        'authority': 'downloader.freemake.com',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="98", "Yandex";v="22"',
        'dnt': '1',
        'x-cf-country': 'RU',
        'sec-ch-ua-mobile': '?0',
        'x-user-platform': 'Win32',
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'x-user-browser': 'YaBrowser',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/98.0.4758.141 YaBrowser/22.3.3.852 Yowser/2.5 Safari/537.36',
        'x-analytics-header': 'UA-18256617-1',
        'x-request-attempt': '1',
        'x-user-id': '94119398-e27a-3e13-be17-bbe7fbc25874',
        'sec-ch-ua-platform': '"Windows"',
        'origin': 'https://www.freemake.com',
        'sec-fetch-site': 'same-site',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': 'https://www.freemake.com/ru/free_video_downloader/',
        'accept-language': 'ru,en;q=0.9,uk;q=0.8',
    }

    print(f'Recieving video title and URL...')
    response = requests.get(f'https://downloader.freemake.com/api/videoinfo/{video_link}', headers=headers).json()
    video_title = str(response['metaInfo']['title'])
    for ch in ["?", '"', "'", "/", ":", "#", "|", ",", " | "]:
        video_title = video_title.replace(ch, "")
    url = response['qualities'][0]['url']
    print(f'Title and URL recieved. Starting download: "{video_title}"...')

    os.mkdir(directory_name)
    print(f'Creating folder for downloaded video...\n')

    req = requests.get(url=url, headers=headers, stream=True)
    total = int(req.headers.get('content-length', 0))
    with open(f'{os.path.join(directory_name, f"{video_title}.mp4")}', 'wb') as file, tqdm(
            desc=f"{video_title[0:int(len(video_title) / 2)]}...",
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in req.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    print(f'\nDownload finished.\n')

    print('directory_name = ', directory_name)
    print('video_title = ', video_title)  
    
    video_path = directory_name + '/' + video_title + '.mp4'
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    directory = 'runs/detect/video/' + video_title

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
        f = request.files['file']
        basepath = os.path.dirname(__file__)
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
            filename = f.filename
            return video_feed(filepath, filename)
                
@app.route('/ContactUs')
def ContactUs():
    return render_template('ContactUs.html')

@app.route('/ObjectDetection_videolink')
def ObjectDetection_videolink():
    return render_template('ObjectDetectionVideoLink.html')

@app.route('/ObjectDetection_videolink_detection',methods=['GET','POST'])
def youtube_detect():
    print('Function youtube_detect called')
    video_link = request.form.get('videolink')
    check_youtube = 'https://www.youtube.com/watch?v='
    start = video_link[:32:]
    if start == check_youtube and len(video_link) == 43:
        return Response(get_video(video_link),
                mimetype='multipart/x-mixed-replace; boundary=frame') 
    else:
        return render_template('exception.html')

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