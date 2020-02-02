import base64, fastai2
from flask import Flask, request, render_template, make_response
from fastai2.vision.all import *
from fastai2.callback.all import *
from fastai2.basics import *
from io import BytesIO
from skimage import io
from utils import make_mnist
from datetime import datetime

app = Flask(__name__, static_url_path='/static')

learner = load_learner('model/model.pkl')

@app.route('/')
def display_gui():
    return render_template('template.html')

@app.route('/recognizer', methods=['POST'])
def recognize():
    data = request.get_json(silent=True)['image']
    
    data = data[22:]

    img = io.imread(BytesIO(base64.b64decode(data)))[:,:,3]
    img = make_mnist(img)

    fn = datetime.timestamp(datetime.now())
    io.imsave(f"./test_folder/{fn}.png", img)

    img = PILImage.create(f"./test_folder/{fn}.png")
    
    pred = learner.predict(img)[0]
    
    print(pred)

    return make_response(str(pred),200)


if __name__=='__main__':
    app.run(debug=True)
