from flask import Flask, request, send_file
from engine.hub import biggan512 as bg, mobilenet2 as mn
import engine.third.pb_colorizer as colorize
import engine.vgg.classify
app = Flask(__name__)
import PIL.Image as Image


@app.route("/create")
def create():
    img = bg.create()
    return send_file(img, mimetype='image/jpg')

@app.route('/color', methods=['post'])
def color():
    file = request.files.get('photo')
    img = Image.open(file)
    img = img.resize((300, 200), Image.ANTIALIAS)
    file = colorize.colorize(img)
    return send_file(file, mimetype='image/jpg')
@app.route('/up_photo', methods=['post'])
def up_photo():
    img = request.files.get('photo')
    color.colorize(img)
    data = img.read()
    pro = mn.predict(data)
    print()

    return pro

if __name__ == '__main__':
    app.run()
