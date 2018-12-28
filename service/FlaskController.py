from flask import Flask, render_template, jsonify, request, Response, make_response, send_from_directory, abort,send_file
import biggan512 as bg
import mobilenet2 as mn
import service.pb_colorizer as colorize
app = Flask(__name__)
import PIL.Image as Image
import vgg.classify as classify
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
