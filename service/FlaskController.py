from flask import Flask, render_template, jsonify, request, Response, make_response, send_from_directory, abort,send_file
import biggan512 as bg
import mobilenet2 as cl
app = Flask(__name__)

@app.route("/create")
def create():
    img = bg.create()
    return send_file(img, mimetype='image/jpg')


@app.route('/up_photo', methods=['post'])
def up_photo():
    img = request.files.get('photo')

    data = img.read()
    pro = cl.predict(data)
    print()

    return pro

if __name__ == '__main__':
    app.run()
