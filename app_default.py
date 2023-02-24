from flask import Flask, request, jsonify, send_file
import base64
import io
import flask_monitoringdashboard as dashboard

app = Flask(__name__)
dashboard.bind(app)

@app.route("/feedback", methods=["POST"])
def create_feedback():
    encoded_tring = request.json["video"]
    # with open("res.mp4", "wb") as fh:
    #     fh.write(base64.b64decode(encoded_tring))

    # return send_file(io.BytesIO(video), mimetype="video/mp4")
    # encoded_string = base64.b64encode(video).decode()
    return jsonify({
        "video": encoded_tring,
        "score": 100
    })


@app.route("/test", methods=["GET", "POST"])
def test():
    return jsonify({
        "score": 100
    })


if __name__ == "__main__":
    app.run()