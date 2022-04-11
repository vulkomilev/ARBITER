import flask
from flask_classful import FlaskView
import sqlite3
from flask import Flask
from flask_jwt_extended import JWTManager
from datetime import timedelta

class Server(FlaskView):

    def __init__(self):
        pass


local_server = Server()

app = Flask(__name__)
app.route('/get_character_info', methods=['POST'])(local_server.get_character_info)


print('LOADING DONE>>>>>>')

app.config["JWT_SECRET_KEY"] = "REWRITE THIS"
jwt = JWTManager(app)

ACCESS_EXPIRES = timedelta(hours=1)

app.config["JWT_ACCESS_TOKEN_EXPIRES"] = ACCESS_EXPIRES
app.run(host='0.0.0.0', port=5001)
