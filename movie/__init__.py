
from flask import Flask


app = Flask(__name__)

app.config['SECRET_KEY'] = 'ec9439cfc6c796ae2029594d'






from movie import routes