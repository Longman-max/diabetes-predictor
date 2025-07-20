from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({'message': 'Flask app is working on Vercel!'})

# Required for Vercel
def handler(environ, start_response):
    return app(environ, start_response)
