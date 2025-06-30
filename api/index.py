from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Hello World!</h1><p>This is a test deployment on Vercel.</p>'

@app.route('/test')
def test():
    return {'status': 'ok', 'message': 'Test endpoint works'}

if __name__ == '__main__':
    app.run(debug=True) 