import json
from flask import Flask, Response
app = Flask(__name__)

@app.route('/events')
def sse():
    def event_stream():
        count = 0
        yield f"data: {json.dumps({'message': 'hello world!'})}\n\n"
        # while True:
        #     count += 1
        #     yield f"data: {count}\n\n"
    return Response(event_stream(), content_type='text/event-stream')
