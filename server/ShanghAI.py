#import bottle to handle connections with HourAI
from bottle import get, post, request, run

#define hello function
##decorator binds a piece of code to an URL path
##in this case, we link the /hello path to the hello() function
@post('/load', method='POST')
def load():
    reveived_data = request.json.get('model')
    print(reveived_data)
    payload = {"reply": "Loading model..."}
    reply = dict(data=payload)
    return reply

#run the server on the host:port combination given
run(host='localhost', port=8000, debug=True)
