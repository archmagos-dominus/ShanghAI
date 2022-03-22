#import bottle to handle connections with HourAI
from bottle import route, run

#define hello function
##decorator binds a piece of code to an URL path
##in this case, we link the /hello path to the hello() function
@route('/hello')
def hello():
    return "blep"

#run the server on the host:port combination given
run(host='localhost', port=8080, debug=True)
