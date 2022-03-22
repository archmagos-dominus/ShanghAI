#import bottle to handle connections with HourAI
from bottle import route, run

#define hello function
##route it to a page
@route('/hello')
def hello():
    return "blep"

#run the server on the host:port combination given
run(host='localhost', port=8080, debug=True)
