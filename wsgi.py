# for the gunicorn server
# Gunicorn ‘Green Unicorn’ is a Python WSGI HTTP Server for UNIX. 
# wsgi - web server gateway interface

from app import app

if __name__ == "__main__":
    app.run()