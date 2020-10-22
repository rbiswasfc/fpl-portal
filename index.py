from app import app, server
from callbacks import router

if __name__ == '__main__':
    app.run_server(debug=True)
