from app import app, server
from callbacks import router, callback_navigation, callback_league, callback_leads, callback_squad

if __name__ == '__main__':
    app.run_server(debug=True)
