'''
 Demo app with ngrok
'''
from pyngrok import ngrok
import uvicorn
import nest_asyncio
from server_app import app


def main():
    '''
    Running server using ngrok
    '''
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)


if __name__ == "__main__":
    # execute only if run as a script
    main()
