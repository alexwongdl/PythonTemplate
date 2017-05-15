import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define , options

from gtts import gTTS

class IndexHandler(tornado .web. RequestHandler):
    def get(self ):
        text = self. get_argument('text' , "" )
        tts = gTTS(text=text, lang='zh')
        tts.save("E://temp/tts/hello.mp3")
        self .write( 'TTS {} done!'.format(text))

if __name__ == "__main__":
    tornado.options .parse_command_line()
    app = tornado .web. Application(handlers =[(r"/", IndexHandler)])
    http_server = tornado .httpserver.HTTPServer(app )
    http_server.listen (18817)
    tornado.ioloop .IOLoop. current().start ()



