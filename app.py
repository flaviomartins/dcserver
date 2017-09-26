import io
import json
import requests
import logging
from wsgiref import simple_server

import falcon

import plac
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv, hsv2rgb
from skimage.color import rgb2lab, lab2rgb
from skimage.color import rgb2luv, luv2rgb

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('dcserver.log'))
logger.setLevel(logging.INFO)

ALLOWED_ORIGINS = ['*']
THUMBNAIL_SIZE = 224, 224


class CorsMiddleware(object):

    def process_request(self, req, resp):
        origin = req.get_header('Origin')
        if '*' in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
            resp.set_header('Access-Control-Allow-Origin', origin)


class RequestLoggerMiddleware(object):

    def process_request(self, req, resp):
        logger.info('{0} {1} {2} {3}'.format(' '.join(req.access_route), req.method, req.relative_uri, resp.status[:3]))


# snipped from matplotlib
def rgb2hex(c):
    """Convert `c` to a hex color.
    Uses the #rrggbb format.
    """
    return "#" + "".join(format(int(np.round(val * 255)), "02x")
                         for val in c)


class DominantColorsResource(object):
    def __init__(self):
        pass

    @staticmethod
    def luv_method(img, n_clusters, n_colors):
        img = rgb2luv(img)

        # make img array for kmeans
        X = img.reshape(-1, 3)

        n_clusters = n_clusters if n_clusters > n_colors else n_colors
        km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=3, random_state=0)
        labels = km.fit_predict(X)
        cluster_centers = km.cluster_centers_
        bincount = np.bincount(labels)

        # sort colors by frequency
        dominants = cluster_centers[np.argsort(bincount, axis=0)[::-1]][:n_colors]

        colors = [luv2rgb([[x]])[0][0] for x in dominants]
        return colors

    @staticmethod
    def lab_method(img, n_clusters, n_colors):
        img = rgb2lab(img)

        # make img array for kmeans
        X = img.reshape(-1, 3)

        n_clusters = n_clusters if n_clusters > n_colors else n_colors
        km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=3, random_state=0)
        labels = km.fit_predict(X)
        cluster_centers = km.cluster_centers_
        bincount = np.bincount(labels)

        # sort colors by frequency
        dominants = cluster_centers[np.argsort(bincount, axis=0)[::-1]][:n_colors]

        colors = [lab2rgb([[x]])[0][0] for x in dominants]
        return colors

    @staticmethod
    def hsv_method(img, n_clusters, n_colors):
        img = rgb2hsv(img)

        # make img array for kmeans
        X = img.reshape(-1, 3)

        n_clusters = n_clusters if n_clusters > n_colors else n_colors
        km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=3, random_state=0)
        labels = km.fit_predict(X)
        cluster_centers = km.cluster_centers_
        bincount = np.bincount(labels)

        # find index of data point closest to each center
        closest, _ = pairwise_distances_argmin_min(cluster_centers, X)

        # sort colors by frequency
        dominants = 1. * X[closest][np.argsort(bincount, axis=0)[::-1]][:n_colors]

        colors = [hsv2rgb([[x]])[0][0] for x in dominants]
        return colors

    def on_get(self, req, resp):
        url = req.get_param('url') or ''
        ncolors = req.get_param_as_int('ncolors') or 4
        nclusters = req.get_param_as_int('nclusters') or ncolors
        format = req.get_param('format') or 'json'
        mode = req.get_param('mode') or 'lab'

        try:
            r = requests.get(url)
            r.raise_for_status()
            r.raw.decode_content = True
            with io.BytesIO(r.content) as f:
                im = Image.open(f)
                im.thumbnail(THUMBNAIL_SIZE, Image.ANTIALIAS)
                img = np.array(im)
                if mode == 'hsv':
                    colors = self.hsv_method(img, nclusters, ncolors)
                elif mode == 'luv':
                    colors = self.luv_method(img, nclusters, ncolors)
                else:
                    colors = self.lab_method(img, nclusters, ncolors)

            colors = [rgb2hex(x) for x in colors]

            if format == 'json':
                result = json.dumps({'colors': np.array(colors).tolist()})
                resp.content_type = falcon.MEDIA_JSON
            else:
                result = '<html>\n<body>\n'
                for x in colors:
                    result += '<div style="font-family: monospace; background: ' + x + ';' \
                                                                                       'padding: 1em">' + x + \
                              '</div>\n'
                result += '</body>\n</html>'
                resp.content_type = falcon.MEDIA_HTML
        except Exception as ex:
            logger.error(ex)

            description = ('Aliens have attacked our base! We will '
                           'be back as soon as we fight them off. '
                           'We appreciate your patience.')

            raise falcon.HTTPServiceUnavailable(
                'Service Outage',
                description,
                30)

        resp.body = result

        resp.set_header('Powered-By', 'Falcon')
        resp.status = falcon.HTTP_200


# Useful for debugging problems in your API; works with pdb.set_trace(). You
# can also use Gunicorn to host your app. Gunicorn can be configured to
# auto-restart workers when it detects a code change, and it also works
# with pdb.


@plac.annotations(
    # in_model=("Location of input model"),
    host=("Bind to host", "option", "b", str),
    port=("Bind to port", "option", "p", int),
)
def main(host='127.0.0.1', port=8001):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Configure your WSGI server to load "quotes.app" (app is a WSGI callable)
    app = falcon.API(middleware=[
        CorsMiddleware(),
        RequestLoggerMiddleware()
    ])

    dominant_colors = DominantColorsResource()
    app.add_route('/', dominant_colors)

    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()


if __name__ == '__main__':
    plac.call(main)
