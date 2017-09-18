import io
import json
import requests
import logging
from wsgiref import simple_server

import falcon

import plac
import numpy as np
from PIL import Image
from skimage.color import lab2rgb, hsv2rgb
import cv2

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = ['*']
HIST_BINS = 16


class CorsMiddleware(object):
    def process_request(self, request, response):
        origin = request.get_header('Origin')
        if '*' in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
            response.set_header('Access-Control-Allow-Origin', origin)


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
    def lab_method(img, ncenters, ncolors):
        img = cv2.resize(np.array(img), (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        channels = [0, 1, 2]
        mask = None

        histSize = [HIST_BINS, HIST_BINS, HIST_BINS]
        lranges = [0, 256]
        aranges = [0, 256]
        branges = [0, 256]
        ranges = [item for sublist in [lranges, aranges, branges] for item in sublist]

        hist = cv2.calcHist([img], channels, mask, histSize, ranges)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # make img array for kmeans
        ar = img.reshape(-1, 3)

        def lab_quantize(hist, x):
            l = np.floor(x[0] / 256.0 * HIST_BINS).astype(int)
            a = np.floor(x[1] / 256.0 * HIST_BINS).astype(int)
            b = np.floor(x[2] / 256.0 * HIST_BINS).astype(int)
            return hist[l, a, b]

        hist_weights = np.apply_along_axis(lambda x: lab_quantize(hist, x), 1, ar)

        n_centroids = ncenters if ncenters > ncolors else ncolors
        num_colors = ncolors

        km = KMeans(n_clusters=n_centroids, init='k-means++', n_init=5, random_state=0)
        clusters = km.fit_predict(ar)
        centroids = km.cluster_centers_

        alpha = 0.5
        dominants = np.zeros((n_centroids, 3))
        counts = np.zeros(n_centroids)

        for i, centroid in enumerate(centroids):
            c_i = np.reshape(centroid, (1, 3))
            members = ar[np.where(clusters == i)]
            members_weights = hist_weights[np.where(clusters == i)]
            distances = euclidean_distances(c_i, members)
            index = np.argmax(
                alpha * (
                    members_weights
                )
                + (1 - alpha) * (
                    (1. / distances)
                )
            )
            d_i = members[index]
            dominants[i, :] = d_i
            counts[i] = lab_quantize(hist, d_i)

        dominants = dominants[np.argsort(counts, axis=0)[::-1]][:num_colors]

        # rescale
        dominants[:, 0] *= 100 / 255.0
        dominants[:, 1] -= 128
        dominants[:, 2] -= 128

        colors = [lab2rgb([[x]])[0][0] for x in dominants]
        return colors

    @staticmethod
    def hsv_method(img, ncenters, ncolors):
        img = cv2.resize(np.array(img), (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        channels = [0, 1, 2]
        mask = None

        histSize = [HIST_BINS, HIST_BINS, HIST_BINS]
        hranges = [0, 180]
        sranges = [0, 256]
        vranges = [0, 256]
        ranges = [item for sublist in [hranges, sranges, vranges] for item in sublist]

        hist = cv2.calcHist([img], channels, mask, histSize, ranges)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # make img array for kmeans
        ar = img.reshape(-1, 3)

        def hsv_quantize(hist, x):
            h = np.floor(x[0] / 180.0 * HIST_BINS).astype(int)
            s = np.floor(x[1] / 256.0 * HIST_BINS).astype(int)
            v = np.floor(x[2] / 256.0 * HIST_BINS).astype(int)
            return hist[h, s, v]

        hist_weights = np.apply_along_axis(lambda x: hsv_quantize(hist, x), 1, ar)

        n_centroids = ncenters if ncenters > ncolors else ncolors
        num_colors = ncolors

        km = KMeans(n_clusters=n_centroids, init='k-means++', n_init=5, random_state=0)
        clusters = km.fit_predict(ar)
        centroids = km.cluster_centers_

        alpha = 0.5
        dominants = np.zeros((n_centroids, 3))
        counts = np.zeros(n_centroids)

        for i, centroid in enumerate(centroids):
            c_i = np.reshape(centroid, (1, 3))
            members = ar[np.where(clusters == i)]
            members_weights = hist_weights[np.where(clusters == i)]
            distances = euclidean_distances(c_i, members)
            index = np.argmax(
                alpha * (
                    members_weights
                )
                + (1 - alpha) * (
                    (1. / distances)
                )
            )
            d_i = members[index]
            dominants[i, :] = d_i
            counts[i] = hsv_quantize(hist, d_i)

        dominants = dominants[np.argsort(counts, axis=0)[::-1]][:num_colors]

        # rescale
        dominants[:, 0] /= 180.0
        dominants[:, 1] /= 256.0
        dominants[:, 2] /= 256.0

        colors = [hsv2rgb([[x]])[0][0] for x in dominants]
        return colors

    def on_get(self, req, resp):
        url = req.get_param('url') or ''
        ncolors = req.get_param_as_int('ncolors') or 4
        ncenters = req.get_param_as_int('ncenters') or ncolors
        format = req.get_param('format') or 'json'
        mode = req.get_param('mode') or 'lab'

        try:
            r = requests.get(url)
            r.raise_for_status()
            r.raw.decode_content = True
            with io.BytesIO(r.content) as f:
                img = Image.open(f)
                if mode == 'lab':
                    colors = self.lab_method(img, ncenters, ncolors)
                else:
                    colors = self.hsv_method(img, ncenters, ncolors)

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
        CorsMiddleware()
    ])

    dominant_colors = DominantColorsResource()
    app.add_route('/', dominant_colors)

    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()


if __name__ == '__main__':
    plac.call(main)
