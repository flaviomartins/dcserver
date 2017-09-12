import io
import json
import requests
import logging
from wsgiref import simple_server

import falcon

import plac
import numpy as np
from PIL import Image
import cv2
import matplotlib

from sklearn import preprocessing
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


class DominantColorsResource(object):
    def __init__(self):
        pass

    def on_get(self, req, resp):
        url = req.get_param('url') or ''
        ncolors = req.get_param_as_int('ncolors') or 4
        format = req.get_param('format') or 'json'

        try:
            r = requests.get(url)
            r.raise_for_status()
            r.raw.decode_content = True
            with io.BytesIO(r.content) as f:
                with Image.open(f) as img:
                    img = cv2.resize(np.array(img), (224, 224), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            channels = [0, 1, 2]
            mask = None

            histSize = [HIST_BINS, HIST_BINS, HIST_BINS]
            hranges = [0, 180]
            sranges = [0, 256]
            vranges = [0, 256]
            ranges = [item for sublist in [hranges, sranges, vranges] for item in sublist]

            hist = cv2.calcHist([img], channels, mask, histSize, ranges)

            # normalize hist to 0-1 range
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            norm_hist = min_max_scaler.fit_transform(hist.ravel().reshape(-1, 1))
            norm_hist = np.reshape(norm_hist, (HIST_BINS, HIST_BINS, HIST_BINS))

            # make img array for kmeans
            ar = img.reshape(-1, 3)

            def quantize(hist, x):
                h = np.floor(x[0] / 180.0 * HIST_BINS).astype(int)
                s = np.floor(x[1] / 256.0 * HIST_BINS).astype(int)
                v = np.floor(x[2] / 256.0 * HIST_BINS).astype(int)
                return hist[h, s, v]

            hist_weights = np.apply_along_axis(lambda x: quantize(norm_hist, x), 1, ar)

            n_centroids = ncolors
            num_colors = ncolors

            km = KMeans(n_centroids, random_state=0)
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
                index = np.argmax(alpha * members_weights + (1 - alpha) * (
                    (1. / distances) + members[:, 1] / 255.0 + members[:, 2] / 255.0))
                d_i = members[index]
                dominants[i, :] = d_i
                counts[i] = quantize(hist, d_i)

            dominants = dominants[np.argsort(counts, axis=0)[::-1]][:num_colors]
            # print dominants
            # print counts[np.argsort(counts, axis=0)[::-1]].astype(int)

            colors = [matplotlib.colors.hsv_to_rgb([x[0] / 180.0, x[1] / 256.0, x[2] / 256.0]) for x in dominants]
            colors = [matplotlib.colors.to_hex([x[0], x[1], x[2]]) for x in colors]

            if format == 'json':
                result = json.dumps({'colors': np.array(colors).tolist()})
                resp.content_type = falcon.MEDIA_JSON
            else:
                result = '<html>\n<body>\n'
                for x in colors:
                    result += '<div style="font-family: monospace; background: ' + x + ';'\
                              'padding: 1em">' + x +\
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
