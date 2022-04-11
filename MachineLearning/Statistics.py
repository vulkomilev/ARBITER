import numpy as np


class Statistics(object):

    def __init__(self):
        self.histogram_db = {}

    def train(self, images):
        for image in images:
            local_image_id = image['img_id']
            if local_image_id not in self.histogram_db.keys():
                self.histogram_db[local_image_id] = []
            self.histogram_db[local_image_id].append(np.histogram(np.array(image['img'])))

    def predict(self, image):
        min_score = -1
        best_index = None
        local_img = image['img']
        local_target_hist = np.histogram(np.array(local_img))
        for key in self.histogram_db.keys():
            local_score = 0
            for hist in self.histogram_db[key]:
                local_score += np.linalg.norm(local_target_hist[0] - hist[0])
                local_score += np.linalg.norm(local_target_hist[1] - hist[1])
            if min_score == -1 or local_score < min_score:
                min_score = local_score
                best_index = key
        return best_index
