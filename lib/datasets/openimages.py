from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from lib.datasets.imdb import imdb
import numpy as np
from lib.model.utils.config import cfg
import pickle
import pathlib
import pandas as pd
import scipy
import PIL
import tqdm
from progressbar import *

import pdb
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class openimages(imdb):
    def __init__(self, name, image_set):
        super().__init__(name)
        root = os.path.join(cfg.DATA_DIR, 'openimages')
        self.root = pathlib.Path(root)
        self.image_set = image_set.lower()
        self._image_ext = '.jpg'

        cached_data_path = os.path.join(self.root, 'cache', '{}_data_cache.pkl'.format(self.image_set))
        if os.path.exists(cached_data_path):
            with open(cached_data_path, 'rb') as f:
                print('Loading OpenImages data cache from file: {}'.format(cached_data_path))
                loaded_data = pickle.load(f)
        else:
            with open(cached_data_path, 'wb') as f:
                print('Caching OpenImages data on file: {}'.format(cached_data_path))
                loaded_data = self._read_data()
                pickle.dump(loaded_data, f)

        self.data, self.class_names, self.class_dict = loaded_data
        self._image_index = sorted(list(self.data.keys()))
        self._classes = self.class_names

        self.class_stat = None
        self.roidb_handler = self.gt_roidb

    def _read_data(self):
        annotation_file = f"{self.root}/sub-{self.image_set}-annotations-bbox.csv"
        annotations = pd.read_csv(annotation_file)
        class_names = ['__background__'] + sorted(list(annotations['ClassName'].unique()))
        images_ids = sorted(list(annotations['ImageID'].unique()))
        #images_id_to_progressive = {prog: i for i, prog in enumerate(images_ids)}
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = {}
        pbar = ProgressBar(widgets=[Percentage(), Bar(), AdaptiveETA()], maxval=len(images_ids)).start()
        i = 0
        for image_id, group in annotations.groupby("ImageID"):
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            size = self._get_size(image_id)
            labels = np.array([class_dict[name] for name in group["ClassName"]])
            data[image_id] = {
                'image_id': image_id,
                'image_size': size,
                'boxes': boxes,
                'labels': labels
            }
            i += 1
            pbar.update(i)
        return data, class_names, class_dict


    '''def _read_image(self, image_id):
        image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image'''

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i
        # return self._image_index[i]

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self.root, self.image_set,
                                  str(index) + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    '''def _image_split_path(self):
        if self._image_set == "minitrain":
          return os.path.join(self._data_path, 'train.txt')
        if self._image_set == "smalltrain":
          return os.path.join(self._data_path, 'train.txt')
        if self._image_set == "minival":
          return os.path.join(self._data_path, 'val.txt')
        if self._image_set == "smallval":
          return os.path.join(self._data_path, 'val.txt')
        else:
          return os.path.join(self._data_path, self._image_set+'.txt')

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        training_split_file = self._image_split_path()
        assert os.path.exists(training_split_file), \
                'Path does not exist: {}'.format(training_split_file)
        with open(training_split_file) as f:
          metadata = f.readlines()
          if self._image_set == "minitrain":
            metadata = metadata[:1000]
          elif self._image_set == "smalltrain":
            metadata = metadata[:20000]
          elif self._image_set == "minival":
            metadata = metadata[:100]
          elif self._image_set == "smallval":
            metadata = metadata[:2000]

        image_index = []
        id_to_dir = {}
        for line in metadata:
          im_file,ann_file = line.split()
          image_id = int(ann_file.split('/')[-1].split('.')[0])
          filename = self._annotation_path(image_id)
          if os.path.exists(filename):
              # Some images have no bboxes after object filtering, so there
              # is no xml annotation for these.
              tree = ET.parse(filename)
              for obj in tree.findall('object'):
                  obj_name = obj.find('name').text.lower().strip()
                  if obj_name in self._class_to_ind:
                      # We have to actually load and check these to make sure they have
                      # at least one object actually in vocab
                      image_index.append(image_id)
                      id_to_dir[image_id] = im_file.split('/')[0]
                      break
        return image_index, id_to_dir
    '''

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cached_data_path = os.path.join(self.root, 'cache', '{}_roidb_cache.pkl'.format(self.image_set))

        def do_cache():
            with open(cached_data_path, 'wb') as f:
                print('Caching OpenImages roidb data on file: {}'.format(cached_data_path))
                gt_roidb = {index: self._load_openimages_annotation(index)
                            for index in tqdm.tqdm(self._image_index)}
                pickle.dump(gt_roidb, f)
                return gt_roidb

        if os.path.exists(cached_data_path):
            with open(cached_data_path, 'rb') as f:
                print('Loading OpenImages roidb data cache from file: {}'.format(cached_data_path))
                gt_roidb = pickle.load(f)
                if len(gt_roidb) < len(self._image_index):
                    raise ValueError('Flipped images not yet supported!')
                    # the loaded db is smaller than the current one, overwrite!
                    print('Upgrading roidb (loaded len: {}; current len: {})'.format(len(gt_roidb), len(self._image_index)))
                    gt_roidb = do_cache()
                elif len(gt_roidb) > len(self._image_index):
                    # the loaded db is bigger than the current one... the loaded db already contains flipped images
                    print('Loading also flipped images')
                    raise ValueError('Flipped images not yet supported!')
        else:
            gt_roidb = do_cache()

        return gt_roidb

    def _get_size(self, index):
        return PIL.Image.open(self.image_path_from_index(index)).size

    def _load_openimages_annotation(self, index):
        boxes = self.data[index]['boxes']
        size = self.data[index]['image_size']
        # boxes are needed in un-normalized fashion
        boxes = boxes * np.tile(size, 2)

        num_objs = boxes.shape[0]
        num_classes = len(self.class_names)
        gt_classes = self.data[index]['labels']

        gt_overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
        for idx, cls in enumerate(gt_classes):
            gt_overlaps[idx, cls] = 1.0
        gt_overlaps = scipy.sparse.csr_matrix(gt_overlaps)

        return {'boxes': boxes,
                'width': size[0],
                'height': size[1],
                'gt_classes': gt_classes,
                'gt_overlaps': gt_overlaps,
                'flipped': False}

    def append_flipped_images(self):
        # TODO: as of now, not working

        num_images = self.num_images
        for i in tqdm.tqdm(self._image_index):
            annotation = self._load_openimages_annotation(i)
            boxes = annotation['boxes'].copy()
            size = self.data[i]['image_size']
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = size[0] - oldx2 - 1
            boxes[:, 2] = size[0] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()

            gt_classes = annotation['gt_classes']
            gt_overlaps = annotation['gt_overlaps']

            entry = {'width': size[0],
                     'height': size[1],
                     'boxes': boxes,
                     'gt_classes': gt_classes,
                     'gt_overlaps': gt_overlaps,
                     'flipped': True
                     }

            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    '''def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(self.classes, all_boxes, output_dir)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_vg_results_file_template(output_dir).format(cls)
                os.remove(filename)

    def evaluate_attributes(self, all_boxes, output_dir):
        self._write_voc_results_file(self.attributes, all_boxes, output_dir)
        self._do_python_eval(output_dir, eval_attributes = True)
        if self.config['cleanup']:
            for cls in self._attributes:
                if cls == '__no_attribute__':
                    continue
                filename = self._get_vg_results_file_template(output_dir).format(cls)
                os.remove(filename)

    def _get_vg_results_file_template(self, output_dir):
        filename = 'detections_' + self._image_set + '_{:s}.txt'
        path = os.path.join(output_dir, filename)
        return path

    def _write_voc_results_file(self, classes, all_boxes, output_dir):
        for cls_ind, cls in enumerate(classes):
            if cls == '__background__':
                continue
            print('Writing "{}" vg results file'.format(cls))
            filename = self._get_vg_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index), dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))


    def _do_python_eval(self, output_dir, pickle=True, eval_attributes = False):
        # We re-use parts of the pascal voc python code for visual genome
        aps = []
        nposs = []
        thresh = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # Load ground truth
        gt_roidb = self.gt_roidb()
        if eval_attributes:
            classes = self._attributes
        else:
            classes = self._classes
        for i, cls in enumerate(classes):
            if cls == '__background__' or cls == '__no_attribute__':
                continue
            filename = self._get_vg_results_file_template(output_dir).format(cls)
            rec, prec, ap, scores, npos = vg_eval(
                filename, gt_roidb, self.image_index, i, ovthresh=0.5,
                use_07_metric=use_07_metric, eval_attributes=eval_attributes)

            # Determine per class detection thresholds that maximise f score
            if npos > 1:
                f = np.nan_to_num((prec*rec)/(prec+rec))
                thresh += [scores[np.argmax(f)]]
            else:
                thresh += [0]
            aps += [ap]
            nposs += [float(npos)]
            print('AP for {} = {:.4f} (npos={:,})'.format(cls, ap, npos))
            if pickle:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap,
                        'scores': scores, 'npos':npos}, f)

        # Set thresh to mean for classes with poor results
        thresh = np.array(thresh)
        avg_thresh = np.mean(thresh[thresh!=0])
        thresh[thresh==0] = avg_thresh
        if eval_attributes:
            filename = 'attribute_thresholds_' + self._image_set + '.txt'
        else:
            filename = 'object_thresholds_' + self._image_set + '.txt'
        path = os.path.join(output_dir, filename)
        with open(path, 'wt') as f:
            for i, cls in enumerate(classes[1:]):
                f.write('{:s} {:.3f}\n'.format(cls, thresh[i]))

        weights = np.array(nposs)
        weights /= weights.sum()
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('Weighted Mean AP = {:.4f}'.format(np.average(aps, weights=weights)))
        print('Mean Detection Threshold = {:.3f}'.format(avg_thresh))
        print('~~~~~~~~')
        print('Results:')
        for ap,npos in zip(aps,nposs):
            print('{:.3f}\t{:.3f}'.format(ap,npos))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** PASCAL VOC Python eval code.')
        print('--------------------------------------------------------------')

'''
if __name__ == '__main__':
    d = openimages('test')
    res = d.roidb
    pdb.set_trace()
    # from IPython import embed; embed()
