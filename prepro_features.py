from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import sys
import os
import csv
import argparse
import numpy as np


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


def main(params):
    output_dir = params.output_dir + '/features'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    count = 0
    with open(params.input_file, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))
            np.savez_compressed(os.path.join(output_dir, str(item['image_id'])),
                                feat=item['features'], boxes=item['boxes'])
            count = count + 1
            if count % 1000 == 0:
                print('Processing : %d iter...' % count)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='./data/coco/trainval_resnet101_faster_rcnn_genome_36.tsv',
                        help='input file')
    parser.add_argument('--output_dir', default='data', help='output np files')
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
