# -*- coding:utf-8 -*-
import json
import argparse


def store(data):
    with open('data/neural_talk_prepare.json', 'w') as json_file:
        json_file.write(json.dumps(data))


def load():
    with open('data/caption_train_annotations_20170902.json', 'r') as f:
        data = json.load(f)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input json
    parser.add_argument('--count', required=False,
                        help='input json process count')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    count = int(params['count'])
    if count < 0:
        count = 100

    source_data = load()
    print type(source_data)
    print len(source_data)
    format_data = []

    for item in source_data:
        data = {}
        data["file_path"] = item["image_id"]
        data["captions"] = item["caption"]
        format_data.append(data)
        if len(format_data) > count:
            break

    store(format_data)
    print "data process successfully"
    print 'processed: ' + str(count)
