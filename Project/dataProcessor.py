# -*- coding:utf-8 -*-
import json


def store(data):
    with open('data/neural_talk_prepare.json', 'w') as json_file:
        json_file.write(json.dumps(data))


def load():
    with open('data/caption_train_annotations_20170902.json', 'r') as f:
        data = json.load(f)
        return data


if __name__ == "__main__":
    source_data = load()
    print type(source_data)
    print len(source_data)
    format_data = []
    for item in source_data:
        data = {}
        data["file_path"] = item["image_id"]
        data["captions"] = item["caption"]
        if len(format_data) < 1000:
            format_data.append(data)

    store(format_data)
