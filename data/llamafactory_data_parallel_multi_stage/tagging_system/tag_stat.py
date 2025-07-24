import json
from collections import Counter


def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def main():
    path = r'/data1/gsn/NewCoEditorEval/data/show_json/test_llm_tag.json'
    data = read_json(path)

    counter = Counter()
    for obj in data:
        counter.update(obj['tags'])

    sorted_result = []
    n = len(data)
    for e in counter.most_common():
        sorted_result.append((*e, e[1] / n))
    print(sorted_result)

if __name__ == '__main__':
    main()
