from flask import Flask, request
from flask_cors import cross_origin, CORS
from collections import defaultdict
import json



api = Flask(__name__)
cors = CORS(api, resources={r"/*": {"origins": "*"}})



entity2text = {}
text2entity = {}
rel2text = {}
text2rel = {}
head_list = defaultdict(list)
with open("./dataset/WN18RR/entity2text.txt") as f:
    for l in f.readlines():
        e, text = l.strip().split('\t')
        text = text.split(",")[0]
        entity2text[int(e)] = text
        text2entity[text] = int(e)

with open('./dataset/WN18RR/relation2text.txt') as f:
    for l in f.readlines():
        e, text = l.strip().split('\t')
        text = text.split(",")[0]
        rel2text[int(e)] = text
        text2rel[text] = int(e)

with open("./dataset/WN18RR/test.tsv") as f:
    for l in f.readlines():
        h,r,t = list(map(str, l.strip().split("\t")))
        r = int(r)
        h = int(h)
        if r not in head_list[h]:
            head_list[h].append(r)

with open("./dataset/WN18RR/result.json") as f:
    result = json.load(f)
        


@api.route("/get_head_entity", methods=["POST"])
def get_head_entity():
    res =  [dict(entity_id=_, entity_name=entity2text[_]) for _ in head_list.keys()]
    return res


@api.route("/get_relation_by_head_entity", methods=["POST"])
@cross_origin()
def get_relation_by_head_entity():
    try:
        head_entity_id = int(request.get_json()['head_entity_id'])
    except:
        head_entity_id = 3515
    res = [dict(relation_id=_, relation_name=rel2text[_]) for _ in head_list[head_entity_id]]
    return res

@api.route("/get_tail_entity_with_prob", methods=["POST"])
@cross_origin()
def get_tail_entity_with_prob():
    try:
        inputs = request.get_json()
    except:
        inputs = dict(head_entity_id=34590, relation_id=0)
    head_entity_id = inputs['head_entity_id']
    rel_id = inputs['relation_id']
    id_ = "_".join([str(head_entity_id), str(rel_id)])
    res = []
    for prob, entity_id in zip(result[id_][0], result[id_][1]):
        res.append([entity2text[entity_id], round(prob,3)])
    return res

if __name__ == "__main__":
    # api.run(host='0.0.0.0', port=9001)
    print(head_list)
    api.run(port=9000, debug=True)