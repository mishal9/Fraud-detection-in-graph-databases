from py2neo import Graph
import pandas as pd
import json

graph = Graph(password="semanticweb", bolt_port=7687, http_port=7474)

df = pd.read_csv("./data/preprocessed.csv")

query = """
MATCH (p:Placeholder)
RETURN p.id AS id, p.degree AS degree, p.pagerank as pagerank, p.community AS community 
"""

data = graph.run(query).data()

#print(data)
with open('./data/graphEnhanced.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

valueDict = {}
for d in data:
    valueDict[d['id']] = {'degree': d['degree'], 'pagerank': d['pagerank'], 'community': d['community']}

#print(valueDict)
