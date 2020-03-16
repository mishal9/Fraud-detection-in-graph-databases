from py2neo import Graph
import pandas as pd

graph = Graph(password="semanticweb", bolt_port=7687, http_port=7474)

df = pd.read_csv("./data/preprocessed.csv")


# Use cypher query to get data from all the nodes 
query = """
MATCH (p:Placeholder)
RETURN p.id AS id, p.degree AS degree, p.pagerank as pagerank, p.community AS community 
"""

data = graph.run(query)
valueDict = {}
for d in data:
    valueDict[d['id']] = {'degree': d['degree'], 'pagerank': d['pagerank'],}

print(valueDict)



