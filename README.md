# Fraud-detection-in-graph-databases
Fraud detection using Neo4j and deep link analysis on graphs

## Getting Started 
This project consists of two main parts
1. Building a graph model using Neo4j
2. Creating supervised learning model and using features extracted for the graph to improve its accuracy  

To create this representation using the Neo4j Desktop, create a new database and make sure to install the GraphAlgorithm and APOC packages. 

When the packages has been installed, put a copy of the "bs140513_032310.csv" file in the database input folder by clicking "open folder" button in the Neo4j Desktop app. 

When the database is up and running, use the Neo4j Browser to execute the following query to import the data into Neo4j.
~~~~
CREATE CONSTRAINT ON (c:Customer) ASSERT c.id IS UNIQUE;
CREATE CONSTRAINT ON (b:Bank) ASSERT b.id IS UNIQUE;

:auto USING PERIODIC COMMIT 
LOAD CSV WITH HEADERS FROM
'file:///preprocessed.csv' AS line
WITH line,
MERGE (customer:Customer {id: line.customer})
MERGE (bank:Bank {id: line.merchant})
CREATE (transaction:Transaction {amount: line.amount, fraud: line.fraud, category: line.category})-[:WITH]->(bank)
CREATE (customer)-[:MAKE]->(transaction);
~~~~

