from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def ingest_data(self, doc_name, df, key_column):
        """
        Ingests a DataFrame into Neo4j.
        - Creates a Document node.
        - Creates Row nodes for each row.
        - Creates KeyNode for the shared key.
        - Links Row to Document and Row to KeyNode.
        """
        with self.driver.session() as session:
            # Create Document node
            session.run("MERGE (d:Document {name: $name})", name=doc_name)

            # Ingest rows in batches
            rows = df.to_dict('records')
            query = """
            UNWIND $rows AS row
            MERGE (d:Document {name: $doc_name})
            CREATE (r:Row)
            SET r += row
            MERGE (d)-[:HAS_ROW]->(r)
            WITH r, row
            WHERE row[$key_col] IS NOT NULL
            MERGE (k:KeyNode {value: toString(row[$key_col])})
            MERGE (r)-[:MATCHES_KEY]->(k)
            """
            session.run(query, rows=rows, doc_name=doc_name, key_col=key_column)

    def link_matching_keys(self):
        """
        Ensures relationships between rows that share the same key are visible.
        Actually, the KeyNode already acts as the bridge. 
        (RowA)-[:MATCHES_KEY]->(KeyNode)<-[:MATCHES_KEY]-(RowB)
        """
        pass
