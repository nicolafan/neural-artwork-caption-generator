from neo4j import GraphDatabase

class ArtGraph:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()

    def test_connection(self):
        with self._driver.session() as session:
            return session.execute_read(self._test_connection)

    @staticmethod
    def _test_connection(tx):
        query = "RETURN 'Connected to Neo4j' as message"
        result = tx.run(query)
        return result.single()["message"]

    def close(self):
        self._driver.close()

    def get_artworks_by_name(self, name):
        with self._driver.session(database="artgraph") as session:
            return session.execute_read(self._get_artworks_by_name, name)

    @staticmethod
    def _get_artworks_by_name(tx, name):
        query = """
        MATCH (artwork:Artwork {name: $name})
        RETURN artwork
        """
        result = tx.run(query, name=name)
        return [record["artwork"] for record in result]