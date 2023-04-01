from neo4j import GraphDatabase

class ArtGraph:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()

    def close(self):
        self._driver.close()

    def get_all_artworks(self):
        with self._driver.session(database="artgraph") as session:
            return session.execute_read(self._get_all_artworks)
        
    @staticmethod
    def _get_all_artworks(tx):
        query = """
        MATCH (artwork:Artwork)
        RETURN artwork
        """
        result = tx.run(query)
        return [record["artwork"] for record in result]

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
    
    def get_all_artists(self):
        with self._driver.session(database="artgraph") as session:
            return session.execute_read(self._get_all_artists)
        
    @staticmethod
    def _get_all_artists(tx):
        query = """
        MATCH (artist:Artist)
        RETURN artist
        """
        result = tx.run(query)
        return [record["artist"] for record in result]
    
    def get_all_artworks_with_artists(self):
        with self._driver.session(database="artgraph") as session:
            return session.execute_read(self._get_all_artworks_with_artists)
        
    @staticmethod
    def _get_all_artworks_with_artists(tx):
        query = """
        MATCH (artwork:Artwork)
        -[:createdBy]->(artist:Artist)
        RETURN artwork, artist
        """
        result = tx.run(query)
        return [(record["artwork"], record["artist"]) for record in result]