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
    
    def get_all_artworks_with_artist(self):
        with self._driver.session(database="artgraph") as session:
            return session.execute_read(self._get_all_artworks_with_artist)
        
    @staticmethod
    def _get_all_artworks_with_artist(tx):
        query = """
        MATCH (artwork:Artwork)
        -[:createdBy]->(artist:Artist)
        RETURN artwork, artist
        """
        result = tx.run(query)
        return {record["artwork"]["name"]: record["artist"] for record in result}
    
    @staticmethod
    def _get_all_artworks_with_style(tx):
        query = """
        MATCH (artwork:Artwork)
        -[:hasStyle]->(style:Style)
        RETURN artwork, style
        """
        result = tx.run(query)
        return {record["artwork"]["name"]: record["style"] for record in result}
    
    @staticmethod
    def _get_all_artworks_with_genre(tx):
        query = """
        MATCH (artwork:Artwork)
        -[:hasGenre]->(genre:Genre)
        RETURN artwork, genre
        """
        result = tx.run(query)
        return {record["artwork"]["name"]: record["genre"] for record in result}
    
    @staticmethod
    def _get_all_artworks_with_tags(tx):
        query = """
        MATCH (artwork:Artwork)
        -[:about]->(tag:Tag)
        RETURN artwork, tag
        """
        result = tx.run(query)
        result_dict = {}

        for record in result:
            if not record["artwork"]["name"] in result_dict:
                result_dict[record["artwork"]["name"]] = []
            result_dict[record["artwork"]["name"]].append(record["tag"])
        return result_dict
    
    @staticmethod
    def _get_all_artworks_with_media(tx):
        query = """
        MATCH (artwork:Artwork)
        -[:madeOf]->(media:Media)
        RETURN artwork, media
        """
        result = tx.run(query)
        result_dict = {}

        for record in result:
            if not record["artwork"]["name"] in result_dict:
                result_dict[record["artwork"]["name"]] = []
            result_dict[record["artwork"]["name"]].append(record["media"])
        return result_dict
    
    def get_all_artworks_for_dataframe(self):
        with self._driver.session(database="artgraph") as session:
            return session.execute_read(self._get_all_artworks_for_dataframe)
        
    @staticmethod
    def _get_all_artworks_for_dataframe(tx):
        artworks_with_artist = ArtGraph._get_all_artworks_with_artist(tx)
        artworks_with_style = ArtGraph._get_all_artworks_with_style(tx)
        artworks_with_genre = ArtGraph._get_all_artworks_with_genre(tx)
        artworks_with_tags = ArtGraph._get_all_artworks_with_tags(tx)
        artworks_with_media = ArtGraph._get_all_artworks_with_media(tx)

        artworks_with_all = {}
        for artwork in artworks_with_artist.keys():
            if not artwork in artworks_with_tags:
                artworks_with_tags[artwork] = []
            if not artwork in artworks_with_media:
                artworks_with_media[artwork] = []
            all_data = [
                artworks_with_artist[artwork],
                artworks_with_style[artwork],
                artworks_with_genre[artwork],
                artworks_with_tags[artwork],
                artworks_with_media[artwork]
            ]
            artworks_with_all[artwork] = all_data

        return artworks_with_all

    def make_dataframe_dict(self):
        artworks_data = self.get_all_artworks_for_dataframe()
        d = {
            "name": [],
            "artist": [],
            "style": [],
            "genre": [],
            "tags": [],
            "media": []
        }

        for artwork_name, data in artworks_data.items():
            d["name"].append(artwork_name)
            d["artist"].append(data[0]["name"])
            d["style"].append(data[1]["name"])
            d["genre"].append(data[2]["name"])
            d["tags"].append(", ".join([i["name"] for i in data[3]])),
            d["media"].append(", ".join([i["name"] for i in data[4]]))

        return d            