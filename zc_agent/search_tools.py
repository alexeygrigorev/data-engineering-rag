from typing import List, Any

class SearchTool:
    def __init__(self, index):
        self.index = index

    def search(self, query: str) -> List[Any]:
        """
        Perform a text-based search on the FAQ index.

        Args:
            query (str): The search query string.

        Returns:
            List[Any]: A list of up to 5 search results returned by the FAQ index.
        """
        return self.index.search(query, num_results=5)