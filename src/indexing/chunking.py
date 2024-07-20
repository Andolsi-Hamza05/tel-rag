import re
from typing import List
from langchain.schema import Document


class TextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int, word_split: bool = False):
        """
        Initializes the TextSplitter with the desired chunk size, overlap, and word-split option.

        Args:
            chunk_size (int): The size of each chunk in characters.
            chunk_overlap (int): The number of characters of overlap between consecutive chunks.
            word_split (bool, optional): If True, ensures that chunks end at word boundaries,
            Defaults to False.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.word_split = word_split
        self.separators_pattern = re.compile(r'[\s,.\-!?\[\]\(\){}":;<>]+')

    def custom_text_splitter(self, text: str) -> List[str]:
        """
        Splits a given text into chunks of a specified size with a defined overlap between them.

        Args:
            text (str): The text to be split into chunks.

        Returns:
            List[str]: A list containing the text chunks.
        """
        chunks = []
        start = 0
        while start < len(text) - self.chunk_overlap:
            end = min(start + self.chunk_size, len(text))
            if self.word_split:
                match = self.separators_pattern.search(text, end)
                if match:
                    end = match.end()
            if end == start:
                end = start + 1
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            if self.word_split:
                match = self.separators_pattern.search(text, start-1)
                if match:
                    start = match.start() + 1
            if start < 0:
                start = 0
        return chunks


class DocumentChunker:
    def __init__(self, text_splitter: TextSplitter):
        """
        Initializes the DocumentChunker with a TextSplitter instance.

        Args:
            text_splitter (TextSplitter): An instance of the TextSplitter class.
        """
        self.text_splitter = text_splitter

    def chunk_doc(self, doc: Document) -> List[Document]:
        """
        Chunks the content of a Document into a list of Documents.

        Args:
            doc (Document): The Document object to be chunked.

        Returns:
            List[LangchainDocument]: A list of Documents with chunked text and source metadata.
        """
        chunks = self.text_splitter.custom_text_splitter(doc.page_content)
        chunked_docs = [Document(page_content=chunk,
                                 metadata={'source': doc.metadata.get('source', 'unknown')})
                        for chunk in chunks]
        return chunked_docs
