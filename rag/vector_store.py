import re
from typing import Any


class CodeVectorStore:
    """
    Lightweight RAG store for low-memory deployments.

    Uses token-overlap scoring instead of SentenceTransformer + ChromaDB.
    This keeps the Agentic TDD pipeline compatible with Render Free's
    limited memory environment.
    """

    def __init__(
        self,
        persist_directory: str = "output/chroma_db",
        model_name: str = "lightweight",
    ):
        self.persist_directory = persist_directory
        self.model_name = model_name

        self.documents = []
        self.metadatas = []

        print("Initializing lightweight retrieval store...")
        print("Embedding model disabled for low-memory deployment.")

    # ============================================================
    # TOKENIZATION
    # ============================================================

    @staticmethod
    def _tokenize(text: str):
        text = str(text).lower()

        # Keep programming identifiers useful.
        tokens = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+",
            text,
        )

        # Remove extremely common/noisy words.
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "are",
            "was",
            "were",
            "have",
            "has",
            "had",
            "you",
            "your",
            "into",
            "using",
            "used",
            "use",
            "can",
            "will",
            "then",
            "than",
            "not",
            "but",
            "its",
            "they",
            "their",
            "our",
            "about",
            "also",
            "only",
            "one",
            "all",
        }

        return {
            token
            for token in tokens
            if len(token) >= 2
            and token not in stop_words
        }

    # ============================================================
    # ADD CHUNKS
    # ============================================================

    def add_chunks(self, chunks: list[Any]):
        if not chunks:
            print("No chunks to store.")
            return

        self.documents = []
        self.metadatas = []

        for index, chunk in enumerate(chunks):

            if isinstance(chunk, str):
                text = chunk
                metadata = {
                    "chunk_index": index
                }

            elif isinstance(chunk, dict):
                text = str(
                    chunk.get(
                        "text",
                        chunk.get(
                            "content",
                            "",
                        ),
                    )
                )

                metadata = {
                    "chunk_index": index
                }

                for key in [
                    "file",
                    "filename",
                    "path",
                    "source",
                    "start_line",
                    "end_line",
                ]:
                    if key in chunk:
                        value = chunk[key]

                        if value is not None:
                            metadata[key] = str(value)

            else:
                text = str(chunk)

                metadata = {
                    "chunk_index": index
                }

            text = text.strip()

            if not text:
                continue

            self.documents.append(text)
            self.metadatas.append(metadata)

        print(
            f"Stored {len(self.documents)} chunks "
            "in lightweight retrieval store."
        )

    # ============================================================
    # SCORE
    # ============================================================

    def _score(self, query: str, document: str) -> float:

        query_tokens = self._tokenize(query)
        document_tokens = self._tokenize(document)

        if not query_tokens or not document_tokens:
            return 0.0

        overlap = query_tokens.intersection(document_tokens)

        if not overlap:
            return 0.0

        # Basic overlap score.
        score = len(overlap) / len(query_tokens)

        # Give a small bonus when exact query terms occur.
        lower_document = document.lower()

        for token in overlap:
            if token in lower_document:
                score += 0.01

        return score

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = 3,
    ):
        query = str(query).strip()

        if not query or not self.documents:
            return []

        scored = []

        for index, document in enumerate(self.documents):

            score = self._score(
                query,
                document,
            )

            if score > 0:
                scored.append(
                    (
                        score,
                        index,
                        document,
                    )
                )

        # Highest score first.
        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        results = [
            item[2]
            for item in scored[:top_k]
        ]

        # If no lexical match exists, return the first chunks
        # rather than returning an empty context.
        if not results:
            results = self.documents[:top_k]

        return results

    # ============================================================
    # SEARCH WITH METADATA
    # ============================================================

    def search_with_metadata(
        self,
        query: str,
        top_k: int = 3,
    ):
        query = str(query).strip()

        if not query or not self.documents:
            return []

        scored = []

        for index, document in enumerate(self.documents):

            score = self._score(
                query,
                document,
            )

            scored.append(
                (
                    score,
                    index,
                    document,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        output = []

        for score, index, document in scored[:top_k]:

            output.append(
                {
                    "text": document,
                    "metadata": self.metadatas[index],
                    "distance": 1.0 - score,
                }
            )

        return output

    # ============================================================
    # CLEAR STORE
    # ============================================================

    def clear(self):
        self.documents = []
        self.metadatas = []

        print(
            "Lightweight retrieval store cleared."
        )


# ================================================================
# DIRECT TEST
# ================================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("LIGHTWEIGHT CODE VECTOR STORE TEST")
    print("=" * 60)
    print()

    store = CodeVectorStore()

    test_chunks = [
        """
        FILE: lib/main.dart

        SonicSplit is a Flutter application
        for processing audio files.
        """,
        """
        FILE: pubspec.yaml

        The project uses Flutter packages
        for file selection and audio playback.
        """,
        """
        FILE: README.md

        SonicSplit allows users to split audio
        and play individual stems.
        """,
    ]

    store.add_chunks(test_chunks)

    results = store.search(
        "audio splitting playback",
        top_k=3,
    )

    print()
    print("Search results:")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(f"[{index}]")
        print(result[:500])

    print()
    print(
        "Lightweight vector store test completed."
    )