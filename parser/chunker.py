from langchain_text_splitters import RecursiveCharacterTextSplitter


class SemanticChunker:
    """
    Splits large source/configuration files into manageable chunks
    while preserving useful context between chunks.
    """

    def __init__(self, chunk_size: int = 6000, chunk_overlap: int = 800):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\nclass ",
                "\ndef ",
                "\nfunction ",
                "\nmodule ",
                "\n[",
                "\n",
                " ",
                "",
            ],
        )

    def split(self, text: str) -> list[str]:
        """Split source text into overlapping chunks."""
        if not text or not text.strip():
            return []

        return self.splitter.split_text(text)

    def add_context(self, chunks: list[str]) -> list[str]:
        """
        Add chunk position information so downstream agents
        know where each piece came from.
        """
        total = len(chunks)

        return [
            f"""[CHUNK {index + 1}/{total}]

{chunk}

[END CHUNK {index + 1}]
"""
            for index, chunk in enumerate(chunks)
        ]


if __name__ == "__main__":
    sample = """
    This is a sample enterprise configuration.

    database:
      host: localhost
      port: 5432

    application:
      name: example
      workers: 4
    """ * 20

    chunker = SemanticChunker()

    chunks = chunker.split(sample)
    chunks = chunker.add_context(chunks)

    print(f"Created {len(chunks)} chunks")

    for chunk in chunks[:2]:
        print(chunk)