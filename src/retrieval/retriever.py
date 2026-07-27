"""Stage 2: page-level retrieval within a known document.

Page+header + BGE-small, top-10: 58.3% hit@5 / 55.0% recall@5 on the
22-document / 60-question stratified eval set. This is the pipeline's
final retrieval configuration -- see configs/pipeline_config.yaml.
"""


def retrieve_pages(query_embedding, index, pages: list[dict], doc_id: str,
                    k: int = 10, search_depth: int = 400) -> list[int]:
    """Return the top-k page numbers within doc_id, ranked by similarity.

    search_depth controls how many global candidates are pulled before
    filtering to doc_id -- must be >= the number of pages in the largest
    document in the corpus, or true top-k within that document may be missed.
    """
    _, idx = index.search(query_embedding, search_depth)
    matches = [pages[i]["page_num"] for i in idx[0] if pages[i]["doc_id"] == doc_id]
    return matches[:k]
