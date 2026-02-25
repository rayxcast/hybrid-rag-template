from fastembed.rerank.cross_encoder import TextCrossEncoder

class Reranker:
    def __init__(self, model_name: str):
        self.model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, nodes, top_n: int = 10):
        # FastEmbed TextCrossEncoder .rerank() accepts query + list of texts directly
        scores = list(self.model.rerank(
            query=query,
            documents=[node.text for node in nodes]
        ))

        # scores is list[float], same order as input documents

        reranked = sorted(
            zip(nodes, scores),
            key=lambda x: x[1],
            reverse=True   # higher score = more relevant
        )

        return [node for node, _ in reranked[:top_n]]