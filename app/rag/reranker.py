from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, nodes, top_n=10):
        pairs = [(query, node.text) for node in nodes]
        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(nodes, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [node for node, _ in reranked[:top_n]]