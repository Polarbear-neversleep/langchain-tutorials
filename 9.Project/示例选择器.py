# example_selector.将示例转换为向量,根据余弦相似度进行匹配选择
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embedding_model,
    Chroma,
    k=1, # 选出匹配度最高的k个示例
)