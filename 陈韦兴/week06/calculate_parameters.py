def calculate_parameters(text_length, vocabulary_size, hidden_size, transformer_layers_number):
    # embedding层大小
    # Token Embedding层
    token_embedding_size = vocabulary_size * hidden_size
    # Segment Embedding层
    segment_embedding_size = 2 * hidden_size
    # Position Embedding层
    position_embedding_size = text_length * hidden_size
    # embedding层总可训练参数
    all_embedding_size = token_embedding_size + segment_embedding_size + position_embedding_size

    # 一层Transformer层可训练参数
    # 两层layerNormalization层可训练参数
    layer_norm_size = 2 * 2 * hidden_size
    # self-Attention层大小
    # QKW层可训练参数
    qkw_weight_size = 3 * hidden_size * hidden_size + 3 * hidden_size
    # self-Attention层外面的线性层
    linear_weight_size1 = hidden_size * hidden_size + hidden_size

    # feed forward层参数
    # 有两个线性层 第一层参数翻四倍 第二层参数减少四倍
    feed_forward_size1 = 4 * hidden_size * hidden_size + 4 * hidden_size
    feed_forward_size2 = 4 * hidden_size * hidden_size + hidden_size

    # 单独一层的transformer的参数总量为
    one_transformer_size = layer_norm_size + qkw_weight_size + linear_weight_size1 + feed_forward_size1 + feed_forward_size2

    # n层的总参数为
    all_transformer_size = transformer_layers_number * one_transformer_size + all_embedding_size

    return all_transformer_size

if __name__ == '__main__':
    text_length = 512 # 文本长度
    vocabulary_size = 1000 # 词典长度
    hidden_size = 768 # 隐藏层参数两
    transformer_layers_number = 12 # transformer层数
    parameter_number = calculate_parameters(text_length, vocabulary_size, hidden_size, transformer_layers_number)

    print(parameter_number)
