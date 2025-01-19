Config = {
    "data_path":f"./data/data.json",
    "schema_path":f"./data/schema.json",
    "train_path":f"./data/train.json",
    "valid_path":f"./data/valid.json",
    "vocab_path":f"./chars.txt",
    "max_length":10,
    "pool_type":"avg",
    "hidden_size":128,
    "positive_ratio":0.5,
    "epoch_data_size":100,
    "batch_size":50000,
    "learning_rate":0.05,
    "epoch":100,
    "optimizer":"adam"
}