import tensorflow as tf
import numpy as np

def get_tensor_and_hashtable(dim, dtype, size, process_rank):
    """
    生成 key tensor 和对应的 HashTable。

    参数:
    - dim: Tensor 维度
    - dtype: Tensor 的数据类型
    - rank: 每个进程中 tensor 的 rank
    - size: 总的 Horovod 进程数量
    - process_rank: 当前进程的 rank

    返回:
    - key_tensor: 当前进程生成的 key tensor
    - hash_table: 包含 key 和随机 value 的哈希表
    """
    # 计算每个进程的 key 范围
    dtype_range = tf.dtypes.as_dtype(dtype).max - tf.dtypes.as_dtype(dtype).min
    keys_per_process = dtype_range // size
    start_key = tf.dtypes.as_dtype(dtype).min + process_rank * keys_per_process
    end_key = start_key + keys_per_process

    # 在该进程的范围内生成随机 keys
    num_keys = 10  # 每个进程中生成的 key 数量（可以调整）
    key_vals = np.random.randint(
        low=start_key,
        high=min(end_key, tf.dtypes.as_dtype(dtype).max),
        size=num_keys,
        dtype=dtype
    )

    # 创建 key tensor
    key_tensor = tf.convert_to_tensor(key_vals, dtype=dtype)
    for _ in range(dim - 1):
        key_tensor = tf.expand_dims(key_tensor, axis=1)
        key_tensor = tf.concat([key_tensor, key_tensor], axis=1)

    # 创建 HashTable 并插入 key 和随机 value
    values = np.random.randint(0, 100, size=num_keys)  # 对应的随机 values
    hash_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.convert_to_tensor(key_vals, dtype=dtype),
            values=tf.convert_to_tensor(values, dtype=dtype)
        ),
        default_value=tf.constant(-1, dtype=dtype)
    )

    return key_tensor, hash_table

dim = 1
dtype = tf.int32
size = 8
process_rank = 2  # 当前进程 rank

key_tensor, hash_table = get_tensor_and_hashtable(dim, dtype, size, process_rank)

# 打印结果
print("Key Tensor:", key_tensor.numpy())
print("Hash Table Lookup for a key:", hash_table.lookup(tf.constant([key_tensor[0].numpy()])))