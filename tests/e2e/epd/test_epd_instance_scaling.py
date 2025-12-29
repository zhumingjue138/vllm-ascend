import copy
import os

import pytest

from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases
from vllm.utils import get_open_port
from tests.e2e.nightly.multi_node.config.utils import get_cluster_ips
from tests.e2e.nightly.multi_node.config.multi_node_epd_config import ClusterManager, EnvManager

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")
CONTAINER_NAME = load_config().get("container_name")

TENSOR_PARALLELS = [1]

SHARED_STORAGE_PATH = "/dev/shm/epd/storage"
ENABLE_PREFIX = [True, False]

DATASET_NAME = ["simulate_truth"]
ROUTER = ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"]



REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["image_4", "simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_redis_1e1p1d_tcp_mooncake_ipv6_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth
    部署形态： redis 1E1P1D、单机
    存储类型：EC mooncake , KV mooncake
    通信方式：redis worker mooncake ipv6
    redis通信方式： 使用域名
    '''
    e_num = 1
    p_num = 1
    d_num = 1

    env = {
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "TIMECOUNT_ENABLED": "1",
        "VLLM_LOG_STATS_INTERVAL": "10",
        "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVER": "1"
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(p_num):
        env_dict.add_env("p", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i)
    for i in range(d_num):
        env_dict.add_env("d", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num + p_num), index=i)

    rpc_port = 50053
    http_metadata_server_port = 8083
    metrics_port = 9003

    mooncake_ip = "::1"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
        "--metastore-client-config",
        '{"metastore_client": "RedisMetastoreClient",'
        '"metastore_address": "redis://redis.example.com:6380/0"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
            "--metastore-client-config",
            '{"metastore_client": "RedisMetastoreClient",'
            '"metastore_address": "redis://redis.example.com:6380/0"}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
            "--metastore-client-config",
            '{"metastore_client": "RedisMetastoreClient",'
            '"metastore_address": "redis://redis.example.com:6380/0"}'
        ]
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", "::", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=::",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "--client_ttl","1","--eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]
    proxy_args = [
            "--metastore-client-config",
            '{"metastore_client": "RedisMetastoreClient",'
            '"metastore_address": "redis://redis.example.com:6380/0"}'
    ]

    aisbench_cases = [{
        "case_type": "pressure",
        "pressure_time": 300,
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
        "baseline": 1,
        "seed": 77,
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args,
                               mooncake_args=mooncake_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=3,
                           aisbench_cases=aisbench_cases,
                           verify=False,
                           save=False)

REQUEST_RATE = [0.28]
DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_redis_proxy_1e1p1d_tcp_mooncake_ipv6_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth
    部署形态： redis 1E1P1D、单机
    存储类型：EC mooncake , KV mooncake
    通信方式：redis worker mooncake ipv6
    redis通信方式： 使用域名
    '''
    e_num = 1
    p_num = 1
    d_num = 1

    env = {
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "TIMECOUNT_ENABLED": "1",
        "VLLM_LOG_STATS_INTERVAL": "10",
        "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVER": "1"
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(p_num):
        env_dict.add_env("p", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i)
    for i in range(d_num):
        env_dict.add_env("d", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num + p_num), index=i)

    rpc_port = 50053
    http_metadata_server_port = 8083
    metrics_port = 9003


    mooncake_ip = "::1"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "20000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
        "--metastore-client-config",
        '{"metastore_client": "RedisMetastoreClient",'
        '"metastore_address": "redis://redis.example.com:6380/0"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "20000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
            "--metastore-client-config",
            '{"metastore_client": "RedisMetastoreClient",'
            '"metastore_address": "redis://redis.example.com:6380/0"}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "20000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
            "--metastore-client-config",
            '{"metastore_client": "RedisMetastoreClient",'
            '"metastore_address": "redis://redis.example.com:6380/0"}'
        ]
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", "::", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=::",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000","--eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]
    proxy_args = [
            "--metastore-client-config",
            '{"metastore_client": "RedisMetastoreClient",'
            '"metastore_address": "redis://redis.example.com:6380/0"}'
    ]

    aisbench_cases = [{
        "case_type": "pressure",
        "pressure_time": 300,
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
        "baseline": 1,
        "seed": 77,
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args,
                               mooncake_args=mooncake_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=3,
                           aisbench_cases=aisbench_cases,
                           verify=False,
                           save=False)

REQUEST_RATE = [0.28]
DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_redis_proxy_1e1p1d_cross_tcp_mooncake_ipv4_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth
    部署形态： redis 1E1P1D、跨机
    存储类型：EC mooncake , KV mooncake
    通信方式：redis worker mooncake ipv4
    redis通信方式： 使用域名
    '''
    e_num = 1
    p_num = 1
    d_num = 1

    env = {
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "TIMECOUNT_ENABLED": "1",
        "VLLM_LOG_STATS_INTERVAL": "10",
        "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVER": "1"
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(p_num):
        env_dict.add_env("p", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i)
    for i in range(d_num):
        env_dict.add_env("d", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num + p_num), index=i)

    rpc_port = 50053
    http_metadata_server_port = 8083
    metrics_port = 9003

    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(p_num):
        cluster.add_node_info("p", 1, CONTAINER_NAME)
    for i in range(d_num):
        cluster.add_node_info("d", 1, CONTAINER_NAME)
    node_ips = get_cluster_ips()
    mooncake_ip = node_ips[0]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "20000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}',
        "--metastore-client-config",
        '{"metastore_client": "RedisMetastoreClient",'
        '"metastore_address": "redis://redis.example.com:6380/0"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "20000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}',
            "--metastore-client-config",
            '{"metastore_client": "RedisMetastoreClient",'
            '"metastore_address": "redis://redis.example.com:6380/0"}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "20000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}',
            "--metastore-client-config",
            '{"metastore_client": "RedisMetastoreClient",'
            '"metastore_address": "redis://redis.example.com:6380/0"}'
        ]
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", f"{mooncake_ip}", "--enable_http_metadata_server=true",
            f"--http_metadata_server_host={mooncake_ip}",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "--eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]
    proxy_args = [
            "--metastore-client-config",
            '{"metastore_client": "RedisMetastoreClient",'
            '"metastore_address": "redis://redis.example.com:6380/0"}'
    ]

    aisbench_cases = [{
        "case_type": "pressure",
        "pressure_time": 300,
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
        "baseline": 1,
        "seed": 77,
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               node_info=cluster,
                               proxy_args=proxy_args,
                               mooncake_args=mooncake_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=3,
                           aisbench_cases=aisbench_cases,
                           verify=False,
                           save=False)

REQUEST_RATE = [0.28]
DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_no_redis_1proxy_1e1p1d_tcp_mooncake_ipv6_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth
    部署形态： 1E1P1D、单机
    存储类型：EC mooncake , KV mooncake
    通信方式：worker mooncake ipv6
    '''
    e_num = 1
    p_num = 1
    d_num = 1

    env = {
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "TIMECOUNT_ENABLED": "1",
        "VLLM_LOG_STATS_INTERVAL": "10",
        "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVER": "1"
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(p_num):
        env_dict.add_env("p", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i)
    for i in range(d_num):
        env_dict.add_env("d", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num + p_num), index=i)

    rpc_port = 50053
    http_metadata_server_port = 8083
    metrics_port = 9003


    mooncake_ip = "::1"
    proxy_addr = f"[{mooncake_ip}]:13800"

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "20000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--proxy-addr", f"{proxy_addr}", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "20000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--proxy-addr", f"{proxy_addr}",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "20000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--proxy-addr", f"{proxy_addr}",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}'
        ]
    ]

    mooncake_args = [
        "--rpc_port", str(rpc_port), "--rpc_address", "::", "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
        f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
        "--default_kv_lease_ttl", "10000","--eviction_ratio", "0.05",
        "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]
    proxy_args = [
        "--enable_health_monitor", "True"
    ]

    aisbench_cases = [{
        "case_type": "pressure",
        "pressure_time": 300,
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
        "baseline": 1,
        "seed": 77,
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args,
                               mooncake_args=mooncake_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=3,
                           aisbench_cases=aisbench_cases,
                           verify=False,
                           save=False)

REQUEST_RATE = [0.28]
DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_no_redis_2proxy_1e1p1d_tcp_mooncake_ipv6_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth
    部署形态： 1E1P1D、单机
    存储类型：EC mooncake , KV mooncake
    通信方式：worker mooncake ipv6
    '''
    e_num = 1
    p_num = 1
    d_num = 1

    env = {
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "1",
        "TRANSFER_PROTOCOL": "tcp",
        "TIMECOUNT_ENABLED": "1",
        "VLLM_LOG_STATS_INTERVAL": "10",
        "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVER": "1"
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(p_num):
        env_dict.add_env("p", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i)
    for i in range(d_num):
        env_dict.add_env("d", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num + p_num), index=i)

    rpc_port = 50053
    http_metadata_server_port = 8083
    metrics_port = 9003


    mooncake_ip = "::1"
    proxy_addr1 = f"[{mooncake_ip}]:13700"
    proxy_addr2 = f"[{mooncake_ip}]:13800"
    e_addr = f"[{mooncake_ip}]:13811"
    p_addr = f"[{mooncake_ip}]:13812"
    d_addr = f"[{mooncake_ip}]:13813"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "20000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--worker-addr", f"{e_addr}", "--proxy-addr", f"{proxy_addr1}", f"{proxy_addr2}", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "20000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--worker-addr", f"{p_addr}", "--proxy-addr", f"{proxy_addr1}", f"{proxy_addr2}", 
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "20000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--worker-addr", f"{d_addr}", "--proxy-addr", f"{proxy_addr1}", f"{proxy_addr2}", 
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}'
        ]
    ]

    mooncake_args = [
        "--rpc_port", str(rpc_port), "--rpc_address", "::", "--enable_http_metadata_server=true",
        "--http_metadata_server_host=::",
        f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
        "--default_kv_lease_ttl", "10000","--eviction_ratio", "0.05",
        "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]
    proxy_args = [
        "--enable_health_monitor", "True"
    ]

    aisbench_cases = [{
        "case_type": "pressure",
        "pressure_time": 300,
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
        "baseline": 1,
        "seed": 77,
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args,
                               mooncake_args=mooncake_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=3,
                           aisbench_cases=aisbench_cases,
                           verify=False,
                           save=False)

REQUEST_RATE = [0.28]
DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_no_redis_proxy_1e1p1d_cross_tcp_mooncake_ipv4_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth
    部署形态： 1E1P1D、跨机
    存储类型：EC mooncake , KV mooncake
    通信方式：worker mooncake ipv4
    '''
    e_num = 1
    p_num = 1
    d_num = 1

    env = {
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "TIMECOUNT_ENABLED": "1",
        "VLLM_LOG_STATS_INTERVAL": "10",
        "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVER": "1"
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(p_num):
        env_dict.add_env("p", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i)
    for i in range(d_num):
        env_dict.add_env("d", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num + p_num), index=i)

    rpc_port = 50053
    http_metadata_server_port = 8083
    metrics_port = 9003

    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(p_num):
        cluster.add_node_info("p", 1, CONTAINER_NAME)
    for i in range(d_num):
        cluster.add_node_info("d", 1, CONTAINER_NAME)
    node_ips = get_cluster_ips()
    mooncake_ip = node_ips[0]
    proxy_addr = f"{mooncake_ip}:13800"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "20000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--proxy-addr", f"{proxy_addr}", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "20000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--proxy-addr", f"{proxy_addr}",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "0"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "20000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--proxy-addr", f"{proxy_addr}",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "0"}}'
        ]
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", f"{mooncake_ip}", "--enable_http_metadata_server=true",
            f"--http_metadata_server_host={mooncake_ip}",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "--eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]
    proxy_args = [
            "--enable_health_monitor", "True"
    ]

    aisbench_cases = [{
        "case_type": "pressure",
        "pressure_time": 600,
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
        "baseline": 1,
        "seed": 77,
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               node_info=cluster,
                               proxy_args=proxy_args,
                               mooncake_args=mooncake_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=3,
                           aisbench_cases=aisbench_cases,
                           verify=False,
                           save=False)