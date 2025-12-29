import os

import pytest
import pytest_asyncio
import copy

from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases
from tests.e2e.nightly.multi_node.config.multi_node_epd_config import ClusterManager, EnvManager
from vllm.utils import get_open_port

model_path = load_config().get("model_path")
CONTAINER_NAME = load_config().get("container_name")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")
SHARED_STORAGE_PATH = "/dev/shm/epd/storage"


TENSOR_PARALLELS = [1]
PREFIX_CACHE = [True, False]
ROUTER = ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"]

REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e1pd_shm_tcp_001(model: str, tp_size: int, dataset_name: str,
                                 request_rate: float):
    '''
    1E1PD 单机部署
    前缀缓存： 开启
    数据集：模拟ZJ
    ec transfer: shm
    通信方式：tcp(ipv4)
    '''

    env = {"TRANSFER_PROTOCOL": "tcp"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    e_num = 1
    pd_num = 1
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "image_4"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * (e_num + pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_shm_tcp",
        "threshold": 0.97
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num+e_num,
                           aisbench_cases=aisbench_cases)



@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e1pd_sc_shm_tcp_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    1E1PD共卡 单机部署
    前缀缓存： 开启
    数据集：模拟ZJ
    ec transfer: shm
    通信方式：tcp(ipv4)
    '''
    env = {"TRANSFER_PROTOCOL": "tcp"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    e_num = 1
    pd_num = 1
    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")
    env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", "0")

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "image_4"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_sc_shm_tcp",
        "threshold": 0.97
    }]
    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)


DATASET_NAME = ["simulate_truth_samereq"]


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("prefix", PREFIX_CACHE)
async def test_1e2pd_shm_tcp_001(model: str, tp_size: int,
                                    dataset_name: str, request_rate: float, prefix: bool):
    '''
    1E2PD 单机部署
    前缀缓存： 开启/关闭
    数据集：同请求及跨请求存在相同图片
    ec transfer: shm
    通信方式：tcp(ipv4),环境变量指定
    '''
    env = {"TRANSFER_PROTOCOL": "tcp"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    if not prefix:
        pd_server_args.append("--no-enable-prefix-caching")

    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "image_4"),
        "request_conf":
        "vllm_api_stream_chat",
        "dataset_conf":
        "textvqa/textvqa_gen_base64",
        "num_prompts":
        50,
        "max_out_len":
        256,
        "batch_size":
        16,
        "temperature":
        0.5,
        "top_k":
        10,
        "top_p":
        0.7,
        "repetition_penalty":
        1.2,
        "request_rate":
        0,
        "seed":
        77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate*(e_num+pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shm_tcp_001",
        "threshold": 0.97
    }]
    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num + e_num,
                           aisbench_cases=aisbench_cases)


DATASET_NAME = ["simulate_truth", "image_4"]


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e2pd_shm_tcp_002(model: str, tp_size: int,
                                    dataset_name: str, request_rate: float):
    '''
    1E2PD 单机部署
    前缀缓存： 开启
    数据集：模拟zj，image_4
    ec transfer: shm
    通信方式：tcp(ipv4)
    '''

    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    proxy_args = ["--transfer-protocol", "tcp"]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--transfer-protocol", "tcp", "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--transfer-protocol", "tcp", "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf":
        "vllm_api_stream_chat",
        "dataset_conf":
        "textvqa/textvqa_gen_base64",
        "num_prompts":
        50,
        "max_out_len":
        256,
        "batch_size":
        16,
        "temperature":
        0.5,
        "top_k":
        10,
        "top_p":
        0.7,
        "repetition_penalty":
        1.2,
        "request_rate":
        0,
        "seed":
        77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate*(e_num+pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shm_tcp_002",
        "threshold": 0.97
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num + e_num,
                           aisbench_cases=aisbench_cases)


DATASET_NAME = ["textvqa-subset"]


@pytest.mark.asyncio
@pytest.mark.acc
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_shm_tcp_003(model: str, tp_size: int,
                                 dataset_name: str):
    '''
    1E2PD 单机部署
    前缀缓存： 开启
    数据集：textvqa-subset
    ec transfer: shm
    通信方式：tcp(ipv4)
    '''

    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    proxy_args = ["--transfer-protocol", "tcp"]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--transfer-protocol", "tcp", "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--transfer-protocol", "tcp", "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    acc_cases = [{
        "case_type": "accuracy",
        "dataset_path": os.path.join(DATASET_PATH, "textvqa_subset"),
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "max_out_len": 2048,
        "batch_size": 128,
        "temperature": 0,
        "top_k": -1,
        "top_p": 1,
        "repetition_penalty": 1,
        "request_rate": 0,
        "baseline": 81,
        "seed": 77,
        "threshold": 1
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num + e_num,
                           aisbench_cases=acc_cases, save=False)




DATASET_NAME = ["simulate_truth", "image_4"]


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e2pd_shm_tcp_004(model: str, tp_size: int,
                                 dataset_name: str, request_rate: float):
    '''
    1E2PD 单机部署
    前缀缓存： 开启
    数据集：模拟ZJ，image_4
    ec transfer: shm
    通信方式：tcp(ipv6)
    '''

    env = {"MC_USE_IPV6": "1", "TRANSFER_PROTOCOL": "tcp"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf":
            "vllm_api_stream_chat",
        "dataset_conf":
            "textvqa/textvqa_gen_base64",
        "num_prompts":
            50,
        "max_out_len":
            256,
        "batch_size":
            16,
        "temperature":
            0.5,
        "top_k":
            10,
        "top_p":
            0.7,
        "repetition_penalty":
            1.2,
        "request_rate":
            0,
        "seed":
            77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate*(e_num+pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shm_tcp_004",
        "threshold": 0.97
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)

        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num+e_num,
                           aisbench_cases=aisbench_cases)




DATASET_NAME = ["simulate_truth", "image_4"]


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e2pd_shm_ipc_001(model: str, tp_size: int,
                                 dataset_name: str, request_rate: float):
    '''
    1E2PD 单机部署
    前缀缓存： 开启
    数据集：模拟ZJ，image_4
    ec transfer: shm
    通信方式：ipc
    '''

    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf":
            "vllm_api_stream_chat",
        "dataset_conf":
            "textvqa/textvqa_gen_base64",
        "num_prompts":
            50,
        "max_out_len":
            256,
        "batch_size":
            16,
        "temperature":
            0.5,
        "top_k":
            10,
        "top_p":
            0.7,
        "repetition_penalty":
            1.2,
        "request_rate":
            0,
        "seed":
            77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate*(e_num+pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shm_ipc_001",
        "threshold": 0.97
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)

        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num+e_num,
                           aisbench_cases=aisbench_cases)



DATASET_NAME = ["simulate_truth"]


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("router", ROUTER)
async def test_3e5pd_shm_tcp_001(model: str, tp_size: int,
                                    dataset_name: str, request_rate: float, router: str):
    '''
    3E5PD 单机部署
    前缀缓存： 开启
    数据集：模拟ZJ
    ec transfer: shm
    通信方式：tcp(ipv6)
    '''
    env = {"TRANSFER_PROTOCOL": "tcp", "MC_USE_IPV6": "1"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    e_num = 3
    pd_num = 5
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    proxy_args = ["--router", router]

    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "image_4"),
        "request_conf":
        "vllm_api_stream_chat",
        "dataset_conf":
        "textvqa/textvqa_gen_base64",
        "num_prompts":
        50,
        "max_out_len":
        256,
        "batch_size":
        16,
        "temperature":
        0.5,
        "top_k":
        10,
        "top_p":
        0.7,
        "repetition_penalty":
        1.2,
        "request_rate":
        0,
        "seed":
        77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate*(e_num+pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_3E5PD_shm_tcp",
        "threshold": 0.97
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num + e_num,
                           aisbench_cases=aisbench_cases)


DATASET_NAME = ["simulate_truth", "image_4"]


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e2pd_cross_p_epd_shm_tcp_001(model: str, tp_size: int,
                                                 dataset_name: str, request_rate: float):
    '''
    proxy-epd 跨机部署, 1E2PD
    前缀缓存： 开启
    数据集：模拟zj, image_4
    测试类型：perf
    ec transfer: shm
    通信方式：tcp(ipv4)
    '''

    e_num = 1
    pd_num = 2
    env = {"TRANSFER_PROTOCOL": "tcp"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(pd_num):
        cluster.add_node_info("pd", 1, CONTAINER_NAME)
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf":
        "vllm_api_stream_chat",
        "dataset_conf":
        "textvqa/textvqa_gen_base64",
        "num_prompts":
        50,
        "max_out_len":
        256,
        "batch_size":
        16,
        "temperature":
        0.5,
        "top_k":
        10,
        "top_p":
        0.7,
        "repetition_penalty":
        1.2,
        "request_rate":
        0,
        "seed":
        77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH,dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "result_file_name": f"{dataset_name}_1E2PD_cross_p_epd_shm_tcp",
        "request_rate": request_rate*(e_num+pd_num),
        "baseline": 1,
        "seed": 77,
        "threshold": 0.97
    }]
    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               node_info=cluster,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:

        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # test perf
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num + e_num,
                           aisbench_cases=aisbench_cases)


DATASET_NAME = ["textvqa-subset"]


@pytest.mark.asyncio
@pytest.mark.acc
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_cross_p_epd_shm_tcp_002(model: str, tp_size: int,
                                                 dataset_name: str):
    '''
    proxy-epd 跨机部署, 1E2PD
    前缀缓存： 开启
    数据集：textvqa-subset
    ec transfer: shm
    通信方式：tcp(ipv4)
    '''

    e_num = 1
    pd_num = 2
    env = {"TRANSFER_PROTOCOL": "tcp"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(pd_num):
        cluster.add_node_info("pd", 1, CONTAINER_NAME)
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    acc_cases = [{
        "case_type": "accuracy",
        "dataset_path": os.path.join(DATASET_PATH, "textvqa_subset"),
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "max_out_len": 2048,
        "batch_size": 128,
        "temperature": 0,
        "top_k": -1,
        "top_p": 1,
        "repetition_penalty": 1,
        "request_rate": 0,
        "baseline": 81,
        "seed": 77,
        "threshold": 1
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               node_info=cluster,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:

        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num + e_num,
                           aisbench_cases=acc_cases, save=False)


DATASET_NAME = ["simulate_truth"]


@pytest.mark.asyncio
@pytest.mark.stability
@pytest.mark.timeout(90000)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_cross_p_epd_shm_tcp_003(model: str, tp_size: int,
                                                 dataset_name: str):
    '''
    proxy-epd 跨机部署, 1E2PD
    前缀缓存： 开启
    数据集：模拟zj
    ec transfer: shm
    通信方式：tcp(ipv4)
    '''

    e_num = 1
    pd_num = 2
    env = {"TRANSFER_PROTOCOL": "tcp"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(pd_num):
        cluster.add_node_info("pd", 1, CONTAINER_NAME)
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    aisbench_cases = [{
        "case_type": "pressure",
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "batch_size": 128,
        "temperature": 0.5,
        "pressure_time": 86400,
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.84,
        "baseline": 1,
        "seed": 77,
        "threshold": 0.97,
        "result_file_name": f"{dataset_name}_1E2PD_cross_p_epd_shm_tcp_003"
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               node_info=cluster,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:

        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num + e_num,
                           aisbench_cases=aisbench_cases)





REQUEST_RATE = [0.02, 0.06, 0.12]
MODELS = [os.path.join(model_path, "Qwen3-VL-30B-A3B-Instruct")]
DATASET_NAME = ["simulate_truth", "image_4"]

@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e1pd_shm_tcp_002(model: str, tp_size: int, dataset_name: str,
                                 request_rate: float):
    '''
    1E1PD 单机部署
    前缀缓存： 开启
    数据集：模拟ZJ
    ec transfer: shm
    通信方式：tcp(ipv6)
    模型：qwen3-30B(E:TP=1, PD: TP=4)
    '''

    env = {"TRANSFER_PROTOCOL": "tcp", "MC_USE_IPV6": "1","VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "120",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    e_num = 1
    pd_num = 1

    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")

    env_dict.add_env("pd",
                     "ASCEND_RT_VISIBLE_DEVICES", "1,2,3,4")

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        "1", "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "20000", "--max-num-batched-tokens", "20000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.9",
        "--tensor-parallel-size",
        "4", "--enforce-eager", "--max-model-len", "20000",
        "--max-num-batched-tokens", "20000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 100,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * (e_num + pd_num*4),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_shm_tcp_002",
        "threshold": 0.97
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num*4+e_num,
                           aisbench_cases=aisbench_cases)




DATASET_NAME = ["simulate_truth"]

@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e1pd_shm_ipc_001(model: str, tp_size: int, dataset_name: str,
                                 request_rate: float):
    '''
    1E1PD 单机部署
    前缀缓存： 开启
    数据集：模拟ZJ
    ec transfer: shm
    通信方式：ipc
    模型：qwen3-30B(E:TP=1, PD: TP=4)
    '''

    env = {"VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "120", "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    e_num = 1
    pd_num = 1

    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")

    env_dict.add_env("pd",
                     "ASCEND_RT_VISIBLE_DEVICES", "1,2,3,4")

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        "1", "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "20000", "--max-num-batched-tokens", "20000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.9",
        "--tensor-parallel-size",
        "4", "--enforce-eager", "--max-model-len", "20000",
        "--max-num-batched-tokens", "20000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "image_4"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 100,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * (e_num + pd_num*4),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_shm_ipc_001",
        "threshold": 0.97
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num*4+e_num,
                           aisbench_cases=aisbench_cases)



DATASET_NAME = ["textvqa-subset"]

@pytest.mark.asyncio
@pytest.mark.acc
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1pd_cross_p_epd_shm_tcp_001(model: str, tp_size: int,
                                 dataset_name: str):
    '''
    1E1PD proxy-epd跨机部署
    前缀缓存： 开启
    数据集：textvqa-subset
    ec transfer: shm
    通信方式：tcp(ipv6)
    '''

    env_dict = EnvManager()
    e_num = 1
    pd_num = 1

    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")

    env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", "1,2,3,4")

    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(pd_num):
        cluster.add_node_info("pd", 1, CONTAINER_NAME)

    proxy_args = ["--transfer-protocol", "tcp"]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--transfer-protocol", "tcp", "--tensor-parallel-size",
        "1", "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "20000", "--max-num-batched-tokens", "20000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.9",
        "--transfer-protocol", "tcp", "--tensor-parallel-size",
        "4", "--enforce-eager", "--max-model-len", "20000",
        "--max-num-batched-tokens", "20000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    acc_cases = [{
        "case_type": "accuracy",
        "dataset_path": os.path.join(DATASET_PATH, "textvqa_subset"),
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "max_out_len": 2048,
        "batch_size": 128,
        "temperature": 0,
        "top_k": -1,
        "top_p": 1,
        "repetition_penalty": 1,
        "request_rate": 0,
        "baseline": 81,
        "seed": 77,
        "threshold": 1
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               node_info=cluster,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=acc_cases, save=False)




DATASET_NAME = ["simulate_truth"]

@pytest.mark.asyncio
@pytest.mark.stability
@pytest.mark.timeout(90000)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1pd_shm_tcp_004(model: str, tp_size: int, dataset_name: str):
    '''
    1E1PD 单机部署
    前缀缓存： 开启
    数据集：模拟ZJ
    ec transfer: shm
    通信方式：tcp(ipv6)
    模型：qwen3-30B(E:TP=1, PD: TP=4)
    '''

    env = {"TRANSFER_PROTOCOL": "tcp", "MC_USE_IPV6": "1","VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "120",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300"}

    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    e_num = 1
    pd_num = 1

    env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", "0")

    env_dict.add_env("pd",
                     "ASCEND_RT_VISIBLE_DEVICES", "1,2,3,4")

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        "1", "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "20000", "--max-num-batched-tokens", "20000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.9",
        "--tensor-parallel-size",
        "4", "--enforce-eager", "--max-model-len", "20000",
        "--max-num-batched-tokens", "20000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    aisbench_cases = [{
        "case_type": "pressure",
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "batch_size": 128,
        "temperature": 0.5,
        "pressure_time": 86400,
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.12,
        "baseline": 1,
        "seed": 77,
        "threshold": 0.97,
        "result_file_name": f"{dataset_name}_1E1PD_shm_tcp_003"
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:

        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num*4+e_num,
                           aisbench_cases=aisbench_cases)

        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=pd_num*4+e_num,
                           aisbench_cases=aisbench_cases)



DATASET_NAME = ["simulate_truth", "image_4"]
TENSOR_PARALLELS = [4]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_pd_mix_001(model: str, tp_size: int, dataset_name: str, request_rate: float):
    '''
    PD合并 单机部署
    前缀缓存： 开启
    数据集：模拟ZJ
    ec transfer: shm
    模型：qwen3-30B(TP=4)
    '''

    env_dict = {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3"}
    api_port = get_open_port()
    vllm_server_args = [
        "--port",
        str(api_port), "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "20000", "--max-num-batched-tokens",
        "20000", "--max-num-seqs", "128", "--enforce-eager",
        "--gpu-memory-utilization", "0.9"
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 100,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate*4,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_PD_mix",
        "threshold": 0.97
    }]

    with RemoteOpenAIServer(model,
                            vllm_server_args,
                            server_host="127.0.0.1",
                            server_port=api_port,
                            env_dict=env_dict,
                            auto_port=False) as server:

        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=4,
                           aisbench_cases=aisbench_cases)


