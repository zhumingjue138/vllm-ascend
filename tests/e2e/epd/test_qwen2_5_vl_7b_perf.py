import os

import pytest
import pytest_asyncio
import copy

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases
from tools.aisbench import create_result_plot, create_ttft_plot
from tests.e2e.nightly.multi_node.config.multi_node_epd_config import EnvManager

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")

TENSOR_PARALLELS = [1]
DATASET_NAME = ["image_4", "simulate_truth"]

SHARED_STORAGE_PATH = "/dev/shm/epd/storage"


@pytest_asyncio.fixture(scope="session")
async def teardown():
    yield
    for dataset in DATASET_NAME:
        create_result_plot(result_file_names=[
            f"qwen2_5_vl_7b_{dataset}_PD_mix",
            f"qwen2_5_vl_7b_{dataset}_1E2PD", f"qwen2_5_vl_7b_{dataset}_1E3PD"
        ],
                           result_figure_prefix=dataset)
        create_ttft_plot(result_file_names=[
            f"{dataset}_1E2PD_ttft", f"{dataset}_1E3PD_ttft"
        ],
                         result_figure_prefix=f"{dataset}_ttft")


REQUEST_CONFIG = [("image_4", 0.2, 170), ("image_4", 0.4, 250), ("image_4", 0.6, 300), ("image_4", 0.8, 420), ("image_4", 1.0, 450),
                  ("simulate_truth", 0.2, 180), ("simulate_truth", 0.3, 200), ("simulate_truth", 0.4, 240),
                  ("simulate_truth", 0.5, 280),
                  ("simulate_truth", 0.6, 300)]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name, request_rate, num_prompts", REQUEST_CONFIG)
async def test_pd_mix_001(model: str, tp_size: int, dataset_name: str,
                          request_rate: int, num_prompts: int, teardown):
    api_port = 10001
    vllm_server_args = [
        "--port",
        str(api_port), "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "300", "--enforce-eager",
        "--gpu-memory-utilization", "0.9"
    ]
    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_req"),
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
        "num_prompts": num_prompts,
        "batch_size": 1024,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"qwen2_5_vl_7b_{dataset_name}_PD_mix",
        "threshold": 0.97
    }]

    with RemoteOpenAIServer(model,
                            vllm_server_args,
                            server_host="127.0.0.1",
                            server_port=api_port,
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
                           aisbench_cases=aisbench_cases)


REQUEST_CONFIG = [("image_4", 0.2, 500), ("image_4", 0.4, 800), ("image_4", 0.6, 1200), ("image_4", 0.8, 1300), ("image_4", 1.0, 1400),
                  ("simulate_truth", 0.2, 500), ("simulate_truth", 0.3, 700), ("simulate_truth", 0.4, 800), ("simulate_truth", 0.5, 1000),
                  ("simulate_truth", 0.6, 1100)]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name, request_rate, num_prompts", REQUEST_CONFIG)
async def test_1e3pd_001(model: str, tp_size: int, dataset_name: str,
                         request_rate: int, num_prompts: int, teardown):

    env = {
        "TIMECOUNT_ENABLED": "1",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "120",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300"
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    e_num = 1
    pd_num = 3
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i)

    e_server_args = [
        "--no-enable-prefix-caching", "--model", model,
        "--tensor-parallel-size",
        "1", "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1", "--enforce-eager",
        "--gpu-memory-utilization", "0.0", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--tensor-parallel-size",
        "1", "--max-num-seqs", "300", "--gpu-memory-utilization",
        "0.9", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    proxy_args = ["--router", "LeastInFlightRouter"]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_req"),
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
        "num_prompts": num_prompts,
        "batch_size": 1024,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate*(e_num+pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"qwen2_5_vl_7b_{dataset_name}_1E3PD",
        "threshold": 0.97
    }]

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               proxy_args=proxy_args,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        for aisbench_case in aisbench_cases:
            run_aisbench_cases(model=model,
                               port=api_port,
                               card_num=e_num+pd_num,
                               aisbench_cases=[aisbench_case])
            server.save_ttft_data(file_name=f"{dataset_name}_1E3PD_ttft",
                                  index=aisbench_case["request_rate"] / (e_num+pd_num))


REQUEST_CONFIG = [("image_4", 0.2, 300), ("image_4", 0.4, 600), ("image_4", 0.6, 900), ("image_4", 0.8, 1200),
                  ("image_4", 1.0, 1300), ("simulate_truth", 0.2, 300), ("simulate_truth", 0.3, 500), ("simulate_truth", 0.4, 600), ("simulate_truth", 0.5, 800),
                  ("simulate_truth", 0.6, 900)]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name, request_rate, num_prompts", REQUEST_CONFIG)
async def test_1e2pd_001(model: str, tp_size: int, dataset_name: str,
                         request_rate: int, num_prompts: int, teardown):
    env = {
        "TIMECOUNT_ENABLED": "1",
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "120",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "300"
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i)

    e_server_args = [
        "--no-enable-prefix-caching", "--model", model,
        "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1", "--enforce-eager",
        "--gpu-memory-utilization", "0.0", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "300", "--gpu-memory-utilization",
        "0.9", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    proxy_args = ["--router", "LeastInFlightRouter"]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_req"),
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
        "num_prompts": num_prompts,
        "batch_size": 1024,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate*(e_num+pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"qwen2_5_vl_7b_{dataset_name}_1E2PD",
        "threshold": 0.97
    }]

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               proxy_args=proxy_args,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        for aisbench_case in aisbench_cases:
            run_aisbench_cases(model=model,
                               port=api_port,
                               card_num=e_num+pd_num,
                               aisbench_cases=[aisbench_case])
            server.save_ttft_data(file_name=f"{dataset_name}_1E2PD_ttft",
                                  index=aisbench_case["request_rate"] / (e_num+pd_num))
