import copy
import os
from PIL import Image
import uuid

import pytest
import socket

from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases
from tests.e2e.nightly.multi_node.config.utils import get_cluster_ips
from tests.e2e.nightly.multi_node.config.multi_node_epd_config import ClusterManager, EnvManager
from vllm.utils import get_open_port
from vllm.multimodal.image import convert_image_mode
from vllm import SamplingParams
from lm_service.apis.vllm.proxy import Proxy

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")
CONTAINER_NAME = load_config().get("container_name")

TENSOR_PARALLELS = [1]

SHARED_STORAGE_PATH = "/dev/shm/epd/storage"
ENABLE_PREFIX = [True, False]

DATASET_NAME = ["simulate_truth"]
ROUTER = ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"]

image = convert_image_mode(
    Image.open(Path(__file__).parent.parent.parent.parent / "tools" / "224.png"),
    "RGB")
IMAGE_ARRAY = np.array(image)

SAMPLING_PARAMS = SamplingParams(
    max_tokens=128,
    temperature=0.0
)


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_lm_service_request_timeout_seconds_001(model: str, tp_size: int,
                                               dataset_name: str):
    '''
    lm_service_request_timeout_seconds为空，调用generate接口，调用失败，返回报错信息
    '''
    e_num = 1
    pd_num = 1
    env = {
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "6000",
        "LM_SERVICE_REQUEST_TIMEOUT_SECONDS": "",
        "MC_MS_AUTO_DISC": "0",
        "MC_USE_IPV6": "0",
        "TRANSFER_PROTOCOL": "tcp",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"
    }
    env_dict = EnvManager()
    env_dict.add_env("common", env_dict=env)

    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i), index=i)
    for i in range(pd_num):
        env_dict.add_env("pd", "ASCEND_RT_VISIBLE_DEVICES", str(i + e_num), index=i)

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()
    proxy_addr = f"127.0.0.1:{get_open_port()}"
    e_addr = f"127.0.0.1:{get_open_port()}"
    pd_addr = f"127.0.0.1:{get_open_port()}"

    mooncake_ip = "0.0.0.0"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000","--worker-addr",f"{e_addr}", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000","--worker-addr",f"{pd_addr}", "--max-num-seqs", "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp","transfer_timeout":"20", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
    ]

    mooncake_args = [
        "--rpc_port", str(rpc_port), "--rpc_address", f"{mooncake_ip}", "--enable_http_metadata_server=true",
        f"--http_metadata_server_host={mooncake_ip}",
        f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
        "--default_kv_lease_ttl", "10000", "--eviction_ratio", "0.05",
        "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]
    #proxy_args = ["--proxy-addr",f"{proxy_addr}","--worker-addr",f"{e_addr},{pd_addr}"]

    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               pd_num=1,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

        p = Proxy(
            proxy_addr=proxy_addr,
            encode_addr_list=e_addr,
            pd_addr_list=pd_addr,
            model_name=model,
            enable_health_monitor=True
        )
        try:

            outputs = p.generate(
                prompt={
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SAMPLING_PARAMS,
                request_id=str(uuid.uuid4())

            )
            output = None
            print("proxy is success")
            async for o in outputs:
                output = o
                print(f"{o.outputs}", flush=True)
            assert output.outputs[0].finish_reason == "stop", "request is success"
            p.shutdown()
        except Exception as message:
            print(f"error message is: {str(message)}")
            assert "KeyError:" in str(message), "init success"



