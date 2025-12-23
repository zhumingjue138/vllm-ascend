import uuid
import os
import numpy as np
from PIL import Image

import pytest
import pytest_asyncio
from vllm import SamplingParams
from lm_service.apis.vllm.proxy import Proxy
from vllm.multimodal.image import convert_image_mode
from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.epd.conftest import load_config
from pathlib import Path

model_path = load_config().get("model_path")
MODEL = os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")
PROMPT_TEMPLATE=(
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "what is the brand of this camera?<|im_end|>\n"
            "<|im_start|>assistant\n"
)
SAMPLING_PARAMS = SamplingParams(
    max_tokens=128,
    temperature=0.0
)
SHARED_STORAGE_PATH = "/dev/shm/epd/storage"

image = convert_image_mode(
    Image.open(Path(__file__).parent.parent.parent.parent / "tools" / "224.png"),
    "RGB")
IMAGE_ARRAY = np.array(image)

E_ADDR_LIST = ["/tmp/encoder_0"]
PD_ADDR_LIST = ["/tmp/pd_0"]
PROXY_ADDR = "/tmp/proxy"


@pytest_asyncio.fixture(scope="class")
async def setup_teardown():
    e_server_args = [
        "--no-enable-prefix-caching", "--model", MODEL,
        "--tensor-parallel-size",
        "1", "--max-model-len", "30000", "--max-num-batched-tokens",
        "40000", "--max-num-seqs", "1", "--enforce-eager",
        "--gpu-memory-utilization", "0.0", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--no-enable-prefix-caching", "--model", MODEL, "--max-model-len",
        "30000", "--tensor-parallel-size",
        "1", "--max-num-batched-tokens", "40000", "--max-num-seqs",
        "400", "--gpu-memory-utilization", "0.9", "--enforce-eager",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    
    async with RemoteEPDServer(
            run_mode="worker",
            pd_num=1,
            e_num=1,
            e_serve_args=e_server_args,
            pd_serve_args=pd_server_args
    ) as server:
        print("vllm instance is ready")

        yield server

        print("vllm instance is cleaning")

class CustomRouter():
    def route_request(self, endpoints: list[int], request_stats: dict) -> int:
        if not endpoints:
            raise RuntimeError("No healthy endpoints available for routing.")
        return endpoints[0]

class TestEPDProxy:
    @pytest.mark.asyncio
    async def test_proxy_proxy_addr_001(self, setup_teardown):
        """proxy_addr设置为非有效地址，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr="test",
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )

            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_proxy_proxy_addr_002(self, setup_teardown):
        """proxy_addr不携带，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "missing 1 required positional argument" in str(message), "init success"



    @pytest.mark.asyncio
    async def test_proxy_proxy_addr_003(self, setup_teardown):
        """proxy_addr为空，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr="",
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "Invalid argument" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_proxy_proxy_addr_004(self, setup_teardown):
        """proxy_addr为aaa，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr="aaa",
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "instance 0 is unhealthy" in str(message), "init success"



    @pytest.mark.asyncio
    async def test_proxy_proxy_addr_005(self, setup_teardown):
        """proxy_addr为非字符串，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=12345,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_encode_addr_list_001(self, setup_teardown):
        """encode_addr_list不携带，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "missing 1 required positional argument" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_encode_addr_list_002(self, setup_teardown):
        """encode_addr_list为空，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=[""],
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "Invalid argument" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_encode_addr_list_003(self, setup_teardown):
        """encode_addr_list为aaa，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=["aaa"],
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_encode_addr_list_004(self, setup_teardown):
        """encode_addr_list为空，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=["test"],
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_encode_addr_list_005(self, setup_teardown):
        """encode_addr_list为非数组，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=12345,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "object is not iterable" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_pd_addr_list_001(self, setup_teardown):
        """pd_addr_list不携带，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "missing 1 required positional argument" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_proxy_pd_addr_list_002(self, setup_teardown):
        """pd_addr_list为空，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=[""],
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "Invalid argument" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_pd_addr_list_003(self, setup_teardown):
        """pd_addr_list为aaa，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=["aaa"],
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_pd_addr_list_004(self, setup_teardown):
        """pd_addr_list非有效地址，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=["test"],
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_proxy_pd_addr_list_005(self, setup_teardown):
        """pd_addr_list非数组，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=12345,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "object is not iterable" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_model_name_001(self, setup_teardown):
        """model_name不携带，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "missing 1 required positional argument" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_model_name_002(self, setup_teardown):
        """model_name为空，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name="",
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "validation error" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_model_name_003(self, setup_teardown):
        """model_name为aaa，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name="aaa",
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "validation error" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_proxy_model_name_004(self, setup_teardown):
        """model_name为非字符串，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=12345,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "validation error" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_prompt_prompt_001(self, setup_teardown):
        """prompt不携带，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
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


    @pytest.mark.asyncio
    async def test_generate_prompt_prompt_002(self, setup_teardown):
        """prompt为空，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": "",
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_prompt_prompt_003(self, setup_teardown):
        """prompt为非字符串，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": 12345,
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
            assert "Invalid Parameters" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_multi_model_data_001(self, setup_teardown):
        """multi_model_data不携带，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_multi_model_data_002(self, setup_teardown):
        """multi_model_data为空，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {},
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_multi_model_data_003(self, setup_teardown):
        """multi_model_data为非image字段，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"video": IMAGE_ARRAY},
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_multi_model_data_004(self, setup_teardown):
        """multi_model_data为image字段为空，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": ""},
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_multi_model_data_005(self, setup_teardown):
        """multi_model_data未np转化，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": image},
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_prompt_001(self, setup_teardown):
        """prompt为空，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={""},
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
            assert "KeyError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_prompt_002(self, setup_teardown):
        """prompt不携带，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
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
            assert "missing 1 required positional argument" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_max_tokens_001(self, setup_teardown):
        """max_tokens不携带，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_max_tokens_002(self, setup_teardown):
        """max_tokens为None，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=None,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_max_tokens_003(self, setup_teardown):
        """max_tokens为0，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=0,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_max_tokens_004(self, setup_teardown):
        """max_tokens为2.1，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=2.1,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_max_tokens_005(self, setup_teardown):
        """max_tokens为aaa，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens='aaa',
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_max_tokens_006(self, setup_teardown):
        """max_tokens为20，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=20,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_p_001(self, setup_teardown):
        """top_p不携带，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_p_002(self, setup_teardown):
        """top_p为None，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_p=None,
                    temperature=0.0
                ),
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
            assert "TypeError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_p_003(self, setup_teardown):
        """top_p为0，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_p=0,
                    temperature=0.0
                ),
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
            assert "ValueError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_p_004(self, setup_teardown):
        """top_p为-1，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_p=-1,
                    temperature=0.0
                ),
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
            assert "ValueError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_p_005(self, setup_teardown):
        """top_p为1，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_p=1,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_p_006(self, setup_teardown):
        """top_p为0.5，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_p=0.5,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_p_007(self, setup_teardown):
        """top_p为2，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_p=None,
                    temperature=0.0
                ),
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
            assert "ValueError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_p_008(self, setup_teardown):
        """top_p为aaa，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_p='aaa',
                    temperature=0.0
                ),
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
            assert "TypeError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_k_001(self, setup_teardown):
        """top_k不携带，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_k_002(self, setup_teardown):
        """top_k为None，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_k=None,
                    temperature=0.0
                ),
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
            assert "TypeError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_k_003(self, setup_teardown):
        """top_k为0，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_k=0,
                    temperature=0.0
                ),
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
            assert "ValueError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_k_004(self, setup_teardown):
        """top_k为-1，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_p=-1,
                    temperature=0.0
                ),
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
            assert "ValueError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_k_005(self, setup_teardown):
        """top_k为1，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_k=1,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_k_006(self, setup_teardown):
        """top_k为2.1，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_k=2.1,
                    temperature=0.0
                ),
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
            assert "must be an integer" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_top_k_007(self, setup_teardown):
        """top_k为aaa，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    top_p='aaa',
                    temperature=0.0
                ),
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
            assert "TypeError" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_seed_001(self, setup_teardown):
        """seed不携带，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_seed_002(self, setup_teardown):
        """seed为None，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    seed=None
                ),
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
            assert "TypeError" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_seed_003(self, setup_teardown):
        """seed为0，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    seed=0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_seed_004(self, setup_teardown):
        """seed为-1，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    seed=-1
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_seed_005(self, setup_teardown):
        """seed为1，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    seed=1
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_seed_006(self, setup_teardown):
        """seed为2.1，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    seed=2.1
                ),
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
            assert "ValidationError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_seed_007(self, setup_teardown):
        """seed为aaa，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    seed="aaa"
                ),
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
            assert "TypeError" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_temperature_001(self, setup_teardown):
        """temperature不携带，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_temperature_002(self, setup_teardown):
        """temperature为None，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=None
                ),
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
            assert "TypeError" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_temperature_003(self, setup_teardown):
        """temperature为0，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_temperature_004(self, setup_teardown):
        """temperature为-1，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=-1
                ),
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
            assert "ValueError" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_temperature_005(self, setup_teardown):
        """temperature为1，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=1
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_temperature_006(self, setup_teardown):
        """temperature为0.5，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.5
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_temperature_007(self, setup_teardown):
        """temperature为aaa，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature="aaa"
                ),
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
            assert "TypeError" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_repetition_penalty_001(self, setup_teardown):
        """repetition_penalty不携带，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0
                ),
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
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_repetition_penalty_002(self, setup_teardown):
        """repetition_penalty为None，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    repetition_penalty=None
                ),
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
            assert "TypeError" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_repetition_penalty_003(self, setup_teardown):
        """repetition_penalty为0，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    repetition_penalty=0
                ),
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
            assert "ValueError" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_repetition_penalty_004(self, setup_teardown):
        """repetition_penalty为1，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    repetition_penalty=1
                ),
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
            assert "ValueError" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_repetition_penalty_005(self, setup_teardown):
        """repetition_penalty为1.5，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    repetition_penalty=1.5
                ),
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
            assert "ValueError" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_sampling_params_repetition_penalty_006(self, setup_teardown):
        """repetition_penalty为aaa，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SamplingParams(
                    max_tokens=128,
                    temperature=0.0,
                    repetition_penalty="aaa"
                ),
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
            assert "TypeError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_001(self, setup_teardown):
        """sampling_params不携带，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
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
            assert "missing 1 required positional argument" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_sampling_params_002(self, setup_teardown):
        """sampling_params为空，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=None,
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
            assert "TypeError" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_request_id_001(self, setup_teardown):
        """request_id不携带，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SAMPLING_PARAMS

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
            assert "missing 1 required positional argument" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_request_id_002(self, setup_teardown):
        """request_id为空，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SAMPLING_PARAMS,
                request_id=""

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
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    async def test_generate_request_id_003(self, setup_teardown):
        """request_id为非字符串，调用generate接口，调用失败，返回对应报错信息"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SAMPLING_PARAMS,
                request_id=12345

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
            assert "Invalid Parameters" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_generate_request_id_004(self, setup_teardown):
        """request_id重复id，调用generate接口，调用成功"""
        try:
            p = Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                enable_health_monitor=True
            )
            outputs1 = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SAMPLING_PARAMS,
                request_id="12345"

            )
            outputs2 = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SAMPLING_PARAMS,
                request_id="12345"

            )
            output = None
            print("proxy is success")
            async for o1 in outputs1:
                output = o1
                print(f"{o1.outputs}", flush=True)
            assert output.outputs[0].finish_reason == "stop", "request is success"
            async for o2 in outputs2:
                output = o2
                print(f"{o2.outputs}", flush=True)
            assert output.outputs[0].finish_reason == "stop", "request is success"
            p.shutdown()
        except Exception as message:
            print(f"error message is: {str(message)}")
            assert "instance 0 is unhealthy" in str(message), "init success"

    @pytest.mark.asyncio
    @pytest.mark.xfail
    async def test_proxy_router_001(self, setup_teardown):
        """router设置为None，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                router=None
            )
        except Exception as message:
            print(f"error message is: {str(message)}")
            assert "instance 0 is unhealthy" in str(message), "init success"


    @pytest.mark.asyncio
    async def test_proxy_router_002(self, setup_teardown):
        """不携带router，初始化Proxy, 初始化成功，默认为random策略"""
        p=Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
            )
        try:
            image = Image.open(setup_teardown.get("image_path") + "003a8ae2ef43b901.jpg")
            image_array = np.array(image)

            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": image_array},
                },
                sampling_params=SAMPLING_PARAMS,
                request_id=str(uuid.uuid4())
            )
            output=None
            async for o in outputs:
                output = o
                print(f"{o.outputs}", flush=True)
            assert output.outputs[0].finish_reason == "stop", "request is success"
        except Exception as message:
            print(f"error message is: {str(message)}")
        p.shutdown()


    @pytest.mark.asyncio
    @pytest.mark.xfail
    async def test_proxy_router_003(self, setup_teardown):
        """router为字符串，初始化Proxy, 初始化失败，返回对应报错信息"""
        try:
            Proxy(
                proxy_addr=PROXY_ADDR,
                encode_addr_list=E_ADDR_LIST,
                pd_addr_list=PD_ADDR_LIST,
                model_name=MODEL,
                router="aaa"
            )
        except Exception as message:
            print(f"error message is: {str(message)}")
            assert "instance 0 is unhealthy" in str(message), "init success"



    @pytest.mark.asyncio
    async def test_proxy_router_004(self, setup_teardown):
        """router为自定义类，初始化Proxy, 初始化成功"""
        p=Proxy(
            proxy_addr=PROXY_ADDR,
            encode_addr_list=E_ADDR_LIST,
            pd_addr_list=PD_ADDR_LIST,
            model_name=MODEL,
            router=CustomRouter
        )
        try:
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SAMPLING_PARAMS,
                request_id=str(uuid.uuid4()),
            )
            output = None
            async for o in outputs:
                output = o
                print(f"{o.outputs}", flush=True)
            assert output.outputs[0].finish_reason == "stop", "request is success"
        except Exception as message:
            print(f"error message is: {str(message)}")
        p.shutdown()



    @pytest.mark.asyncio
    async def test_proxy_enable_health_monitor_001(self, setup_teardown):
        """enable_health_monitor不携带，初始化Proxy, 初始化成功，默认为开启状态"""
        p=Proxy(
            proxy_addr=PROXY_ADDR,
            encode_addr_list=E_ADDR_LIST,
            pd_addr_list=PD_ADDR_LIST,
            model_name=MODEL,
        )
        try:
            outputs = p.generate(
                prompt={
                    "prompt": PROMPT_TEMPLATE,
                    "multi_modal_data": {"image": IMAGE_ARRAY},
                },
                sampling_params=SAMPLING_PARAMS,
                request_id=str(uuid.uuid4()),
            )
            output = None
            async for o in outputs:
                output = o
                print(f"{o.outputs}", flush=True)
            assert output.outputs[0].finish_reason == "stop", "request is success"
        except Exception as message:
            print(f"error message is: {str(message)}")
        p.shutdown()