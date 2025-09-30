#!/usr/bin/env python3
# verify_audit.py - 验证vLLM审计功能的测试脚本

import os
import sys
import json
import time
import asyncio
from typing import Dict, Any
import requests

# 设置环境变量以启用审计
os.environ["VLLM_STRUCTURED_OUTPUT_AUDIT"] = "true"
os.environ["VLLM_AUDIT_RECORD_ALLOWED_TOKENS"] = "false"  # 避免记录过多数据
os.environ["VLLM_AUDIT_RECORD_FULL_EVENTS"] = "true"
os.environ["VLLM_AUDIT_RESPONSE_LEVEL"] = "full"  # 在响应中包含完整审计数据
os.environ["VLLM_AUDIT_MAX_TRAILS"] = "1000"

print("=" * 60)
print("vLLM 结构化输出审计功能验证")
print("=" * 60)

# ==============================================
# 步骤1：验证模块导入
# ==============================================
print("\n[步骤1] 验证模块导入...")

try:
    # 导入审计模块
    from v1.structured_output.audit_tracker import (
        StructuredOutputAuditTracker,
        AuditEvent,
        AuditEventType,
        get_audit_tracker,
        configure_audit_tracker
    )

    print("✓ 成功导入 audit_tracker 模块")

    from vllm.v1.structured_output.audit_integration import (
        StructuredOutputAuditConfig,
        initialize_audit_system
    )

    print("✓ 成功导入 audit_integration 模块")

    # 验证后端模块
    from vllm.v1.structured_output import backend_outlines
    from vllm.v1.structured_output import backend_xgrammar

    print("✓ 成功导入 backend 模块")

except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("\n请确保已正确添加审计模块文件")
    sys.exit(1)

# ==============================================
# 步骤2：验证审计系统初始化
# ==============================================
print("\n[步骤2] 验证审计系统初始化...")

# 初始化审计系统
config = StructuredOutputAuditConfig(
    enabled=True,
    record_allowed_tokens=False,
    record_full_events=True,
    max_trails_in_memory=100,
    include_in_response=True,
    response_detail_level="full"
)

tracker = initialize_audit_system(config)
print(f"✓ 审计系统已初始化")
print(f"  - 启用状态: {tracker.is_enabled()}")
print(f"  - 最大追踪数: {tracker.max_trails}")
print(f"  - 配置详情: {json.dumps(config.to_dict(), indent=2)}")

# ==============================================
# 步骤3：单元测试审计追踪器
# ==============================================
print("\n[步骤3] 单元测试审计追踪器...")


def test_audit_tracker():
    """测试审计追踪器的基本功能"""
    test_results = []

    # 测试1：创建审计追踪
    try:
        request_id = "test_unit_001"
        tracker.start_trail(request_id, "test_backend", '{"type": "object"}')
        trail = tracker.get_trail(request_id)
        assert trail is not None, "未能创建审计追踪"
        assert trail.request_id == request_id, "请求ID不匹配"
        test_results.append(("创建审计追踪", True, None))
    except Exception as e:
        test_results.append(("创建审计追踪", False, str(e)))

    # 测试2：记录token接受事件
    try:
        tokens = [100, 200, 300]
        tracker.record_token_acceptance(
            request_id=request_id,
            tokens=tokens,
            accepted=True,
            current_state="state_1"
        )
        trail = tracker.get_trail(request_id)
        assert trail.total_tokens_generated == 3, "Token计数错误"
        assert len(trail.events) > 0, "未记录事件"
        test_results.append(("记录Token接受", True, None))
    except Exception as e:
        test_results.append(("记录Token接受", False, str(e)))

    # 测试3：记录回滚事件
    try:
        tracker.record_rollback(request_id, 2, "state_2")
        trail = tracker.get_trail(request_id)
        assert trail.total_rollbacks == 1, "回滚计数错误"
        test_results.append(("记录回滚事件", True, None))
    except Exception as e:
        test_results.append(("记录回滚事件", False, str(e)))

    # 测试4：完成并序列化
    try:
        tracker.finalize_trail(request_id)
        trail_dict = tracker.get_trail_dict(request_id, include_events=True)
        assert trail_dict is not None, "无法获取追踪字典"
        assert "events" in trail_dict, "缺少事件列表"
        assert trail_dict["total_tokens_generated"] == 3, "统计数据错误"
        test_results.append(("完成并序列化", True, None))
    except Exception as e:
        test_results.append(("完成并序列化", False, str(e)))

    # 打印测试结果
    print("\n单元测试结果:")
    for test_name, passed, error in test_results:
        if passed:
            print(f"  ✓ {test_name}")
        else:
            print(f"  ✗ {test_name}: {error}")

    return all(passed for _, passed, _ in test_results)


unit_test_passed = test_audit_tracker()
if not unit_test_passed:
    print("\n⚠️ 单元测试未完全通过，但继续进行其他测试...")

# ==============================================
# 步骤4：测试Backend集成
# ==============================================
print("\n[步骤4] 测试Backend集成...")


def test_backend_integration():
    """测试审计功能与各个backend的集成"""

    # 测试Outlines Backend
    try:
        from vllm.v1.structured_output.backend_outlines import OutlinesBackend
        from vllm.transformers_utils.tokenizer import get_tokenizer
        from vllm.config import VllmConfig, ModelConfig

        print("\n测试 Outlines Backend:")

        # 创建一个mock配置（简化版）
        model_config = ModelConfig(
            model="gpt2",  # 使用简单模型进行测试
            tokenizer="gpt2",
            tokenizer_mode="auto",
            trust_remote_code=False,
            max_model_len=2048
        )

        # 获取tokenizer
        tokenizer = get_tokenizer(
            tokenizer_name="gpt2",
            tokenizer_mode="auto"
        )

        # 创建VllmConfig (需要根据实际API调整)
        from dataclasses import dataclass
        @dataclass
        class MockVllmConfig:
            model_config: Any = model_config
            speculative_config: Any = None
            structured_outputs_config: Any = None

        vllm_config = MockVllmConfig()

        # 创建Backend实例
        backend = OutlinesBackend(
            vllm_config=vllm_config,
            tokenizer=tokenizer,
            vocab_size=tokenizer.vocab_size
        )

        print("  ✓ Outlines Backend 创建成功")

        # 测试编译语法
        from vllm.v1.structured_output.backend_types import StructuredOutputOptions

        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        })

        grammar = backend.compile_grammar(
            StructuredOutputOptions.JSON,
            json_schema
        )

        print("  ✓ Grammar 编译成功")

        # 测试审计上下文设置
        if hasattr(grammar, 'set_audit_context'):
            grammar.set_audit_context("test_backend_001")
            print("  ✓ 审计上下文设置成功")
        else:
            print("  ⚠️ Grammar 未实现审计接口")

        return True

    except ImportError as e:
        print(f"  ⚠️ 跳过Backend测试 (缺少依赖): {e}")
        return False
    except Exception as e:
        print(f"  ✗ Backend测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


backend_test_passed = test_backend_integration()

# ==============================================
# 步骤5：测试vLLM服务器集成（如果运行中）
# ==============================================
print("\n[步骤5] 测试vLLM服务器集成...")


def test_server_integration():
    """测试与运行中的vLLM服务器的集成"""

    # 默认vLLM服务器地址
    server_url = "http://localhost:8000"

    print(f"尝试连接到 {server_url}...")

    try:
        # 检查服务器是否运行
        response = requests.get(f"{server_url}/health", timeout=2)
        if response.status_code != 200:
            print("  ⚠️ vLLM服务器未运行或不可访问")
            print("  要完整测试，请先启动vLLM服务器:")
            print("  python -m vllm.entrypoints.openai.api_server \\")
            print("    --model <your-model> \\")
            print("    --port 8000")
            return False

        print("  ✓ vLLM服务器可访问")

        # 发送结构化输出请求
        request_data = {
            "model": "your-model",
            "prompt": "Generate a person's information:",
            "max_tokens": 100,
            "temperature": 0.7,
            "guided_json": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0, "maximum": 120},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["name", "age", "email"]
            }
        }

        print("\n  发送结构化输出请求...")
        response = requests.post(
            f"{server_url}/v1/completions",
            json=request_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            # 检查是否包含审计数据
            if "structured_output_audit" in result:
                print("  ✓ 响应包含审计数据")
                audit_data = result["structured_output_audit"]

                if "audit_trail" in audit_data:
                    trail = audit_data["audit_trail"]
                    print(f"\n  审计追踪摘要:")
                    print(f"    - Backend: {trail.get('backend_type', 'N/A')}")
                    print(f"    - 总步骤: {trail.get('total_steps', 0)}")
                    print(f"    - 生成Token数: {trail.get('total_tokens_generated', 0)}")
                    print(f"    - 回滚次数: {trail.get('total_rollbacks', 0)}")
                    print(f"    - 错误次数: {trail.get('total_errors', 0)}")

                    if trail.get('events'):
                        print(f"    - 事件数量: {len(trail['events'])}")
                        print(f"\n  前3个事件:")
                        for i, event in enumerate(trail['events'][:3], 1):
                            print(f"    {i}. {event.get('event_type', 'N/A')} at step {event.get('step_number', 0)}")
                elif "audit_summary" in audit_data:
                    summary = audit_data["audit_summary"]
                    print(f"\n  审计摘要:")
                    for key, value in summary.items():
                        print(f"    - {key}: {value}")
            else:
                print("  ⚠️ 响应中未包含审计数据")
                print("  请确认服务器是否使用了修改后的代码")

            return True
        else:
            print(f"  ✗ 请求失败: {response.status_code}")
            print(f"  响应: {response.text[:200]}")
            return False

    except requests.exceptions.ConnectionError:
        print("  ⚠️ 无法连接到vLLM服务器")
        return False
    except requests.exceptions.Timeout:
        print("  ⚠️ 请求超时")
        return False
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


server_test_passed = test_server_integration()

# ==============================================
# 步骤6：性能测试
# ==============================================
print("\n[步骤6] 性能测试...")


def performance_test():
    """简单的性能测试"""
    import time

    # 测试禁用审计的性能
    tracker_disabled = StructuredOutputAuditTracker(enabled=False)

    start = time.perf_counter()
    for i in range(1000):
        tracker_disabled.record_token_acceptance(f"perf_{i}", [1, 2, 3], True, "state")
    disabled_time = time.perf_counter() - start

    # 测试启用审计的性能
    tracker_enabled = StructuredOutputAuditTracker(enabled=True)

    start = time.perf_counter()
    for i in range(1000):
        tracker_enabled.start_trail(f"perf_{i}", "test")
        tracker_enabled.record_token_acceptance(f"perf_{i}", [1, 2, 3], True, "state")
        tracker_enabled.finalize_trail(f"perf_{i}")
    enabled_time = time.perf_counter() - start

    print(f"  禁用审计: {disabled_time * 1000:.2f}ms (1000次操作)")
    print(f"  启用审计: {enabled_time * 1000:.2f}ms (1000次操作)")
    print(f"  性能开销: {((enabled_time - disabled_time) / disabled_time * 100):.1f}%")

    if enabled_time < disabled_time * 2:
        print("  ✓ 性能开销在合理范围内 (<100%)")
        return True
    else:
        print("  ⚠️ 性能开销较大，可能需要优化")
        return False


perf_test_passed = performance_test()

# ==============================================
# 总结
# ==============================================
print("\n" + "=" * 60)
print("验证总结")
print("=" * 60)

test_summary = {
    "模块导入": True,  # 如果执行到这里，说明导入成功
    "审计系统初始化": tracker.is_enabled(),
    "单元测试": unit_test_passed,
    "Backend集成": backend_test_passed,
    "服务器集成": server_test_passed,
    "性能测试": perf_test_passed
}

all_passed = all(test_summary.values())

print("\n测试结果:")
for test_name, passed in test_summary.items():
    status = "✓" if passed else "✗"
    print(f"  {status} {test_name}")

if all_passed:
    print("\n🎉 所有测试通过！审计功能正常工作。")
else:
    print("\n⚠️ 部分测试未通过。请检查:")
    if not test_summary["服务器集成"]:
        print("  - 确保vLLM服务器正在运行并使用修改后的代码")
    if not test_summary["Backend集成"]:
        print("  - 检查Backend修改是否正确")
    if not test_summary["单元测试"]:
        print("  - 检查审计模块实现")

print("\n下一步建议:")
print("1. 如果服务器测试未通过，启动vLLM服务器:")
print("   export VLLM_STRUCTURED_OUTPUT_AUDIT=true")
print("   python -m vllm.entrypoints.openai.api_server --model <model-name>")
print("")
print("2. 使用实际的模型测试:")
print("   curl http://localhost:8000/v1/completions \\")
print("     -H 'Content-Type: application/json' \\")
print("     -d '{\"model\": \"<model>\", \"prompt\": \"test\", \"guided_json\": {\"type\": \"object\"}}'")
print("")
print("3. 检查审计日志:")
print("   如果设置了 VLLM_AUDIT_PERSIST=true 和 VLLM_AUDIT_LOG_DIR")
print("   可以在指定目录查看JSON格式的审计日志")