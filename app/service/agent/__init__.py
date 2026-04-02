# 에이전트 자동 등록 — agent/ 하위의 모든 agent.py를 스캔하여 @register 실행
import importlib
import pkgutil
from pathlib import Path


def _auto_discover():
    """service/agent/ 하위에서 **/agent.py 파일을 찾아 자동 import"""
    base_dir = Path(__file__).parent

    for path in base_dir.rglob("agent.py"):
        # 자기 자신(__init__.py가 있는 디렉토리의 agent.py가 아닌, 하위 버전 폴더의 agent.py)
        relative = path.relative_to(base_dir)
        if relative.name == "agent.py" and len(relative.parts) > 1:
            module_path = "service.agent." + ".".join(relative.with_suffix("").parts)
            importlib.import_module(module_path)


_auto_discover()
