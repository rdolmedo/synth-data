"""harbor_ns environment that stages a task's inputs/ before the agent starts (task_inputs.py has the format)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from aa_evals.common.harbor_mswea.ns_env import NsSidecarEnvironment
from aa_evals.common.pi_eval.task_inputs import INPUTS, MANIFEST, RECEIPT, VERSION, load, sha256


class NsTaskInputsEnvironment(NsSidecarEnvironment):
    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        if '__verifier__' in (self.session_id or ''):
            return
        task_dir = Path(self.environment_dir).parent
        m = load(task_dir)
        if m is None:
            return
        t0 = time.time()
        root = task_dir / INPUTS
        for d in m['dirs']:
            await self._mkdirs('/' + d)
        for top in sorted(root.iterdir()) if root.exists() else []:
            if top.is_dir():
                await self.upload_dir(top, '/' + top.name)
            else:
                await self.upload_file(top, '/' + top.name)
        # the sandbox side, file by file: what the agent starts from
        missing = [r for r in m['files'] if not self._host_path('/' + r).is_file()]
        if missing:
            raise RuntimeError(f'task inputs not staged: {missing[:5]}')
        receipt = {'version': VERSION, 'task': task_dir.name, 'n_files': len(m['files']), 'n_dirs': len(m['dirs']),
                   'bytes': sum((root / r).stat().st_size for r in m['files']),
                   'manifest_sha256': sha256(task_dir / MANIFEST), 'seconds': round(time.time() - t0, 2)}
        try:
            (Path(self.trial_paths.trial_dir) / RECEIPT).write_text(json.dumps(receipt, indent=1) + '\n')
        except OSError:
            pass
