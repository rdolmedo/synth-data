"""synth.prefix_bases: exact install-prefix splitting and Dockerfile rewriting."""
import unittest

from synth import prefix_bases as pb

APT = """FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates curl git tar \\
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app && echo hi > /app/x.txt
WORKDIR /app
CMD ["/bin/bash"]
"""


class Split(unittest.TestCase):
    def test_package_names_like_git_curl_tar_do_not_end_the_prefix(self):
        key, end, stock, reason = pb.split_prefix(APT)
        self.assertEqual((stock, reason), ('ubuntu:22.04', 'prefix'))
        self.assertEqual(APT.split('\n')[end - 1], 'RUN mkdir -p /app && echo hi > /app/x.txt')
        self.assertEqual(pb.apt_packages(key), ['ca-certificates', 'curl', 'git', 'tar'])

    def test_whitespace_and_line_breaks_do_not_split_groups(self):
        other = APT.replace('ca-certificates curl git tar \\\n', 'ca-certificates  curl git tar \\\n')
        self.assertEqual(pb.split_prefix(APT)[0], pb.split_prefix(other)[0])

    def test_task_writes_redirections_and_conditionals_are_not_install_only(self):
        for cmd in ('mkdir -p /app', 'echo deb x > /etc/apt/sources.list.d/x.list', 'apt-get install -y x && cd /app',
                    'if true; then apt-get install -y x; fi', 'pip install -r /tmp/requirements.txt && python3 -c "1"',
                    'export X=1 && apt-get install -y x', 'apt-get install -y $(cat /pkgs)'):
            self.assertFalse(pb.install_only(pb.shell_tokens(cmd)), cmd)
        for cmd in ('apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*',
                    'pip3 install --no-cache-dir pytest==8.4.1 requests', 'python3 -m pip install pyyaml',
                    'curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && apt-get install -y nodejs',
                    'DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata 2>/dev/null', 'npm install -g typescript'):
            self.assertTrue(pb.install_only(pb.shell_tokens(cmd)), cmd)

    def test_no_install_prefix_and_multistage(self):
        self.assertEqual(pb.split_prefix('FROM python:3.11-slim\nRUN mkdir /app\n')[3], 'no install prefix')
        self.assertEqual(pb.split_prefix('FROM a AS b\nRUN x\nFROM c\nCOPY --from=b / /\n')[3], 'multi-stage or global ARG')

    def test_rewrite_keeps_the_task_part_verbatim(self):
        _, end, _, _ = pb.split_prefix(APT)
        out = pb.rewrite(APT, 'local/rts-prefix-abc:1', end, 'abc')
        lines = out.split('\n')
        self.assertTrue(lines[0].startswith('# synth.prefix_bases'))
        self.assertEqual(lines[1], 'FROM local/rts-prefix-abc:1')
        self.assertEqual('\n'.join(lines[2:]), '\n'.join(APT.split('\n')[end - 1:]))


class RewardFallback(unittest.TestCase):
    def test_inserted_after_the_shebang_once(self):
        text = '#!/bin/bash\nset -euo pipefail\npython3 -m pytest /tests/t.py\necho 1 > /logs/verifier/reward.txt\n'
        out = pb.add_reward_fallback(text)
        self.assertTrue(out.startswith('#!/bin/bash\n\n# harbor-reward-fallback'))
        self.assertTrue(out.endswith(text.split('\n', 1)[1]))
        self.assertLess(out.index('trap _harbor_write_missing_reward EXIT'), out.index('set -euo pipefail'))
        self.assertIsNone(pb.add_reward_fallback(out))
        self.assertIsNone(pb.add_reward_fallback('#!/bin/sh\ntrap cleanup EXIT\n'))


if __name__ == '__main__':
    unittest.main()
