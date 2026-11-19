from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import runtime_cache as cache

class RuntimeCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        task=self.root/'adaptive-rejection-sampler';(task/'environment').mkdir(parents=True)
        (task/'instruction.md').write_text('Install R if not already available.\n')
        payload=self.root/'payload';(payload/'apt/pool').mkdir(parents=True)
        (payload/'apt/pool/fake.deb').write_bytes(b'fake-only public runtime package')
        for name in ('Packages','Release','apt.conf','sources.list'):
            (payload/'apt'/name).write_text('fake-only package metadata\n')
        self.entry={'manager':'apt','requested':['r-base-core'],'payload_root':str(payload),
                    'instruction_sha256':cache.sha256(task/'instruction.md'),'image_probe':'native-image-and-dpkg-inventory',
                    'native_rootfs_key':'native-key',
                    'files':{str(p.relative_to(payload)):cache.sha256(p) for p in payload.rglob('*') if p.is_file()}}
        self.manifest=self.root/'manifest.json'
        self.write_manifest()
        self.calls=[];self.uploads=[]
        parent=self
        class Environment:
            environment_dir=task/'environment'
            _main=SimpleNamespace(rootfs=Path('/cache/native-key/rootfs'))
            async def exec(self, command, **kwargs):
                parent.calls.append(command)
                return SimpleNamespace(return_code=0,stdout=parent.entry['image_probe'],stderr='')
            async def upload_dir(self,source,destination):parent.uploads.append((source,destination))
        self.env=Environment()

    def tearDown(self):self.tmp.cleanup()
    def write_manifest(self):
        self.manifest.write_text(json.dumps({'schema_version':1,'tasks':{'adaptive-rejection-sampler':self.entry}}))
        self.pin=cache.sha256(self.manifest)

    async def test_stages_public_bytes_but_does_not_install_or_change_global_package_policy(self):
        result=await cache.prepare(self.env,str(self.manifest),self.pin)
        self.assertEqual(result['agent_env'],{'APT_CONFIG':'/opt/aa-runtime-cache/apt/apt.conf'})
        self.assertEqual(len(self.uploads),1)
        self.assertEqual(result['requested'],['r-base-core'])
        self.assertIn('Install packages',result['notice'])
        self.assertTrue(all('apt-get' not in c and '/etc/apt/apt.conf.d' not in c and '/etc/pip.conf' not in c for c in self.calls))

    async def test_bad_manifest_or_instruction_or_payload_rejected_before_staging(self):
        with self.assertRaises(ValueError):await cache.prepare(self.env,str(self.manifest),'0'*64)
        (self.env.environment_dir.parent/'instruction.md').write_text('changed task')
        with self.assertRaises(ValueError):await cache.prepare(self.env,str(self.manifest),self.pin)
        self.entry['instruction_sha256']=cache.sha256(self.env.environment_dir.parent/'instruction.md');self.write_manifest()
        (Path(self.entry['payload_root'])/'apt/pool/fake.deb').write_bytes(b'tampered')
        with self.assertRaises(ValueError):await cache.prepare(self.env,str(self.manifest),self.pin)
        self.assertFalse(self.uploads)

    async def test_extra_payload_and_wrong_image_rejected(self):
        (Path(self.entry['payload_root'])/'unexpected').write_text('not approved')
        with self.assertRaises(ValueError):await cache.prepare(self.env,str(self.manifest),self.pin)
        (Path(self.entry['payload_root'])/'unexpected').unlink()
        original=self.env.exec
        async def wrong(command,**kwargs):return SimpleNamespace(return_code=0,stdout='wrong image',stderr='')
        self.env.exec=wrong
        with self.assertRaises(ValueError):await cache.prepare(self.env,str(self.manifest),self.pin)
        self.assertFalse(self.uploads)

    async def test_unlisted_task_is_noop(self):
        self.env.environment_dir=self.root/'unlisted/environment'
        self.assertIsNone(await cache.prepare(self.env,str(self.manifest),self.pin))
        self.assertFalse(self.uploads);self.assertFalse(self.calls)

    async def test_pip_is_agent_environment_only(self):
        p=Path(self.entry['payload_root'])/'wheels/fake.whl';p.parent.mkdir();p.write_bytes(b'fake-only wheel')
        self.entry['files']['wheels/fake.whl']=cache.sha256(p)
        self.entry.update(manager='pip',requested=['grpcio==1.73.0']);self.write_manifest()
        result=await cache.prepare(self.env,str(self.manifest),self.pin)
        self.assertEqual(result['agent_env'],{'PIP_NO_INDEX':'1','PIP_FIND_LINKS':'/opt/aa-runtime-cache/wheels'})
        self.assertFalse(any('/etc/pip.conf' in c for c in self.calls))

    async def test_mixed_managers_and_offline_git_transport_are_agent_only(self):
        root=Path(self.entry['payload_root'])
        for rel in ('wheels/fake.whl','cran/Rprofile','cran/src/contrib/PACKAGES','sources/upstream.git/HEAD','sources/upstream.git/config'):
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fake public metadata')
            self.entry['files'][rel]=cache.sha256(p)
        for directory in ('objects','refs'):(root/'sources/upstream.git'/directory).mkdir()
        self.entry['managers']=['apt','pip','cran','source']
        self.entry['upstream_sources']=[{'kind':'git','filename':'upstream.git','url':'https://github.com/public/upstream.git'}]
        self.write_manifest();result=await cache.prepare(self.env,str(self.manifest),self.pin)
        env=result['agent_env']
        self.assertEqual(env['GIT_CONFIG_COUNT'],'2')
        self.assertEqual(env['GIT_CONFIG_VALUE_0'],'https://github.com/public/upstream.git')
        self.assertIn('file:///opt/aa-runtime-cache/sources/upstream.git',env['GIT_CONFIG_KEY_0'])
        self.assertEqual(env['R_PROFILE_USER'],'/opt/aa-runtime-cache/cran/Rprofile')
        self.assertFalse(any('git config' in c for c in self.calls))

    async def test_missing_empty_directory_is_rejected(self):
        root=Path(self.entry['payload_root']);(root/'empty-public-metadata').mkdir()
        self.entry['directories']=[str(p.relative_to(root))for p in root.rglob('*')if p.is_dir()]
        self.write_manifest();(root/'empty-public-metadata').rmdir()
        with self.assertRaisesRegex(ValueError,'directory inventory'):
            await cache.prepare(self.env,str(self.manifest),self.pin)
        self.assertFalse(self.uploads)

    async def test_changed_native_image_is_rejected(self):
        self.env._main=SimpleNamespace(rootfs=Path('/cache/changed-image/rootfs'))
        with self.assertRaises(ValueError):await cache.prepare(self.env,str(self.manifest),self.pin)
        self.assertFalse(self.uploads)

    async def test_missing_manager_metadata_is_rejected(self):
        self.entry['files'].pop('apt/apt.conf');(Path(self.entry['payload_root'])/'apt/apt.conf').unlink();self.write_manifest()
        with self.assertRaises(ValueError):await cache.prepare(self.env,str(self.manifest),self.pin)
        self.assertFalse(self.uploads)


    def test_native_no_submit_pins_cache_and_rejects_policy_change(self):
        from . import run
        bundle=self.root/'bundle.tar';bundle.write_bytes(b'fake-only bundle bytes')
        code=self.root/'code';code.mkdir();(code/'.complete').write_text('{"git_sha":"fake-source"}')
        ep={'served':'fake-model','urls':['http://127.0.0.1:9/v1'],
            'request':{'pi_thinking':'max','overrides':{'temperature':1.0}}}
        args=['--bench','tb21','--run-id','fake-cache','--model','fake-model',
              '--seed-namespace','fake-seed','--bundle',str(bundle),'--attempts','1','--no-submit',
              '--runtime-cache-manifest',str(self.manifest),'--runtime-cache-manifest-sha256',self.pin]
        common=SimpleNamespace(PROTOCOLS={run.HARNESS:{'context_window':131072,'agent_timeout_sec':21600}})
        with mock.patch.object(run,'RUNS_ROOT',self.root/'runs'), \
             mock.patch.object(run.endpoint,'alive',return_value=(False,ep,'fake-only')), \
             mock.patch.object(run,'snapshot_code',return_value=str(code)), \
             mock.patch.object(run,'q35_common',return_value=common), \
             mock.patch.object(run,'bundle_tasks',return_value=['adaptive-rejection-sampler']), \
             mock.patch.object(run,'submit_workers') as submit, \
             mock.patch.object(run,'status',return_value={}):
            run.main(args)
            cfg=json.loads((self.root/'runs/tb21/fake-cache/config.json').read_text())
            self.assertEqual((cfg['runtime_cache_manifest'],cfg['runtime_cache_manifest_sha256']),(str(self.manifest),self.pin))
            self.assertEqual(cfg['task_environment_policy'], {'schema_version': 1, 'overrides': {}})
            self.assertEqual(cfg['task_environment_policy'], cfg['protocol']['task_environment_policy'])
            self.assertEqual(cfg['task_environment_policy_sha256'], cfg['protocol']['task_environment_policy_sha256'])
            self.assertFalse(cfg['agent_internet']);submit.assert_not_called()
            changed=args.copy();changed[-1]='0'*64
            with self.assertRaisesRegex(SystemExit,'cache policy differs'):run.main(changed)
            submit.assert_not_called()

    def test_worker_transports_pin_and_oracle_remains_without_pi_policy(self):
        from .worker import PiEvalWorker,q35
        worker=object.__new__(PiEvalWorker)
        worker.config={'benchmark':'tb21','agent':'pi','agent_internet':False,
                       'runtime_cache_manifest':str(self.manifest),'runtime_cache_manifest_sha256':self.pin}
        with mock.patch.object(q35.Worker,'env',return_value={}):
            env=worker.env('task__a0__r0','task',0)
            self.assertEqual(env['PI_RUNTIME_CACHE_MANIFEST_SHA256'],self.pin)
            self.assertEqual(env['PI_RUNTIME_CACHE_MANIFEST'],str(self.manifest))
            self.assertEqual(env['PI_AGENT_INTERNET'],'0')
            worker.config['agent']='oracle'
            self.assertFalse(any(x.startswith('PI_')for x in worker.env('task__a0__r0','task',0)))

if __name__=='__main__':unittest.main()
