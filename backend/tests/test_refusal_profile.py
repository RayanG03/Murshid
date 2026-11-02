import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('refusal_profile',Path(__file__).resolve().parents[1]/'apply_refusal_profile.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_profile_preserves_provider_keys_and_source_guards():
    original='# اختبار\nDEEPINFRA_API_KEY=fake_secret\nVECTOR_BACKEND=flat\nEVIDENCE_MODE=documents_only\nRETRIEVAL_MODE=dense\nRETRIEVAL_MODE=lexical\n'
    updated=module.update_text(original,module.PROFILES['hybrid'])
    assert 'DEEPINFRA_API_KEY=fake_secret\n' in updated
    assert 'VECTOR_BACKEND=flat\n' in updated and 'EVIDENCE_MODE=documents_only\n' in updated
    assert updated.count('RETRIEVAL_MODE=hybrid\n')==2
    assert 'RERANK_REFUSE_THRESHOLD' not in updated
    assert module.update_text(updated,module.PROFILES['hybrid'])==updated


def test_atomic_replacement_keeps_no_temp_files(tmp_path):
    env=tmp_path/'.env';env.write_bytes(b'old')
    module.replace_bytes(env,'قيم جديدة'.encode('utf-8'))
    assert env.read_text()=='قيم جديدة'
    assert list(tmp_path.iterdir())==[env]


def test_cli_backup_and_exact_restore_do_not_print_secret(tmp_path):
    import subprocess
    import sys
    script=tmp_path/'apply_refusal_profile.py'
    script.write_bytes(Path(module.__file__).read_bytes())
    env=tmp_path/'.env'
    original=b'DEEPINFRA_API_KEY=fake_private_value\r\nVECTOR_BACKEND=flat\r\nEVIDENCE_MODE=documents_only\r\n'
    env.write_bytes(original)
    applied=subprocess.run([sys.executable,str(script),'--profile','hybrid'],capture_output=True,text=True,check=True)
    assert 'fake_private_value' not in applied.stdout+applied.stderr
    assert 'RETRIEVAL_MODE=hybrid' in env.read_text()
    backup=next((tmp_path/'data'/'config_backups').iterdir())
    assert backup.read_bytes()==original
    subprocess.run([sys.executable,str(script),'--restore',str(backup)],capture_output=True,text=True,check=True)
    assert env.read_bytes()==original
