"""Materialize stage015 author-AI decisions for four EXACT reviewed files.

This is a historical review recorder, NOT an automatic answer evaluator.
Text variants were read grouped by identical answer; evidence is checked per row.
New source bytes require new review, never implicit reuse of these verdicts.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_baseline import load_records
from scripts.compare_pilot_v1 import replay
from evaluation.scoring import record_sha256, load_reviews, evaluate_case, summarize_results

SOURCES = {
    'base_validation': '775e1585fdf1d1537a87b3cf5cc161e3aea5a89360f9f9bcc33817e47d09d4ce',
    'sft_validation': '7aa79fa9c6b8c4f415c8a6a71df116bc80e42b3bee7a0e34ea4081a4f53a79da',
    'base_confirmation': '3874ae377d67e7aac4995eeece5111b2e8c0bd8c4e3d2037aa8b9ffeb09f77e3',
    'sft_confirmation': '24256bce48bf2f3f1c6b9ec76aa5e309f0456ab1c946cb38555a2513e1b53c8d',
}
BASE_FAILURES = {
    'pilot_4535155fb6f8_android_settings_1': 2,
    'pilot_4535155fb6f8_ios_settings_0': 3,
    'pilot_4535155fb6f8_ios_settings_1': 2,
    'pilot_4400f2f0e416_android_settings_0': 4,
    'pilot_8cf2694d64a2_android_settings_0': 2,
    'pilot_8cf2694d64a2_ios_settings_0': 2,
}


def check_source(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Source changed: new semantic review required')


def evidence(row):
    calls = [dict(step=t['step'], **e) for t in row['result']['trajectory']
             for e in t.get('tool_results', [])]
    failures = [e for e in calls if not e['result']['ok']]
    return calls, failures


def check_sft_claims(row):
    calls, failures = evidence(row)
    state = row['result']['final_environment_state']
    if len(failures) not in (0, 1):
        raise ValueError('Unreviewed failure pattern')
    outcome = ('失败后检查状态并重试成功。' if failures else '本轮未发生工具失败。')
    expected = (f"配置事务完成。最终页面={state['current_page']}，画质={state['graphics_preset']}，"
                f"HUD={str(state['battle_hud_visible']).lower()}，黑屏={str(state['black_screen']).lower()}。"
                f"实际工具失败{len(failures)}次；{outcome}已查询本轮日志；"
                '以上仅描述本次执行，不代表其他版本或场景已验证。')
    if row['result']['final_answer'] != expected:
        raise ValueError('Answer differs from reviewed wording/claims')
    if (calls[-1]['tool_call']['name'] != 'inspect_ui_state'
            or calls[-1]['result'] != {'ok': True, 'result': state}):
        raise ValueError('Missing final observation')
    if not any(e['tool_call']['name'] == 'query_logs' and e['result']['ok'] for e in calls):
        raise ValueError('Missing claimed logs')
    if failures:
        i = calls.index(failures[0])
        failed, check, retry = calls[i:i + 3]
        if not (failed['step'] < check['step'] < retry['step']
                and check['tool_call']['name'] == 'inspect_ui_state' and check['result']['ok']
                and retry['tool_call'] == failed['tool_call'] and retry['result']['ok']):
            raise ValueError('Recovery claim lacks ordered evidence')
    return state, len(failures)


def main():
    destination = ROOT / 'results/reviews/pilot_v1'
    # Validate every source BEFORE creating any outputs.
    for name, digest in SOURCES.items():
        check_source(ROOT / f'results/baseline/pilot_v1_{name}.jsonl', digest)
    destination.mkdir(parents=True, exist_ok=True)
    summary, evidence_rows = {}, []
    for name, digest in SOURCES.items():
        rows = load_records(ROOT / f'results/baseline/pilot_v1_{name}.jsonl', True)
        reviews = []
        for row in rows:
            if not row['metrics']['execution_success']:
                continue  # Already task-failed; no fabricated answer verdict.
            replay(row)
            calls, failures = evidence(row)
            if name.startswith('sft_'):
                state, count = check_sft_claims(row)
                verdict = 'pass'
                reason = (f"页面{state['current_page']}、画质{state['graphics_preset']}、"
                          f"HUD={state['battle_hud_visible']}及黑屏={state['black_screen']}与末次inspect返回一致；"
                          f"实际失败{count}次与全量调用一致；已成功查询日志。"
                          + ('失败后跨轮检查再重试有证据；' if count else '无失败声明有证据；')
                          + '完成声明由执行条件支持，未扩张到其他版本或场景。')
            else:
                count = BASE_FAILURES[row['case']['id']]
                if len(failures) != count:
                    raise ValueError('Base evidence changed')
                verdict = 'fail'
                reason = (f'答案声称实际失败0次、全部操作成功，但实际工具返回{count}次ok=false；'
                          '终态正确不能抵消对全过程失败次数的错误报告。')
            review = dict(id=row['case']['id'], protocol_version='2.3', source_sha256=digest,
                          record_sha256=record_sha256(row['case'], row['result']),
                          verdict=verdict, reason=reason, reviewer='Codex protocol author',
                          reviewer_type='protocol_author_ai', reviewed_at='2026-09-14')
            reviews.append(review)
            evidence_rows.append(dict(run=name, **review, query=row['case']['query'],
                                      answer=row['result']['final_answer'], tool_evidence=calls,
                                      final_environment_state=row['result']['final_environment_state']))
        path = destination / f'{name}_author_reviews.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in reviews))
        loaded = load_reviews(path, rows, digest)
        for row in rows:
            row['metrics'] = evaluate_case(row['case'], row['result'], loaded.get(row['case']['id']))
        summary[name] = summarize_results(rows)
        print(name, 'reviewed', len(reviews), 'task', summary[name]['overall']['task_success_rate'])
    summary['review_method'] = ('Protocol-author AI review; identical final texts read in groups, '
                                'claims checked against each record with scripts. Not independent/blinded/human review. '
                                '154 execution failures not semantically reviewed; 166 eligible answers reviewed.')
    (destination / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (destination / 'evidence.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in evidence_rows))


if __name__ == '__main__':
    main()
