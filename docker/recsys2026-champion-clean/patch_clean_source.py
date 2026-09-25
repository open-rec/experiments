"""Patch the upstream exporter to stop after strict-holdout prediction."""

from pathlib import Path


path = Path("/workspace/src/basic_candidate_generators/launchers_dro_oneshot/retrain_and_export_oneshot_ckpt.py")
text = path.read_text()
arg_anchor = '    p.add_argument("--skip_checkpoints", action="store_true")\n'
arg_patch = arg_anchor + '    p.add_argument("--stop_after_holdout", action="store_true")\n'
run_anchor = ('    _save(run_inference_dispatch(rec_nh, holdout, args.top_n, inf_mode, track_meta),\n'
              '          out_dir / "holdout_candidates.parquet",\n'
              '          gt_parquet=splitk_dir / "holdout_test.parquet")\n')
run_patch = run_anchor + '    if args.stop_after_holdout:\n        return\n'
if "--stop_after_holdout" not in text:
    if arg_anchor not in text or run_anchor not in text:
        raise RuntimeError("upstream exporter anchors changed")
    path.write_text(text.replace(arg_anchor, arg_patch).replace(run_anchor, run_patch))
print(path)

bert = Path("/workspace/src/bert4rec/src/basic_candidate_generators/launchers_crossvalidation/retrain_and_export.py")
text = bert.read_text()
arg_anchor = ('    p.add_argument("--skip_holdout_candidates", action="store_true",\n'
              '                   help="Skip fitting on non-holdout and inferring top_k candidates on holdout_test.")\n')
arg_patch = arg_anchor + '    p.add_argument("--stop_after_holdout", action="store_true")\n'
run_anchor = ('        _score_holdout(hc_path, non_holdout, min_turn=best_params.get("eval_min_turn", 2))\n')
run_patch = run_anchor + '        if args.stop_after_holdout:\n            return\n'
if "--stop_after_holdout" not in text:
    if arg_anchor not in text or run_anchor not in text:
        raise RuntimeError("upstream BERT exporter anchors changed")
    bert.write_text(text.replace(arg_anchor, arg_patch).replace(run_anchor, run_patch))
print(bert)

standard = Path("/workspace/src/basic_candidate_generators/launchers_crossvalidation/retrain_and_export.py")
text = standard.read_text()
arg_anchor = ('    p.add_argument("--skip_holdout_candidates", action="store_true",\n'
              '                   help="Skip fitting on non-holdout and inferring top_k candidates on holdout_test.")\n')
arg_patch = arg_anchor + '    p.add_argument("--stop_after_holdout", action="store_true")\n'
run_anchor = ('        _infer_multiturn(rec_nh, holdout_df, top_k, inference_mode, track_meta,\n'
              '                         out_dir / "datasets" / "holdout_candidates.parquet",\n'
              '                         "non_holdout→holdout",\n'
              '                         query_bundle=_qbundle("holdout"))\n')
run_patch = run_anchor + '        if args.stop_after_holdout:\n            return\n'
if "--stop_after_holdout" not in text:
    if arg_anchor not in text or run_anchor not in text:
        raise RuntimeError("upstream standard exporter anchors changed")
    standard.write_text(text.replace(arg_anchor, arg_patch).replace(run_anchor, run_patch))
print(standard)

dro = Path("/workspace/src/basic_candidate_generators/launchers_dro/retrain_and_export_dro.py")
text = dro.read_text()
arg_anchor = '    p.add_argument("--skip_holdout_candidates", action="store_true")\n'
arg_patch = arg_anchor + '    p.add_argument("--stop_after_holdout", action="store_true")\n'
shim_anchor = '    shim.skip_holdout_candidates = args.skip_holdout_candidates\n'
shim_patch = shim_anchor + '    shim.stop_after_holdout = args.stop_after_holdout\n'
if "--stop_after_holdout" not in text:
    if arg_anchor not in text or shim_anchor not in text:
        raise RuntimeError("upstream DRO exporter anchors changed")
    dro.write_text(text.replace(arg_anchor, arg_patch).replace(shim_anchor, shim_patch))
print(dro)

heuristic = Path("/workspace/src/heuristic/scripts/launchers/gambling_updated.py")
text = heuristic.read_text()
arg_anchor = '    parser.add_argument("--query-cache-dir", type=Path, default=None)\n'
arg_patch = arg_anchor + (
    '    parser.add_argument("--query-cache-includes-goal", action="store_true",\n'
    '                        help="Use conversation_goal when rebuilding cached query text.")\n'
)
goal_anchor = ('    def _row_no_goal(_r):\n'
               '        if _r.get("conversation_goal") is None:\n')
goal_patch = ('    def _row_no_goal(_r):\n'
              '        if args.query_cache_includes_goal:\n'
              '            return _r\n'
              '        if _r.get("conversation_goal") is None:\n')
if "--query-cache-includes-goal" not in text:
    if arg_anchor not in text or goal_anchor not in text:
        raise RuntimeError("upstream heuristic anchors changed")
    heuristic.write_text(text.replace(arg_anchor, arg_patch).replace(goal_anchor, goal_patch))
print(heuristic)

text = heuristic.read_text()
arg_anchor = ('    parser.add_argument("--query-cache-includes-goal", action="store_true",\n'
              '                        help="Use conversation_goal when rebuilding cached query text.")\n')
arg_patch = arg_anchor + (
    '    parser.add_argument("--skip-query-text-validation", action="store_true",\n'
    '                        help="Trust key-aligned cached vectors from another query template.")\n'
)
sig_anchor = ('    expected_instructions: list[str] | None = None,\n'
              ') -> np.ndarray:\n')
sig_patch = ('    expected_instructions: list[str] | None = None,\n'
             '    validate_query_text: bool = True,\n'
             ') -> np.ndarray:\n')
check_anchor = '        if query_texts[i] != rec["query_text"]:\n'
check_patch = '        if validate_query_text and query_texts[i] != rec["query_text"]:\n'
call_anchor = ('        args.query_cache_dir, tasks, query_texts,\n'
               '        expected_instructions=expected_instructions,\n'
               '    )\n')
call_patch = ('        args.query_cache_dir, tasks, query_texts,\n'
              '        expected_instructions=expected_instructions,\n'
              '        validate_query_text=not args.skip_query_text_validation,\n'
              '    )\n')
if "--skip-query-text-validation" not in text:
    if any(anchor not in text for anchor in (arg_anchor, sig_anchor, check_anchor, call_anchor)):
        raise RuntimeError("upstream heuristic validation anchors changed")
    heuristic.write_text(text.replace(arg_anchor, arg_patch)
                         .replace(sig_anchor, sig_patch)
                         .replace(check_anchor, check_patch)
                         .replace(call_anchor, call_patch, 1))
print(heuristic)
