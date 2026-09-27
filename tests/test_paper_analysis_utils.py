import json
import pytest
from src.analysis.aggregate_common_relaxation import bootstrap_mean_ci, summarize
from src.analysis.compare_paper_conditions import compare_conditions, parse_condition
from src.analysis.linear_probe import parse_checkpoint_selector

def test_bootstrap_and_single_seed():
    assert bootstrap_mean_ci([1,2,3],100,7)==bootstrap_mean_ci([1,2,3],100,7)
    assert summarize([2],100,7)["mean_ci_95"] is None and summarize([1,2,3],100,7)["median"]==2
def test_checkpoint_selection():
    assert parse_checkpoint_selector("final",(0,10))==10 and parse_checkpoint_selector("relaxation-step:0",(0,10))==0
    with pytest.raises(ValueError): parse_checkpoint_selector("bad",(0,10))
def test_condition_parsing_duplicates_and_seed_overlap(tmp_path):
    assert parse_condition("a=x.json")[0]=="a"; path=tmp_path/"s.json";path.write_text(json.dumps({"seeds":[1,2],"matched_endpoint":{}}))
    rows,details=compare_conditions([f"a={path}",f"b={path}"]);assert details["overlapping_seeds"]["a__b"]==[1,2]
    with pytest.raises(ValueError,match="Duplicate"): compare_conditions([f"a={path}",f"a={path}"])
