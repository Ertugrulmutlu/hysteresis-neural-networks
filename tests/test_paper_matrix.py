from src.experiments.run_paper_matrix import build_manifest_entries, generate_seed_config, validate_generated_pair

def base(scenario): return {"experiment":{"seed":1},"train":{"scenario":scenario},"logging":{"run_name":f"pilot_{scenario}_seed1_normnone","save_dir":"results"}}
def test_generation_is_deterministic_and_paired(tmp_path):
    a,b=generate_seed_config(base("SABC"),202,"SABC"),generate_seed_config(base("SBAC"),202,"SBAC");validate_generated_pair(a,b)
    assert a["experiment"]["seed"]==b["experiment"]["seed"]==202
    assert build_manifest_entries(base("SABC"),base("SBAC"),[202],tmp_path)==build_manifest_entries(base("SABC"),base("SBAC"),[202],tmp_path)
