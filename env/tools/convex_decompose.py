import os

env_dir = os.path.dirname(os.path.dirname(__file__))
vhacd_path = os.path.join(env_dir, "third_party", "v-hacd", "app", "build", "TestVHACD")

def convex_decompose(obj_path: str) -> str:
    obj_dir = os.path.dirname(obj_path)
    obj_filename = os.path.basename(obj_path)
    os.system(f"cd {obj_dir} && {vhacd_path} {obj_filename}")
    return os.path.join(obj_dir, "decomp.obj")

coacd_path = os.path.join(env_dir, "third_party", "CoACD", "build", "main")
def convex_decompose_coacd(object_path: str):
    object_dir = os.path.dirname(object_path)
    result_path = os.path.join(object_dir, "coacd_decomposed.obj")
    os.system(f"{coacd_path} -i {object_path} -o {result_path} -k 0.3 -t 0.08 --seed 0")

if __name__ == "__main__":
    convex_decompose("/share1/junyu/HRI/genh2r/env/data/assets/table/table.obj")

"""
python -m env.tools.convex_decompose
"""