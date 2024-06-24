import os

from .convex_decompose import convex_decompose
from .sdf import gen_sdf

def main():
    env_path = os.path.dirname(os.path.dirname(__file__))
    table_dir = os.path.join(env_path, "data", "assets", "table")
    table_path = os.path.join(table_dir, "table.obj")
    convex_decompose(table_path)
    convex_table_path = os.path.join(table_dir, "decomp.obj")
    gen_sdf(convex_table_path)

if __name__ == "__main__":
    main()

"""
python -m env.tools.process_table
"""