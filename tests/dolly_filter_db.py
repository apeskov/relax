import tvm.meta_schedule as ms

import tvm.tir.tensor_intrin.cuda

from dolly_tune import get_matmul_int4, get_matmul_int4_dyn_m


def main(input_db_name, output_db_name):
    M = 1
    HS = 5120
    configs = [
    #    M   N      K      G
        # (M, HS*3, HS,   40),
        (M, HS,   HS,   40),
        # (M, HS*4, HS,   40),
        # (M, HS,   HS*4, 160),
    ]
    target = tvm.target.Target("nvidia/nvidia-a10g")
    in_db = ms.database.JSONDatabase(work_dir=input_db_name, allow_missing=False)

    for M, N, K, G in configs:
        s_mod = get_matmul_int4(M, N, K, G)
        s_mod = ms.tir_integration._normalize_mod(s_mod)
        rec = in_db.query(s_mod, target=target, kind="record")
        print(rec.trace)



if __name__ == "__main__":
    # main("dolly_tune_all_1", "dolly_filtered_all_1")
    main("dolly_tune_cuda_1", "dolly_filtered_all_1")
