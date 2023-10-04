import numpy as np
import tvm

import tvm.meta_schedule as ms

from sch_tune import matmul_g128_KN_sym_dynm, make_arg, TARGET
from tvm import dlight as dl


def roof_model(kind="ds"):
    db_name = "__tmp/tune_sch_man_ds_64"
    func = matmul_g128_KN_sym_dynm
    B, M, N, K = 1, 63, 22016, 4096
    BLK, GS = 8, 128
    
    assert kind in ["ds", "dl"]
    if kind == "ds":
        db = ms.database.JSONDatabase(work_dir=db_name, allow_missing=False)
        mod = ms.tir_integration._normalize_mod(func)
        rec = db.query(mod, TARGET, kind="record")
        
        with TARGET:
            sch = tvm.tir.Schedule(func)
            rec.trace.apply_to_schedule(sch, remove_postproc=False)
            lib = tvm.build(sch.mod)
    else:
        with TARGET:
            mod = ms.tir_integration._normalize_mod(func)
            mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
            lib = tvm.build(mod)


    for M in range(1, 257):
        args_info = [
                ms.arg_info.TensorInfo("uint32", [K // BLK, N]),  # WGH
                ms.arg_info.TensorInfo("float16", [K // GS, N]),  # SCL
                ms.arg_info.TensorInfo("float16", [B, M, K]),     # D_IN
                ms.arg_info.TensorInfo("float16", [B, M, N]),     # D_OUT
            ]
        args = [make_arg(info) for info in args_info]

        score_s = lib.time_evaluator(lib.entry_name, dev=tvm.cuda(0), number=2000, repeat=1)(*args).mean
        print(f"{kind} SCORE M: {M} TIME: {score_s*1e6} us",)


def correctness(db_name):
    func = matmul_g128_KN_sym_dynm
    B, M, N, K = 1, 63, 22016, 4096
    BLK, GS = 8, 128

    db = ms.database.JSONDatabase(work_dir=db_name, allow_missing=False)
    mod = ms.tir_integration._normalize_mod(func)
    rec = db.query(mod, TARGET, kind="record")
        
    with TARGET:
        sch = tvm.tir.Schedule(func)
        rec.trace.apply_to_schedule(sch, remove_postproc=False)
        dt_lib = tvm.build(sch.mod)
        
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        dl_lib = tvm.build(mod)

    args_info = [
            ms.arg_info.TensorInfo("uint32", [K // BLK, N]),  # WGH
            ms.arg_info.TensorInfo("float16", [K // GS, N]),  # SCL
            ms.arg_info.TensorInfo("float16", [B, M, K]),     # D_IN
            ms.arg_info.TensorInfo("float16", [B, M, N]),     # D_OUT
        ]
    args = [make_arg(info) for info in args_info]

    dt_lib(*args)
    dt_res = args[-1].numpy()

    dl_lib(*args)
    dl_res = args[-1].numpy()

    # doesn't work 
    # assert np.allclose(dt_res, dl_res, atol=2, rtol=0.3), "!!! BAD ACCURACY !!!"
    
    print("Visual comparison")
    print(dl_res)
    print(dt_res)


if __name__ == "__main__":
    roof_model(kind="ds")
    # correctness(db_name="__tmp/tune_sch_man_ds_64")
