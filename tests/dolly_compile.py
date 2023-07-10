import numpy as np

import tvm
from tvm.target import Target
from tvm import meta_schedule as ms
from tvm.script import tir as T

import tvm.tir.tensor_intrin.cuda

from dolly_tune import get_matmul_int4_dyn_m, apply_trace_int_v1
from dolly_bench import find_best_dyn_m
from dolly_collection import get_static_mamtmul


def main():
    configs = [
    #    M   N      K      G    Schedule/db_name
        (32, 15360, 5120,  40,  "dolly_tune_all_1"),
        (32, 5120,  5120,  40,  "dolly_tune_all_1"),
        (32, 20480, 5120,  40,  "dolly_tune_all_1"),
        (32, 5120,  20480, 160, "dolly_tune_all_1"),
    ]
    
    dev = tvm.cuda(0)
    target = Target("nvidia/nvidia-a10g")
    top_k_to_analize = 40

    funcs = {}
    # Find dyn_M schedules
    for M, N, K, G, sch_provider in configs:
        if isinstance(sch_provider, str):
            db = ms.database.JSONDatabase(work_dir=sch_provider, allow_missing=False)

            best_dyn_m_trace = find_best_dyn_m(M, N, K, G, db=db, top_k=top_k_to_analize, target=target, dev=dev)

            def apply_trace_(sch):
                best_dyn_m_trace.apply_to_schedule(sch, remove_postproc=False)        

            sch_provider = apply_trace_

        dyn_m_mod = get_matmul_int4_dyn_m(N, K, G)
        name = dyn_m_mod.attrs["global_symbol"]
        dyn_m_sch = tvm.tir.Schedule(dyn_m_mod)
        sch_provider(dyn_m_sch)
        funcs[name] = dyn_m_sch.mod["main"]
    
    # Find static_M1 schedules
    for M, N, K, G, sch_provider in configs:
        M = 1
        func = get_static_mamtmul(M, N, K, G, target)
        if func is not None:
            funcs[func.attrs["global_symbol"]] = func

    # Compile
    mod_to_compile = tvm.IRModule(funcs)
    print(mod_to_compile, file=open('impl_a10g_dynm.py', 'w'))
    with target:
        lib = tvm.build(mod_to_compile)
        lib.export_library(f"int4_dolly_ker_tuned_a10g.so")


if __name__ == "__main__":
    main()
