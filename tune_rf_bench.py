import numpy as np

import tvm
import tvm.meta_schedule as ms
from tvm import te
from tvm._ffi import get_global_func

import tvm.script.tir as T
from tvm.tir.schedule import BlockRV, Schedule

import tvm.tir.tensor_intrin.cuda

TARGET = tvm.target.Target("nvidia/nvidia-a10g")
DEV = tvm.cuda(0)

# q4f16_3
@T.prim_func
def matmul_g128_KN_sym(lv503: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), lv504: T.Buffer((T.int64(32), T.int64(22016)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv503[v_i // T.int64(8), v_j], lv504[v_i // T.int64(128), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            # Original zp FP16
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv503[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv504[v_i // T.int64(128), v_j]
            # zp Int32 
            # p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv503[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15)) - 7)) * lv504[v_i // T.int64(128), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]


# q4f16_3
@T.prim_func
def matmul_g128_KN_sym_dynm(lv503: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), lv504: T.Buffer((T.int64(32), T.int64(22016)), "float16"), lv1654_hdl: T.handle, var_matmul_intermediate_hdl: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    m = T.int64()
    lv1654 = T.match_buffer(lv1654_hdl, (T.int64(1), m, T.int64(4096)), "float16")
    var_matmul_intermediate = T.match_buffer(var_matmul_intermediate_hdl, (T.int64(1), m, T.int64(22016)), "float16")

    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv503[v_i // T.int64(8), v_j], lv504[v_i // T.int64(128), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            # Original zp FP16
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv503[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv504[v_i // T.int64(128), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), m, T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]


def make_arg(info):
    if info.dtype in ["float16", "float32"]:
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype(info.dtype)
    elif info.dtype in ["int32", "uint32", "int16", "int8"]:
        arr_np = np.random.randint(0, 16, size=info.shape).astype(info.dtype)
    else:
        assert False, f"Unimplemented, dtype={info.dtype}"

    return tvm.nd.array(arr_np, device=DEV)



def read_db():
    # db_path = "__tmp/tune_simple_qmm_xxx"
    db_path = "__tmp/tune_mlc_llm_22016_q4f16_kn_zpf16"
    
    db = ms.database.JSONDatabase(work_dir=db_path, allow_missing=False)

    # M, N, K = 1, 22016, 4096
    # GS = 128
    # func = gen_simple_qmm(M, N, K, GS, dim_dtype="int32")
    func = matmul_g128_KN_sym
    
    mod = ms.tir_integration._normalize_mod(func)
    rec = db.query(mod, TARGET, kind="record")

    print(rec.trace)

    sch = tvm.tir.Schedule(mod)
    rec.trace.apply_to_schedule(sch, remove_postproc=False)
    
    with TARGET:
        lib = tvm.build(sch.mod)
    
    args_info = ms.arg_info.ArgInfo.from_prim_func(func)
    args = [make_arg(info) for info in args_info]

    # from torch.profiler import profile, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CUDA], with_modules=False, with_stack=True) as prof:
    if True:
        score_s = lib.time_evaluator(lib.entry_name, dev=tvm.cuda(0), number=2000, repeat=1)(*args).mean
        print(f"SCORE : {score_s*1e6} us",)
        print(f"TUNING REC : {rec.run_secs[0]*1e6} us",)

    if 'prof' in locals(): 
        prof.export_chrome_trace(f"./trace_qmm_rf_tuned.json")


def read_db_topk():
    db_path = "__tmp/tune_simple_qmm"
    
    db = ms.database.JSONDatabase(work_dir=db_path, allow_missing=False)

    M, N, K = 1, 22016, 4096
    GS = 128
    func = gen_simple_qmm(M, N, K, GS, dim_dtype="int32")
    
    mod = ms.tir_integration._normalize_mod(func)
    assert db.has_workload(mod)
    wkld = db.commit_workload(mod)
    recs = db.get_top_k(wkld, top_k=2000)

    for i, rec in enumerate(recs):
        spt_kind = tvm.tir.schedule.InstructionKind.get("SamplePerfectTile")
        for inst in rec.trace.insts:
            first_spt = inst
            if first_spt.kind == spt_kind:
                break
        d_des = rec.trace.decisions[first_spt]
        if d_des[2] != 1:
            print(f"{i} {rec.run_secs[0]}")
        

def main():
    B, N, K = 1, 22016, 4096
    GS = 128
    BLK = 8
    func_stat = matmul_g128_KN_sym
    func_dyn = matmul_g128_KN_sym_dynm
    mod = ms.tir_integration._normalize_mod(func_dyn)    

    sch_mods = ["DLIGHT"] + [f"TOP{i}" for i in range(1, 40)]
        

    USE_DLIGHT = True
    with TARGET:
        if USE_DLIGHT:
            import tvm.dlight as dl
            dl_mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
            dl_mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(dl_mod)
            dl_mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(dl_mod)
            lib = tvm.build(dl_mod)
        else:
            sch = tvm.tir.Schedule(mod)
            apply_trace(sch)
            lib = tvm.build(sch.mod)
    
    for M in range(16, 2048, 16):
        # args_info = ms.arg_info.ArgInfo.from_prim_func(func_stat)
        args_info = [
            # == 2D case ==
            # ms.arg_info.TensorInfo("float16", [M, K]),
            # ms.arg_info.TensorInfo("uint32", [K // BLK, N]),
            # ms.arg_info.TensorInfo("float16", [K // GS, N]),
            # ms.arg_info.TensorInfo("float16", [M, N]),
            # == 3D case ==
            ms.arg_info.TensorInfo("uint32", [K // BLK, N]),  # WGH
            ms.arg_info.TensorInfo("float16", [K // GS, N]),  # SCL
            ms.arg_info.TensorInfo("float16", [B, M, K]),     # D_IN
            ms.arg_info.TensorInfo("float16", [B, M, N]),     # D_OUT
        ]
        args = [make_arg(info) for info in args_info]

        score_s = lib.time_evaluator(lib.entry_name, dev=tvm.cuda(0), number=2000, repeat=1)(*args).mean
        print(f"M: {M} TIME: {score_s*1e6} us",)


def check_tir():
    ...
    

if __name__ == "__main__":
    main()
    check_tir()
