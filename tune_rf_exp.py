import numpy as np

import tvm
import tvm.meta_schedule as ms
from tvm import te
from tvm._ffi import get_global_func

import tvm.script.tir as T
from tvm.tir.schedule import BlockRV, Schedule

import tvm.tir.tensor_intrin.cuda
from tvm.contrib import nvcc

TARGET = tvm.target.Target("nvidia/nvidia-a10g")
DEV = tvm.cuda(0)


@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code, target):
    """Use nvcc compiler for better perf."""
    ptx = nvcc.compile_cuda(code, target_format="ptx")
    with open("code.cu", "w") as f:
        f.write(code)
    return ptx


# q4f16_2
@T.prim_func
def matmul_g128_NK_sym(lv503: T.Buffer((T.int64(22016), T.int64(512)), "uint32"), lv504: T.Buffer((T.int64(22016), T.int64(32)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_NT_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv503[v_i, v_j // T.int64(8)], lv504[v_i, v_j // T.int64(128)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv503[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv504[v_i, v_j // T.int64(128)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]


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

# q4f16_3
@T.prim_func
def matmul_g128_KN_sym_m63(lv503: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), lv504: T.Buffer((T.int64(32), T.int64(22016)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(63), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(63), T.int64(22016)), "float16")):
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
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(63), T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]

# q4f16_3
@T.prim_func
def matmul_g128_KN_sym_m127(lv503: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), lv504: T.Buffer((T.int64(32), T.int64(22016)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(127), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(127), T.int64(22016)), "float16")):
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
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(127), T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]



# q4f16_3 - Compatibility
# 3d -> 2d
# zp f16 -> zp in32
@T.prim_func
def matmul_g128_KN_sym_compat(lv503: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), lv504: T.Buffer((T.int64(32), T.int64(22016)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv503[v_i // T.int64(8), v_j], lv504[v_i // T.int64(128), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv503[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15)) - T.int32(7))) * lv504[v_i // T.int64(128), v_j]
    for i1, i2, k in T.grid(T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i1, v_i2, v_k = T.axis.remap("SSR", [i1, i2, k])
            # v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
            # T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            # T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            # with T.init():
            #     var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            # var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]

            T.reads(lv1654[0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[0, v_i1, v_i2] = var_matmul_intermediate[0, v_i1, v_i2] + lv1654[0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]

def make_arg(info):
    if info.dtype in ["float16", "float32"]:
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype(info.dtype)
    elif info.dtype in ["int32", "uint32", "int16", "int8"]:
        arr_np = np.random.randint(0, 16, size=info.shape).astype(info.dtype)
    else:
        assert False, f"Unimplemented, dtype={info.dtype}"

    return tvm.nd.array(arr_np, device=DEV)


def gen_simple_qmm(M, N, K, GS, dim_dtype="int32"):
    assert K % GS == 0
    G = K // GS

    BLK = 8

    assert dim_dtype in ["int64", "int32"]
    if dim_dtype == "int64":
        M, N, K, GS, G, BLK = (T.int64(v) for v in (M, N, K, GS, G, BLK))


    # Groupt quantization
    a = te.placeholder((M, K), name="A", dtype="float16")
    qwgh = te.placeholder((K // BLK, N), name="QWGH", dtype="uint32")
    scl = te.placeholder((G, N), name="SCL", dtype="float16")

    # Decode weights
    wgh = te.compute(
        (K, N),
        lambda vk, vn: 
            ((qwgh[vk // BLK, vn] >> (vk % BLK * 4) & 0xF) - 7).astype("float16") * 
            scl[vk // GS, vn],
        name="decode_wgh",
    )

    # Matmul
    k = te.reduce_axis((0, K), name="k")
    x = te.compute(
        (M, N),
        lambda vm, vn: te.sum(a[vm, k] * wgh[k, vn], axis=k),
        name="matmul"
    )

    return tvm.te.create_prim_func([a, qwgh, scl, x])

def gen_simple_qmm_dynm(N, K, GS, dim_dtype="int32"):
    assert K % GS == 0
    G = K // GS

    BLK = 8

    assert dim_dtype in ["int64", "int32"]
    if dim_dtype == "int64":
        N, K, GS, G, BLK = (T.int64(v) for v in (N, K, GS, G, BLK))

    M = te.var("m")

    # Groupt quantization
    a = te.placeholder((M, K), name="A", dtype="float16")
    qwgh = te.placeholder((K // BLK, N), name="QWGH", dtype="uint32")
    scl = te.placeholder((G, N), name="SCL", dtype="float16")

    # Decode weights
    wgh = te.compute(
        (K, N),
        lambda vk, vn: 
            ((qwgh[vk // BLK, vn] >> (vk % BLK * 4) & 0xF) - 7).astype("float16") * 
            scl[vk // GS, vn],
        name="decode_wgh",
    )

    # Matmul
    k = te.reduce_axis((0, K), name="k")
    x = te.compute(
        (M, N),
        lambda vm, vn: te.sum(a[vm, k] * wgh[k, vn], axis=k),
        name="matmul"
    )

    return tvm.te.create_prim_func([a, qwgh, scl, x])


@ms.utils.derived_object
class RFactorScheduleRule(ms.schedule_rule.PyScheduleRule):
    def __init__(self, mlt_rule) -> None:
        super().__init__()
        self._mlt_rule = mlt_rule

    def _initialize_with_tune_context(self, context) -> None:
        pass
    
    def is_gemm(self, sch: Schedule, rv_block: BlockRV):
        """Check if provided block is gemm
        """
        iter_vars = sch.get(rv_block).iter_vars
        if len(iter_vars) < 2: # at leats SR
            return False
        # expected S..SRS..S
        iter_types = [iv.iter_type for iv in iter_vars]
        return iter_types.count(tvm.tir.IterVar.CommReduce) == 1 and \
               iter_types.count(tvm.tir.IterVar.DataPar) == len(iter_types) - 1


    def apply(self, sch: Schedule, block: BlockRV):
        if not self.is_gemm(sch, block):
            return [sch]

        RF = 4  # make it tunable

        new_sch = sch.copy()
        
        # Make id 2d
        *i_loops, _, _ = new_sch.get_loops(block)  # loops s0, s1,... sn, N, K 
        if len(i_loops) > 1:
            # new_sch.fuse(*i_loops)
            new_sch.transform_block_layout(block, index_map=lambda vb, vm, v_vn, v_vk: (vb * 1 + vm, v_vn, v_vk))
            # b0 = new_sch.reindex(block, ("write", 0))
            # new_sch.transform_layout(block, buffer=("write", 0), index_map=lambda b, m, n: (b, n, m))
            # new_sch.transform_block_layout(block=b0, index_map=lambda vk_0, v_vm, v_vn: (vk_0, v_vm, v_vn,))
        

        # Apply "rfactor"
        loops = new_sch.get_loops(block)
        lk = loops[-1]
        lko, lki = new_sch.split(lk, factors=[RF, None])
        new_sch.reorder(lko, *loops[0:-1], lki)
        main_rf_block = new_sch.rfactor(lko, factor_axis=0)  # TODO: does factor_axis=0 affect performance?
        final_rf_block, = new_sch.get_consumers(main_rf_block)
        new_sch.annotate(final_rf_block, ann_key="schedule_rule", ann_val="None")  # Prevent recursive apply of this rule

        # attempt to make final reduction better. EXPERIMENT
        lr, *l_other = new_sch.get_loops(block=final_rf_block)
        new_sch.reorder(*l_other, lr)
        l_other_fused = new_sch.fuse(*l_other, preserve_unit_iters=True)
        thx_size = new_sch.sample_categorical(candidates=[32, 64, 128, 256, 512, 1024], probs=[0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666], decision=3)
        l_other_fused_bx, l_other_fused_tx = new_sch.split(loop=l_other_fused, factors=[None, thx_size], preserve_unit_iters=True)
        new_sch.bind(loop=l_other_fused_bx, thread_axis="blockIdx.x")
        new_sch.bind(loop=l_other_fused_tx, thread_axis="threadIdx.x")
        
        # Apply original multi level tiling rule for main reduction block
        res_schs = self._mlt_rule.apply(new_sch, main_rf_block)

        return res_schs

    def clone(self) -> ms.schedule_rule.ScheduleRule:
        new_mlt_rule = self._mlt_rule.clone()
        return RFactorScheduleRule(new_mlt_rule)


def main():
    M, N, K = 63, 22016, 4096
    GS = 128
    
    # funcs = {f"simple_qmm_{M}_{N}_{K}_{GS}": gen_simple_qmm(M, N, K, GS, dim_dtype="int32")}
    # funcs = {f"mlc_llm_22016_q4f16_2": matmul_g128_NK_sym}
    # funcs = {f"mlc_llm_22016_q4f16_kn": matmul_g128_KN_sym}
    funcs = {f"mlc_llm_22016_q4f16_kn_m63": matmul_g128_KN_sym_m63}

    mod = tvm.IRModule(funcs)

    rules_kind = "cuda-tensorcore"

    rules = ms.schedule_rule.schedule_rule.create(rules_kind)
    new_rules = rules[0:1]
    new_rules += [RFactorScheduleRule(r) for r in rules[1:3]]  # Add rfactor injection on top of original 
    new_rules += rules[3:]

    ms.tir_integration.tune_tir(
        mod=mod,
        target=TARGET,
        work_dir="__tmp/tune_mlc_llm_22016_q4f16_kn_zpf16_m63",
        max_trials_global=100500,
        max_trials_per_task=4096,
        num_trials_per_iter=32,
        cost_model="random",
        # cost_model="xgb",
        space=ms.space_generator.PostOrderApply(
            sch_rules=new_rules,
            postprocs=rules_kind,
            mutator_probs=rules_kind,
        )
    )


def read_db():
    # db_path = "__tmp/tune_simple_qmm_xxx"
    # db_path = "__tmp/tune_mlc_llm_22016_q4f16_kn_zpf16"
    db_path = "__tmp/tune_mlc_llm_22016_q4f16_kn_zpf16_m63"
    
    db = ms.database.JSONDatabase(work_dir=db_path, allow_missing=False)

    # M, N, K = 1, 22016, 4096
    # GS = 128
    # func = gen_simple_qmm(M, N, K, GS, dim_dtype="int32")
    # func = matmul_g128_KN_sym
    func = matmul_g128_KN_sym_m63
    
    mod = ms.tir_integration._normalize_mod(func)
    rec = db.query(mod, TARGET, kind="record")

    print(rec.trace)
    exit()
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
        

def manual_sch():
    def apply_trace_m1(sch: tvm.tir.Schedule) -> None:
        b0 = sch.get_block(name="matmul", func_name="main")
        b1 = sch.get_block(name="root", func_name="main")
        sch.transform_block_layout(block=b0, index_map=lambda vb, vm, v_vn, v_vk: (vb + vm, v_vn, v_vk,))
        l2, l3, l4 = sch.get_loops(block=b0)
        l5, l6 = sch.split(loop=l4, factors=[4, None], preserve_unit_iters=True)
        sch.reorder(l5, l2, l3, l6)
        b7 = sch.rfactor(loop=l5, factor_axis=0)
        b8, = sch.get_consumers(block=b7)
        sch.annotate(block_or_loop=b8, ann_key="schedule_rule", ann_val="None")
        l9, l10, l11 = sch.get_loops(block=b8)
        sch.reorder(l10, l11, l9)
        l12 = sch.fuse(l10, l11, preserve_unit_iters=True)
        v13 = sch.sample_categorical(candidates=[32, 64, 128, 256, 512, 1024], probs=[0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666], decision=1)
        l14, l15 = sch.split(loop=l12, factors=[None, v13], preserve_unit_iters=True)
        sch.bind(loop=l14, thread_axis="blockIdx.x")
        sch.bind(loop=l15, thread_axis="threadIdx.x")
        sch.annotate(block_or_loop=b7, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        b16 = sch.reindex(block=b7, buffer=("write", 0))
        b17 = sch.reindex(block=b7, buffer=("read", 0))
        b18 = sch.reindex(block=b7, buffer=("read", 1))
        sch.transform_layout(block=b7, buffer=("read", 0), index_map=lambda vax2_0, v0, vax2_1: (vax2_0, v0, vax2_1,), pad_value=None, assume_injective_transform=True)
        sch.transform_layout(block=b7, buffer=("read", 1), index_map=lambda vax2_0, v1, vax2_1: (vax2_0, vax2_1, v1,), pad_value=None, assume_injective_transform=True)
        sch.transform_layout(block=b7, buffer=("write", 0), index_map=lambda vax2_0, v0, v1: (vax2_0, v0, v1,), pad_value=None, assume_injective_transform=True)
        sch.transform_block_layout(block=b16, index_map=lambda vax2_0, v0, v1: (vax2_0, v0, v1,))
        sch.transform_block_layout(block=b17, index_map=lambda vax2_0, v0, vax2_1: (vax2_0, v0, vax2_1,))
        sch.transform_block_layout(block=b18, index_map=lambda vax2_0, v1, vax2_1: (vax2_0, vax2_1, v1,))
        sch.transform_block_layout(block=b7, index_map=lambda vax2_0, v0, v1, vax2_1: (vax2_0, v0, v1, vax2_1,))
        sch.pad_einsum(block=b7, padding=[1, 16, 1, 1])
        b19, b20 = sch.get_producers(block=b7)
        b21, = sch.get_producers(block=b19)
        sch.compute_inline(block=b21)
        b22, = sch.get_producers(block=b20)
        sch.compute_inline(block=b22)
        b23, = sch.get_consumers(block=b7)
        sch.compute_inline(block=b23)
        l24, l25, l26, l27 = sch.get_loops(block=b7)
        l28, l29 = sch.split(loop=l27, factors=[None, 16], preserve_unit_iters=True)
        l30, l31 = sch.split(loop=l26, factors=[None, 16], preserve_unit_iters=True)
        l32, l33 = sch.split(loop=l25, factors=[None, 16], preserve_unit_iters=True)
        l34, l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b7)
        sch.reorder(l37, l39, l33, l31, l29)
        b41 = sch.blockize(target=l33, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b41, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")
        sch.annotate(block_or_loop=b41, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")
        sch.annotate(block_or_loop=b41, ann_key="warp_execution", ann_val=1)
        l42, l43, l44, l45 = sch.get_loops(block=b41)
        v46, v47, v48, v49, v50 = sch.sample_perfect_tile(loop=l42, n=5, max_innermost_factor=4, decision=[1, 4, 1, 1, 1])
        l51, l52, l53, l54, l55 = sch.split(loop=l42, factors=[v46, v47, v48, v49, v50], preserve_unit_iters=True)
        # v56, v57, v58, v59, v60 = sch.sample_perfect_tile(loop=l43, n=5, max_innermost_factor=4, decision=[1, 1, 1, 1, 1])
        v56, v57, v58, v59, v60 = None, 1, 1, 1, 1
        l61, l62, l63, l64, l65 = sch.split(loop=l43, factors=[v56, v57, v58, v59, v60], preserve_unit_iters=True)
        v66, v67, v68, v69, v70 = sch.sample_perfect_tile(loop=l44, n=5, max_innermost_factor=4, decision=[172, 1, 4, 1, 2])
        l71, l72, l73, l74, l75 = sch.split(loop=l44, factors=[v66, v67, v68, v69, v70], preserve_unit_iters=True)
        v76, v77, v78 = sch.sample_perfect_tile(loop=l45, n=3, max_innermost_factor=4, decision=[32, 1, 2])
        l79, l80, l81 = sch.split(loop=l45, factors=[v76, v77, v78], preserve_unit_iters=True)
        sch.reorder(l51, l61, l71, l52, l62, l72, l53, l63, l73, l79, l80, l54, l64, l74, l81, l55, l65, l75)
        l82 = sch.fuse(l51, l61, l71, preserve_unit_iters=True)
        sch.bind(loop=l82, thread_axis="blockIdx.y")
        l83 = sch.fuse(l52, l62, l72, preserve_unit_iters=True)
        sch.bind(loop=l83, thread_axis="blockIdx.x")
        l84 = sch.fuse(l53, l63, l73, preserve_unit_iters=True)
        sch.bind(loop=l84, thread_axis="threadIdx.y")
        sch.transform_layout(block=b41, buffer=("write", 0), index_map=lambda i0, i1, i2: (i0, i1 // 16 // (v59 * v60), i2 // 16 // (v69 * v70), i1 // 16 % (v59 * v60), i2 // 16 % (v69 * v70), i1 % 16, i2 % 16,), pad_value=None, assume_injective_transform=True)
        b85 = sch.cache_write(block=b41, write_buffer_index=0, storage_scope="shared.dyn")
        sch.reverse_compute_at(block=b85, loop=l83, preserve_unit_loops=True, index=-1)
        b86 = sch.cache_write(block=b41, write_buffer_index=0, storage_scope="wmma.accumulator")
        l87, l88, l89, l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b85)
        sch.reorder(l92, l90, l91, l93)
        sch.compute_at(block=b86, loop=l92, preserve_unit_loops=True, index=-1)
        l96, l97, l98, l99, l100, l101, l102, l103, l104, l105, l106 = sch.get_loops(block=b86)
        l107 = sch.fuse(l101, l102, preserve_unit_iters=True)
        sch.bind(loop=l107, thread_axis="threadIdx.y")
        sch.reverse_compute_inline(block=b16)
        l108, l109, l110, l111, l112, l113, l114, l115, l116, l117 = sch.get_loops(block=b86)
        b118 = sch.blockize(target=l116, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b118, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")
        l119, l120, l121, l122, l123, l124, l125, l126, l127 = sch.get_loops(block=b85)
        l128 = sch.fuse(l123, l124, l125, l126, l127, preserve_unit_iters=True)
        v129 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
        sch.annotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch", ann_val=v129)
        b130 = sch.cache_read(block=b41, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b41])
        sch.compute_at(block=b130, loop=l79, preserve_unit_loops=True, index=-1)
        l131, l132, l133, l134, l135, l136, l137 = sch.get_loops(block=b130)
        l138 = sch.fuse(l135, l136, l137, preserve_unit_iters=True)
        v139 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
        sch.annotate(block_or_loop=b130, ann_key="meta_schedule.cooperative_fetch", ann_val=v139)
        b140 = sch.cache_read(block=b41, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b41])
        sch.compute_at(block=b140, loop=l79, preserve_unit_loops=True, index=-1)
        l141, l142, l143, l144, l145, l146, l147 = sch.get_loops(block=b140)
        l148 = sch.fuse(l145, l146, l147, preserve_unit_iters=True)
        v149 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
        sch.annotate(block_or_loop=b140, ann_key="meta_schedule.cooperative_fetch", ann_val=v149)
        b150 = sch.cache_read(block=b41, read_buffer_index=0, storage_scope="wmma.matrix_a")
        sch.compute_at(block=b150, loop=l80, preserve_unit_loops=True, index=-1)
        l151, l152, l153, l154, l155, l156, l157, l158 = sch.get_loops(block=b150)
        l159, l160 = sch.split(loop=l158, factors=[None, 16], preserve_unit_iters=True)
        l161, l162 = sch.split(loop=l157, factors=[None, 16], preserve_unit_iters=True)
        l163, l164, l165, l166, l167, l168, l169, l170, l171, l172 = sch.get_loops(block=b150)
        sch.reorder(l171, l162, l160)
        b173 = sch.blockize(target=l162, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b173, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")
        b174 = sch.cache_read(block=b41, read_buffer_index=1, storage_scope="wmma.matrix_b")
        sch.compute_at(block=b174, loop=l80, preserve_unit_loops=True, index=-1)
        l175, l176, l177, l178, l179, l180, l181, l182 = sch.get_loops(block=b174)
        l183, l184 = sch.split(loop=l182, factors=[None, 16], preserve_unit_iters=True)
        l185, l186 = sch.split(loop=l181, factors=[None, 16], preserve_unit_iters=True)
        l187, l188, l189, l190, l191, l192, l193, l194, l195, l196 = sch.get_loops(block=b174)
        sch.reorder(l195, l186, l184)
        b197 = sch.blockize(target=l186, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b197, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")
        b198, = sch.get_producers(block=b130)
        sch.compute_inline(block=b198)
        sch.storage_align(block=b130, buffer_index=0, axis=-2, factor=32, offset=8)
        b199, = sch.get_producers(block=b140)
        sch.compute_inline(block=b199)
        sch.storage_align(block=b140, buffer_index=0, axis=-2, factor=32, offset=8)
        v200 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=4)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v200)
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch")
        l201, l202, l203, l204, l205 = sch.get_loops(block=b85)
        l206, l207, l208, l209 = sch.split(loop=l205, factors=[None, 4, 32, 4], preserve_unit_iters=True)
        sch.vectorize(loop=l209)
        sch.bind(loop=l208, thread_axis="threadIdx.x")
        sch.bind(loop=l207, thread_axis="threadIdx.y")
        sch.unannotate(block_or_loop=b130, ann_key="meta_schedule.cooperative_fetch")
        l210, l211, l212, l213, l214 = sch.get_loops(block=b130)
        l215, l216, l217 = sch.split(loop=l214, factors=[None, 4, 32], preserve_unit_iters=True)
        sch.bind(loop=l217, thread_axis="threadIdx.x")
        sch.bind(loop=l216, thread_axis="threadIdx.y")
        sch.unannotate(block_or_loop=b140, ann_key="meta_schedule.cooperative_fetch")
        l218, l219, l220, l221, l222 = sch.get_loops(block=b140)
        l223, l224, l225, l226 = sch.split(loop=l222, factors=[None, 4, 32, 2], preserve_unit_iters=True)
        sch.vectorize(loop=l226)
        sch.bind(loop=l225, thread_axis="threadIdx.x")
        sch.bind(loop=l224, thread_axis="threadIdx.y")
        b227 = sch.get_block(name="root", func_name="main")
        sch.unannotate(block_or_loop=b227, ann_key="meta_schedule.unroll_explicit")
        b228, b229, b230, b231, b232, b233, b234, b235 = sch.get_child_blocks(b227)
        l236, l237, l238, l239, l240, l241, l242 = sch.get_loops(block=b228)
        l243, l244, l245, l246, l247, l248, l249, l250 = sch.get_loops(block=b229)
        l251, l252, l253, l254, l255, l256, l257, l258 = sch.get_loops(block=b230)
        l259, l260, l261, l262, l263, l264, l265, l266 = sch.get_loops(block=b231)
        l267, l268, l269, l270, l271, l272, l273, l274, l275, l276, l277, l278 = sch.get_loops(block=b232)
        sch.annotate(block_or_loop=l267, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l267, ann_key="pragma_unroll_explicit", ann_val=1)
        l279, l280, l281, l282, l283, l284, l285, l286 = sch.get_loops(block=b233)
        l287, l288, l289, l290, l291, l292, l293, l294 = sch.get_loops(block=b234)
        l295, l296, l297 = sch.get_loops(block=b235)
        sch.annotate(block_or_loop=l295, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l295, ann_key="pragma_unroll_explicit", ann_val=1)
        b298 = sch.get_block(name="matmul_rf_o", func_name="main")
        l299, l300, l301, l302, l303, l304, l305, l306, l307, l308, l309, l310 = sch.get_loops(block=b298)
        b311 = sch.decompose_reduction(block=b298, loop=l302)
        sch.unannotate(block_or_loop=b311, ann_key="meta_schedule.auto_tensorize")
        sch.annotate(block_or_loop=b311, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
        sch.unannotate(block_or_loop=b298, ann_key="meta_schedule.auto_tensorize_init")
        sch.unannotate(block_or_loop=b311, ann_key="meta_schedule.auto_tensorize_init")
        b312 = sch.get_block(name="matmul", func_name="main")
        l313, l314, l315 = sch.get_loops(block=b312)
        b316 = sch.decompose_reduction(block=b312, loop=l315)
        b317 = sch.get_block(name="matmul_rf_o_init", func_name="main")
        sch.unannotate(block_or_loop=b317, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b317, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
        b318 = sch.get_block(name="lv1654_reindex_pad_shared.dyn_wmma.matrix_a_o", func_name="main")
        sch.unannotate(block_or_loop=b318, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b318, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
        b319 = sch.get_block(name="p_output0_intermediate_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
        sch.unannotate(block_or_loop=b319, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b319, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
        b320 = sch.get_block(name="matmul_rf_o_update", func_name="main")
        sch.unannotate(block_or_loop=b320, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b320, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
        b321 = sch.get_block(name="var_matmul_intermediate.rf_reindex_pad_shared.dyn_wmma.accumulator_o", func_name="main")
        sch.unannotate(block_or_loop=b321, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b321, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)

    def apply_trace_m63_v1(sch: tvm.tir.Schedule) -> None:
        b0 = sch.get_block(name="matmul", func_name="main")
        b1 = sch.get_block(name="root", func_name="main")
        sch.transform_block_layout(block=b0, index_map=lambda vb, vm, v_vn, v_vk: (vb + vm, v_vn, v_vk,))
        l2, l3, l4 = sch.get_loops(block=b0)
        l5, l6 = sch.split(loop=l4, factors=[4, None], preserve_unit_iters=True)
        sch.reorder(l5, l2, l3, l6)
        b7 = sch.rfactor(loop=l5, factor_axis=0)
        b8, = sch.get_consumers(block=b7)
        sch.annotate(block_or_loop=b8, ann_key="schedule_rule", ann_val="None")
        l9, l10, l11 = sch.get_loops(block=b8)
        sch.reorder(l10, l11, l9)
        l12 = sch.fuse(l10, l11, preserve_unit_iters=True)
        v13 = sch.sample_categorical(candidates=[32, 64, 128, 256, 512, 1024], probs=[0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666], decision=5)
        l14, l15 = sch.split(loop=l12, factors=[None, v13], preserve_unit_iters=True)
        sch.bind(loop=l14, thread_axis="blockIdx.x")
        sch.bind(loop=l15, thread_axis="threadIdx.x")
        sch.annotate(block_or_loop=b7, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        b16 = sch.reindex(block=b7, buffer=("write", 0))
        b17 = sch.reindex(block=b7, buffer=("read", 0))
        b18 = sch.reindex(block=b7, buffer=("read", 1))
        sch.transform_layout(block=b7, buffer=("read", 0), index_map=lambda vax2_0, v0, vax2_1: (vax2_0, v0, vax2_1,), pad_value=None, assume_injective_transform=True)
        sch.transform_layout(block=b7, buffer=("read", 1), index_map=lambda vax2_0, v1, vax2_1: (vax2_0, vax2_1, v1,), pad_value=None, assume_injective_transform=True)
        sch.transform_layout(block=b7, buffer=("write", 0), index_map=lambda vax2_0, v0, v1: (vax2_0, v0, v1,), pad_value=None, assume_injective_transform=True)
        sch.transform_block_layout(block=b16, index_map=lambda vax2_0, v0, v1: (vax2_0, v0, v1,))
        sch.transform_block_layout(block=b17, index_map=lambda vax2_0, v0, vax2_1: (vax2_0, v0, vax2_1,))
        sch.transform_block_layout(block=b18, index_map=lambda vax2_0, v1, vax2_1: (vax2_0, vax2_1, v1,))
        sch.transform_block_layout(block=b7, index_map=lambda vax2_0, v0, v1, vax2_1: (vax2_0, v0, v1, vax2_1,))
        # sch.pad_einsum(block=b7, padding=[1, 16, 1, 1])
        sch.pad_einsum(block=b7, padding=[1, 64, 1, 1])
        b19, b20 = sch.get_producers(block=b7)
        b21, = sch.get_producers(block=b19)
        sch.compute_inline(block=b21)
        b22, = sch.get_producers(block=b20)
        sch.compute_inline(block=b22)
        b23, = sch.get_consumers(block=b7)
        sch.compute_inline(block=b23)
        l24, l25, l26, l27 = sch.get_loops(block=b7)
        l28, l29 = sch.split(loop=l27, factors=[None, 16], preserve_unit_iters=True)
        l30, l31 = sch.split(loop=l26, factors=[None, 16], preserve_unit_iters=True)
        l32, l33 = sch.split(loop=l25, factors=[None, 16], preserve_unit_iters=True)
        l34, l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b7)
        sch.reorder(l37, l39, l33, l31, l29)
        b41 = sch.blockize(target=l33, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b41, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")
        sch.annotate(block_or_loop=b41, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")
        sch.annotate(block_or_loop=b41, ann_key="warp_execution", ann_val=1)
        l42, l43, l44, l45 = sch.get_loops(block=b41)
        v46, v47, v48, v49, v50 = sch.sample_perfect_tile(loop=l42, n=5, max_innermost_factor=4, decision=[2, 1, 2, 1, 1])
        l51, l52, l53, l54, l55 = sch.split(loop=l42, factors=[v46, v47, v48, v49, v50], preserve_unit_iters=True)
        # v56, v57, v58, v59, v60 = sch.sample_perfect_tile(loop=l43, n=5, max_innermost_factor=4, decision=[1, 1, 1, 1, 4])
        v56, v57, v58, v59, v60 = None, 1, 1, 1, 4
        l61, l62, l63, l64, l65 = sch.split(loop=l43, factors=[v56, v57, v58, v59, v60], preserve_unit_iters=True)
        v66, v67, v68, v69, v70 = sch.sample_perfect_tile(loop=l44, n=5, max_innermost_factor=4, decision=[1, 86, 4, 2, 2])
        l71, l72, l73, l74, l75 = sch.split(loop=l44, factors=[v66, v67, v68, v69, v70], preserve_unit_iters=True)
        v76, v77, v78 = sch.sample_perfect_tile(loop=l45, n=3, max_innermost_factor=4, decision=[64, 1, 1])
        l79, l80, l81 = sch.split(loop=l45, factors=[v76, v77, v78], preserve_unit_iters=True)
        sch.reorder(l51, l61, l71, l52, l62, l72, l53, l63, l73, l79, l80, l54, l64, l74, l81, l55, l65, l75)
        l82 = sch.fuse(l51, l61, l71, preserve_unit_iters=True)
        sch.bind(loop=l82, thread_axis="blockIdx.y")
        l83 = sch.fuse(l52, l62, l72, preserve_unit_iters=True)
        sch.bind(loop=l83, thread_axis="blockIdx.x")
        l84 = sch.fuse(l53, l63, l73, preserve_unit_iters=True)
        sch.bind(loop=l84, thread_axis="threadIdx.y")
        sch.transform_layout(block=b41, buffer=("write", 0), index_map=lambda i0, i1, i2: (i0, i1 // 16 // (v59 * v60), i2 // 16 // (v69 * v70), i1 // 16 % (v59 * v60), i2 // 16 % (v69 * v70), i1 % 16, i2 % 16,), pad_value=None, assume_injective_transform=True)
        b85 = sch.cache_write(block=b41, write_buffer_index=0, storage_scope="shared.dyn")
        sch.reverse_compute_at(block=b85, loop=l83, preserve_unit_loops=True, index=-1)
        b86 = sch.cache_write(block=b41, write_buffer_index=0, storage_scope="wmma.accumulator")
        l87, l88, l89, l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b85)
        sch.reorder(l92, l90, l91, l93)
        sch.compute_at(block=b86, loop=l92, preserve_unit_loops=True, index=-1)
        l96, l97, l98, l99, l100, l101, l102, l103, l104, l105, l106 = sch.get_loops(block=b86)
        l107 = sch.fuse(l101, l102, preserve_unit_iters=True)
        sch.bind(loop=l107, thread_axis="threadIdx.y")
        sch.reverse_compute_inline(block=b16)
        l108, l109, l110, l111, l112, l113, l114, l115, l116, l117 = sch.get_loops(block=b86)
        b118 = sch.blockize(target=l116, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b118, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")
        l119, l120, l121, l122, l123, l124, l125, l126, l127 = sch.get_loops(block=b85)
        l128 = sch.fuse(l123, l124, l125, l126, l127, preserve_unit_iters=True)
        v129 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
        sch.annotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch", ann_val=v129)
        b130 = sch.cache_read(block=b41, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b41])
        sch.compute_at(block=b130, loop=l79, preserve_unit_loops=True, index=-1)
        l131, l132, l133, l134, l135, l136, l137 = sch.get_loops(block=b130)
        l138 = sch.fuse(l135, l136, l137, preserve_unit_iters=True)
        v139 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
        sch.annotate(block_or_loop=b130, ann_key="meta_schedule.cooperative_fetch", ann_val=v139)
        b140 = sch.cache_read(block=b41, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b41])
        sch.compute_at(block=b140, loop=l79, preserve_unit_loops=True, index=-1)
        l141, l142, l143, l144, l145, l146, l147 = sch.get_loops(block=b140)
        l148 = sch.fuse(l145, l146, l147, preserve_unit_iters=True)
        v149 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
        sch.annotate(block_or_loop=b140, ann_key="meta_schedule.cooperative_fetch", ann_val=v149)
        b150 = sch.cache_read(block=b41, read_buffer_index=0, storage_scope="wmma.matrix_a")
        sch.compute_at(block=b150, loop=l80, preserve_unit_loops=True, index=-1)
        l151, l152, l153, l154, l155, l156, l157, l158 = sch.get_loops(block=b150)
        l159, l160 = sch.split(loop=l158, factors=[None, 16], preserve_unit_iters=True)
        l161, l162 = sch.split(loop=l157, factors=[None, 16], preserve_unit_iters=True)
        l163, l164, l165, l166, l167, l168, l169, l170, l171, l172 = sch.get_loops(block=b150)
        sch.reorder(l171, l162, l160)
        b173 = sch.blockize(target=l162, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b173, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")
        b174 = sch.cache_read(block=b41, read_buffer_index=1, storage_scope="wmma.matrix_b")
        sch.compute_at(block=b174, loop=l80, preserve_unit_loops=True, index=-1)
        l175, l176, l177, l178, l179, l180, l181, l182 = sch.get_loops(block=b174)
        l183, l184 = sch.split(loop=l182, factors=[None, 16], preserve_unit_iters=True)
        l185, l186 = sch.split(loop=l181, factors=[None, 16], preserve_unit_iters=True)
        l187, l188, l189, l190, l191, l192, l193, l194, l195, l196 = sch.get_loops(block=b174)
        sch.reorder(l195, l186, l184)
        b197 = sch.blockize(target=l186, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b197, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")
        b198, = sch.get_producers(block=b130)
        sch.compute_inline(block=b198)
        sch.storage_align(block=b130, buffer_index=0, axis=-2, factor=32, offset=8)
        b199, = sch.get_producers(block=b140)
        sch.compute_inline(block=b199)
        sch.storage_align(block=b140, buffer_index=0, axis=-2, factor=32, offset=8)
        v200 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v200)
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch")
        l201, l202, l203, l204, l205 = sch.get_loops(block=b85)
        l206, l207, l208, l209 = sch.split(loop=l205, factors=[None, 4, 32, 8], preserve_unit_iters=True)
        sch.vectorize(loop=l209)
        sch.bind(loop=l208, thread_axis="threadIdx.x")
        sch.bind(loop=l207, thread_axis="threadIdx.y")
        sch.unannotate(block_or_loop=b130, ann_key="meta_schedule.cooperative_fetch")
        l210, l211, l212, l213, l214 = sch.get_loops(block=b130)
        l215, l216, l217 = sch.split(loop=l214, factors=[None, 4, 32], preserve_unit_iters=True)
        sch.bind(loop=l217, thread_axis="threadIdx.x")
        sch.bind(loop=l216, thread_axis="threadIdx.y")
        sch.unannotate(block_or_loop=b140, ann_key="meta_schedule.cooperative_fetch")
        l218, l219, l220, l221, l222 = sch.get_loops(block=b140)
        l223, l224, l225, l226 = sch.split(loop=l222, factors=[None, 4, 32, 2], preserve_unit_iters=True)
        sch.vectorize(loop=l226)
        sch.bind(loop=l225, thread_axis="threadIdx.x")
        sch.bind(loop=l224, thread_axis="threadIdx.y")
        b227 = sch.get_block(name="root", func_name="main")
        sch.unannotate(block_or_loop=b227, ann_key="meta_schedule.unroll_explicit")
        b228, b229, b230, b231, b232, b233, b234, b235 = sch.get_child_blocks(b227)
        l236, l237, l238, l239, l240, l241, l242 = sch.get_loops(block=b228)
        l243, l244, l245, l246, l247, l248, l249, l250 = sch.get_loops(block=b229)
        l251, l252, l253, l254, l255, l256, l257, l258 = sch.get_loops(block=b230)
        l259, l260, l261, l262, l263, l264, l265, l266 = sch.get_loops(block=b231)
        l267, l268, l269, l270, l271, l272, l273, l274, l275, l276, l277, l278 = sch.get_loops(block=b232)
        sch.annotate(block_or_loop=l267, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        sch.annotate(block_or_loop=l267, ann_key="pragma_unroll_explicit", ann_val=1)
        l279, l280, l281, l282, l283, l284, l285, l286 = sch.get_loops(block=b233)
        l287, l288, l289, l290, l291, l292, l293, l294 = sch.get_loops(block=b234)
        l295, l296, l297 = sch.get_loops(block=b235)
        sch.annotate(block_or_loop=l295, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        sch.annotate(block_or_loop=l295, ann_key="pragma_unroll_explicit", ann_val=1)
        b298 = sch.get_block(name="matmul_rf_o", func_name="main")
        l299, l300, l301, l302, l303, l304, l305, l306, l307, l308, l309, l310 = sch.get_loops(block=b298)
        b311 = sch.decompose_reduction(block=b298, loop=l302)
        sch.unannotate(block_or_loop=b311, ann_key="meta_schedule.auto_tensorize")
        sch.annotate(block_or_loop=b311, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
        sch.unannotate(block_or_loop=b298, ann_key="meta_schedule.auto_tensorize_init")
        sch.unannotate(block_or_loop=b311, ann_key="meta_schedule.auto_tensorize_init")
        b312 = sch.get_block(name="matmul", func_name="main")
        l313, l314, l315 = sch.get_loops(block=b312)
        b316 = sch.decompose_reduction(block=b312, loop=l315)
        b317 = sch.get_block(name="matmul_rf_o_init", func_name="main")
        sch.unannotate(block_or_loop=b317, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b317, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
        b318 = sch.get_block(name="lv1654_reindex_pad_shared.dyn_wmma.matrix_a_o", func_name="main")
        sch.unannotate(block_or_loop=b318, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b318, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
        b319 = sch.get_block(name="p_output0_intermediate_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
        sch.unannotate(block_or_loop=b319, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b319, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
        b320 = sch.get_block(name="matmul_rf_o_update", func_name="main")
        sch.unannotate(block_or_loop=b320, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b320, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
        b321 = sch.get_block(name="var_matmul_intermediate.rf_reindex_pad_shared.dyn_wmma.accumulator_o", func_name="main")
        sch.unannotate(block_or_loop=b321, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b321, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)


    def apply_trace_m63_v2(sch: tvm.tir.Schedule) -> None:
        b0 = sch.get_block(name="matmul", func_name="main")
        b1 = sch.get_block(name="root", func_name="main")
        sch.transform_block_layout(block=b0, index_map=lambda vb, vm, v_vn, v_vk: (vb + vm, v_vn, v_vk,))
        l2, l3, l4 = sch.get_loops(block=b0)
        l5, l6 = sch.split(loop=l4, factors=[4, None], preserve_unit_iters=True)
        sch.reorder(l5, l2, l3, l6)
        b7 = sch.rfactor(loop=l5, factor_axis=0)
        b8, = sch.get_consumers(block=b7)
        sch.annotate(block_or_loop=b8, ann_key="schedule_rule", ann_val="None")
        l9, l10, l11 = sch.get_loops(block=b8)
        sch.reorder(l10, l11, l9)
        l12 = sch.fuse(l10, l11, preserve_unit_iters=True)
        v13 = sch.sample_categorical(candidates=[32, 64, 128, 256, 512, 1024], probs=[0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666], decision=4)
        l14, l15 = sch.split(loop=l12, factors=[None, v13], preserve_unit_iters=True)
        sch.bind(loop=l14, thread_axis="blockIdx.x")
        sch.bind(loop=l15, thread_axis="threadIdx.x")
        sch.annotate(block_or_loop=b7, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        b16 = sch.reindex(block=b7, buffer=("write", 0))
        b17 = sch.reindex(block=b7, buffer=("read", 0))
        b18 = sch.reindex(block=b7, buffer=("read", 1))
        sch.transform_layout(block=b7, buffer=("read", 0), index_map=lambda vax2_0, v0, vax2_1: (vax2_0, v0, vax2_1,), pad_value=None, assume_injective_transform=True)
        sch.transform_layout(block=b7, buffer=("read", 1), index_map=lambda vax2_0, v1, vax2_1: (vax2_0, vax2_1, v1,), pad_value=None, assume_injective_transform=True)
        sch.transform_layout(block=b7, buffer=("write", 0), index_map=lambda vax2_0, v0, v1: (vax2_0, v0, v1,), pad_value=None, assume_injective_transform=True)
        sch.transform_block_layout(block=b16, index_map=lambda vax2_0, v0, v1: (vax2_0, v0, v1,))
        sch.transform_block_layout(block=b17, index_map=lambda vax2_0, v0, vax2_1: (vax2_0, v0, vax2_1,))
        sch.transform_block_layout(block=b18, index_map=lambda vax2_0, v1, vax2_1: (vax2_0, vax2_1, v1,))
        sch.transform_block_layout(block=b7, index_map=lambda vax2_0, v0, v1, vax2_1: (vax2_0, v0, v1, vax2_1,))
        # sch.pad_einsum(block=b7, padding=[1, 16, 1, 1])
        sch.pad_einsum(block=b7, padding=[1, 64, 1, 1])
        b19, b20 = sch.get_producers(block=b7)
        b21, = sch.get_producers(block=b19)
        sch.compute_inline(block=b21)
        b22, = sch.get_producers(block=b20)
        sch.compute_inline(block=b22)
        b23, = sch.get_consumers(block=b7)
        sch.compute_inline(block=b23)
        l24, l25, l26, l27 = sch.get_loops(block=b7)
        l28, l29 = sch.split(loop=l27, factors=[None, 16], preserve_unit_iters=True)
        l30, l31 = sch.split(loop=l26, factors=[None, 16], preserve_unit_iters=True)
        l32, l33 = sch.split(loop=l25, factors=[None, 16], preserve_unit_iters=True)
        l34, l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b7)
        sch.reorder(l37, l39, l33, l31, l29)
        b41 = sch.blockize(target=l33, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b41, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")
        sch.annotate(block_or_loop=b41, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")
        sch.annotate(block_or_loop=b41, ann_key="warp_execution", ann_val=1)
        l42, l43, l44, l45 = sch.get_loops(block=b41)
        v46, v47, v48, v49, v50 = sch.sample_perfect_tile(loop=l42, n=5, max_innermost_factor=4, decision=[2, 1, 2, 1, 1])    # RF
        l51, l52, l53, l54, l55 = sch.split(loop=l42, factors=[v46, v47, v48, v49, v50], preserve_unit_iters=True)
        # v56, v57, v58, v59, v60 = sch.sample_perfect_tile(loop=l43, n=5, max_innermost_factor=4, decision=[1, 1, 4, 1, 1])  # M
        v56, v57, v58, v59, v60 = None, 1, 4, 1, 1
        l61, l62, l63, l64, l65 = sch.split(loop=l43, factors=[v56, v57, v58, v59, v60], preserve_unit_iters=True)
        v66, v67, v68, v69, v70 = sch.sample_perfect_tile(loop=l44, n=5, max_innermost_factor=4, decision=[1, 172, 2, 1, 4])  # N
        l71, l72, l73, l74, l75 = sch.split(loop=l44, factors=[v66, v67, v68, v69, v70], preserve_unit_iters=True)
        v76, v77, v78 = sch.sample_perfect_tile(loop=l45, n=3, max_innermost_factor=4, decision=[32, 1, 2])                   # K
        l79, l80, l81 = sch.split(loop=l45, factors=[v76, v77, v78], preserve_unit_iters=True)
        sch.reorder(l51, l61, l71, l52, l62, l72, l53, l63, l73, l79, l80, l54, l64, l74, l81, l55, l65, l75)
        l82 = sch.fuse(l51, l61, l71, preserve_unit_iters=True)
        sch.bind(loop=l82, thread_axis="blockIdx.y")
        l83 = sch.fuse(l52, l62, l72, preserve_unit_iters=True)
        sch.bind(loop=l83, thread_axis="blockIdx.x")
        l84 = sch.fuse(l53, l63, l73, preserve_unit_iters=True)
        sch.bind(loop=l84, thread_axis="threadIdx.y")
        sch.transform_layout(block=b41, buffer=("write", 0), index_map=lambda i0, i1, i2: (i0, i1 // 16 // (v59 * v60), i2 // 16 // (v69 * v70), i1 // 16 % (v59 * v60), i2 // 16 % (v69 * v70), i1 % 16, i2 % 16,), pad_value=None, assume_injective_transform=True)
        b85 = sch.cache_write(block=b41, write_buffer_index=0, storage_scope="shared.dyn")
        sch.reverse_compute_at(block=b85, loop=l83, preserve_unit_loops=True, index=-1)
        b86 = sch.cache_write(block=b41, write_buffer_index=0, storage_scope="wmma.accumulator")
        l87, l88, l89, l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b85)
        sch.reorder(l92, l90, l91, l93)
        sch.compute_at(block=b86, loop=l92, preserve_unit_loops=True, index=-1)
        l96, l97, l98, l99, l100, l101, l102, l103, l104, l105, l106 = sch.get_loops(block=b86)
        l107 = sch.fuse(l101, l102, preserve_unit_iters=True)
        sch.bind(loop=l107, thread_axis="threadIdx.y")
        sch.reverse_compute_inline(block=b16)
        l108, l109, l110, l111, l112, l113, l114, l115, l116, l117 = sch.get_loops(block=b86)
        b118 = sch.blockize(target=l116, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b118, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")
        l119, l120, l121, l122, l123, l124, l125, l126, l127 = sch.get_loops(block=b85)
        l128 = sch.fuse(l123, l124, l125, l126, l127, preserve_unit_iters=True)
        v129 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
        sch.annotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch", ann_val=v129)
        b130 = sch.cache_read(block=b41, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b41])
        sch.compute_at(block=b130, loop=l79, preserve_unit_loops=True, index=-1)
        l131, l132, l133, l134, l135, l136, l137 = sch.get_loops(block=b130)
        l138 = sch.fuse(l135, l136, l137, preserve_unit_iters=True)
        v139 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
        sch.annotate(block_or_loop=b130, ann_key="meta_schedule.cooperative_fetch", ann_val=v139)
        b140 = sch.cache_read(block=b41, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b41])
        sch.compute_at(block=b140, loop=l79, preserve_unit_loops=True, index=-1)
        l141, l142, l143, l144, l145, l146, l147 = sch.get_loops(block=b140)
        l148 = sch.fuse(l145, l146, l147, preserve_unit_iters=True)
        v149 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
        sch.annotate(block_or_loop=b140, ann_key="meta_schedule.cooperative_fetch", ann_val=v149)
        b150 = sch.cache_read(block=b41, read_buffer_index=0, storage_scope="wmma.matrix_a")
        sch.compute_at(block=b150, loop=l80, preserve_unit_loops=True, index=-1)
        l151, l152, l153, l154, l155, l156, l157, l158 = sch.get_loops(block=b150)
        l159, l160 = sch.split(loop=l158, factors=[None, 16], preserve_unit_iters=True)
        l161, l162 = sch.split(loop=l157, factors=[None, 16], preserve_unit_iters=True)
        l163, l164, l165, l166, l167, l168, l169, l170, l171, l172 = sch.get_loops(block=b150)
        sch.reorder(l171, l162, l160)
        b173 = sch.blockize(target=l162, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b173, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")
        b174 = sch.cache_read(block=b41, read_buffer_index=1, storage_scope="wmma.matrix_b")
        sch.compute_at(block=b174, loop=l80, preserve_unit_loops=True, index=-1)
        l175, l176, l177, l178, l179, l180, l181, l182 = sch.get_loops(block=b174)
        l183, l184 = sch.split(loop=l182, factors=[None, 16], preserve_unit_iters=True)
        l185, l186 = sch.split(loop=l181, factors=[None, 16], preserve_unit_iters=True)
        l187, l188, l189, l190, l191, l192, l193, l194, l195, l196 = sch.get_loops(block=b174)
        sch.reorder(l195, l186, l184)
        b197 = sch.blockize(target=l186, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b197, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")
        b198, = sch.get_producers(block=b130)
        sch.compute_inline(block=b198)
        sch.storage_align(block=b130, buffer_index=0, axis=-2, factor=32, offset=8)
        b199, = sch.get_producers(block=b140)
        sch.compute_inline(block=b199)
        sch.storage_align(block=b140, buffer_index=0, axis=-2, factor=32, offset=8)
        v200 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
        # sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v200)
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch")
        l201, l202, l203, l204, l205 = sch.get_loops(block=b85)
        l206, l207, l208, l209 = sch.split(loop=l205, factors=[None, 8, 32, 2], preserve_unit_iters=True)
        sch.vectorize(loop=l209)
        sch.bind(loop=l208, thread_axis="threadIdx.x")
        sch.bind(loop=l207, thread_axis="threadIdx.y")
        sch.unannotate(block_or_loop=b130, ann_key="meta_schedule.cooperative_fetch")
        l210, l211, l212, l213, l214 = sch.get_loops(block=b130)
        l215, l216, l217 = sch.split(loop=l214, factors=[None, 8, 32], preserve_unit_iters=True)
        sch.bind(loop=l217, thread_axis="threadIdx.x")
        sch.bind(loop=l216, thread_axis="threadIdx.y")
        sch.unannotate(block_or_loop=b140, ann_key="meta_schedule.cooperative_fetch")
        l218, l219, l220, l221, l222 = sch.get_loops(block=b140)
        l223, l224, l225, l226 = sch.split(loop=l222, factors=[None, 8, 32, 4], preserve_unit_iters=True)
        sch.vectorize(loop=l226)
        sch.bind(loop=l225, thread_axis="threadIdx.x")
        sch.bind(loop=l224, thread_axis="threadIdx.y")
        b227 = sch.get_block(name="root", func_name="main")
        # sch.unannotate(block_or_loop=b227, ann_key="meta_schedule.unroll_explicit")
        b228, b229, b230, b231, b232, b233, b234, b235 = sch.get_child_blocks(b227)
        l236, l237, l238, l239, l240, l241, l242 = sch.get_loops(block=b228)
        l243, l244, l245, l246, l247, l248, l249, l250 = sch.get_loops(block=b229)
        l251, l252, l253, l254, l255, l256, l257, l258 = sch.get_loops(block=b230)
        l259, l260, l261, l262, l263, l264, l265, l266 = sch.get_loops(block=b231)
        l267, l268, l269, l270, l271, l272, l273, l274, l275, l276, l277, l278 = sch.get_loops(block=b232)
        # sch.annotate(block_or_loop=l267, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        # sch.annotate(block_or_loop=l267, ann_key="pragma_unroll_explicit", ann_val=1)
        l279, l280, l281, l282, l283, l284, l285, l286 = sch.get_loops(block=b233)
        l287, l288, l289, l290, l291, l292, l293, l294 = sch.get_loops(block=b234)
        l295, l296, l297 = sch.get_loops(block=b235)
        # sch.annotate(block_or_loop=l295, ann_key="pragma_auto_unroll_max_step", ann_val=512)
        # sch.annotate(block_or_loop=l295, ann_key="pragma_unroll_explicit", ann_val=1)
        b298 = sch.get_block(name="matmul_rf_o", func_name="main")
        l299, l300, l301, l302, l303, l304, l305, l306, l307, l308, l309, l310 = sch.get_loops(block=b298)
        b311 = sch.decompose_reduction(block=b298, loop=l302)
        sch.unannotate(block_or_loop=b311, ann_key="meta_schedule.auto_tensorize")
        sch.annotate(block_or_loop=b311, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
        sch.unannotate(block_or_loop=b298, ann_key="meta_schedule.auto_tensorize_init")
        sch.unannotate(block_or_loop=b311, ann_key="meta_schedule.auto_tensorize_init")
        b312 = sch.get_block(name="matmul", func_name="main")
        l313, l314, l315 = sch.get_loops(block=b312)
        b316 = sch.decompose_reduction(block=b312, loop=l315)
        b317 = sch.get_block(name="matmul_rf_o_init", func_name="main")
        sch.unannotate(block_or_loop=b317, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b317, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
        b318 = sch.get_block(name="lv1654_reindex_pad_shared.dyn_wmma.matrix_a_o", func_name="main")
        sch.unannotate(block_or_loop=b318, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b318, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
        b319 = sch.get_block(name="p_output0_intermediate_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
        sch.unannotate(block_or_loop=b319, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b319, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
        b320 = sch.get_block(name="matmul_rf_o_update", func_name="main")
        sch.unannotate(block_or_loop=b320, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b320, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
        b321 = sch.get_block(name="var_matmul_intermediate.rf_reindex_pad_shared.dyn_wmma.accumulator_o", func_name="main")
        sch.unannotate(block_or_loop=b321, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b321, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)

    def apply_trace_m63_v3(sch: tvm.tir.Schedule) -> None:
        """ 206 us """
        b0 = sch.get_block(name="matmul", func_name="main")
        b1 = sch.get_block(name="root", func_name="main")
        sch.transform_block_layout(block=b0, index_map=lambda vb, vm, v_vn, v_vk: (vb + vm, v_vn, v_vk,))
        l2, l3, l4 = sch.get_loops(block=b0)
        l5, l6 = sch.split(loop=l4, factors=[4, None], preserve_unit_iters=True)
        sch.reorder(l5, l2, l3, l6)
        b7 = sch.rfactor(loop=l5, factor_axis=0)
        b8, = sch.get_consumers(block=b7)
        sch.annotate(block_or_loop=b8, ann_key="schedule_rule", ann_val="None")
        l9, l10, l11 = sch.get_loops(block=b8)
        sch.reorder(l10, l11, l9)
        l12 = sch.fuse(l10, l11, preserve_unit_iters=True)
        v13 = sch.sample_categorical(candidates=[32, 64, 128, 256, 512, 1024], probs=[0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666], decision=5)
        l14, l15 = sch.split(loop=l12, factors=[None, v13], preserve_unit_iters=True)
        sch.bind(loop=l14, thread_axis="blockIdx.x")
        sch.bind(loop=l15, thread_axis="threadIdx.x")
        sch.annotate(block_or_loop=b7, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        b16 = sch.reindex(block=b7, buffer=("write", 0))
        b17 = sch.reindex(block=b7, buffer=("read", 0))
        b18 = sch.reindex(block=b7, buffer=("read", 1))
        sch.transform_layout(block=b7, buffer=("read", 0), index_map=lambda vax2_0, v0, vax2_1: (vax2_0, v0, vax2_1,), pad_value=None, assume_injective_transform=True)
        sch.transform_layout(block=b7, buffer=("read", 1), index_map=lambda vax2_0, v1, vax2_1: (vax2_0, vax2_1, v1,), pad_value=None, assume_injective_transform=True)
        sch.transform_layout(block=b7, buffer=("write", 0), index_map=lambda vax2_0, v0, v1: (vax2_0, v0, v1,), pad_value=None, assume_injective_transform=True)
        sch.transform_block_layout(block=b16, index_map=lambda vax2_0, v0, v1: (vax2_0, v0, v1,))
        sch.transform_block_layout(block=b17, index_map=lambda vax2_0, v0, vax2_1: (vax2_0, v0, vax2_1,))
        sch.transform_block_layout(block=b18, index_map=lambda vax2_0, v1, vax2_1: (vax2_0, vax2_1, v1,))
        sch.transform_block_layout(block=b7, index_map=lambda vax2_0, v0, v1, vax2_1: (vax2_0, v0, v1, vax2_1,))
        # sch.pad_einsum(block=b7, padding=[1, 16, 1, 1])
        sch.pad_einsum(block=b7, padding=[1, 64, 1, 1])
        b19, b20 = sch.get_producers(block=b7)
        b21, = sch.get_producers(block=b19)
        sch.compute_inline(block=b21)
        b22, = sch.get_producers(block=b20)
        sch.compute_inline(block=b22)
        b23, = sch.get_consumers(block=b7)
        sch.compute_inline(block=b23)
        l24, l25, l26, l27 = sch.get_loops(block=b7)
        l28, l29 = sch.split(loop=l27, factors=[None, 16], preserve_unit_iters=True)
        l30, l31 = sch.split(loop=l26, factors=[None, 16], preserve_unit_iters=True)
        l32, l33 = sch.split(loop=l25, factors=[None, 16], preserve_unit_iters=True)
        l34, l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b7)
        sch.reorder(l37, l39, l33, l31, l29)
        b41 = sch.blockize(target=l33, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b41, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")
        sch.annotate(block_or_loop=b41, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")
        sch.annotate(block_or_loop=b41, ann_key="warp_execution", ann_val=1)
        l42, l43, l44, l45 = sch.get_loops(block=b41)
        v46, v47, v48, v49, v50 = sch.sample_perfect_tile(loop=l42, n=5, max_innermost_factor=4, decision=[2, 1, 2, 1, 1])
        l51, l52, l53, l54, l55 = sch.split(loop=l42, factors=[v46, v47, v48, v49, v50], preserve_unit_iters=True)
        # v56, v57, v58, v59, v60 = sch.sample_perfect_tile(loop=l43, n=5, max_innermost_factor=4, decision=[1, 1, 1, 4, 1])
        v56, v57, v58, v59, v60 = None, 1, 1, 4, 1
        l61, l62, l63, l64, l65 = sch.split(loop=l43, factors=[v56, v57, v58, v59, v60], preserve_unit_iters=True)
        v66, v67, v68, v69, v70 = sch.sample_perfect_tile(loop=l44, n=5, max_innermost_factor=4, decision=[2, 86, 4, 1, 2])
        l71, l72, l73, l74, l75 = sch.split(loop=l44, factors=[v66, v67, v68, v69, v70], preserve_unit_iters=True)
        v76, v77, v78 = sch.sample_perfect_tile(loop=l45, n=3, max_innermost_factor=4, decision=[64, 1, 1])
        l79, l80, l81 = sch.split(loop=l45, factors=[v76, v77, v78], preserve_unit_iters=True)
        sch.reorder(l51, l61, l71, l52, l62, l72, l53, l63, l73, l79, l80, l54, l64, l74, l81, l55, l65, l75)
        l82 = sch.fuse(l51, l61, l71, preserve_unit_iters=True)
        sch.bind(loop=l82, thread_axis="blockIdx.y")
        l83 = sch.fuse(l52, l62, l72, preserve_unit_iters=True)
        sch.bind(loop=l83, thread_axis="blockIdx.x")
        l84 = sch.fuse(l53, l63, l73, preserve_unit_iters=True)
        sch.bind(loop=l84, thread_axis="threadIdx.y")
        sch.transform_layout(block=b41, buffer=("write", 0), index_map=lambda i0, i1, i2: (i0, i1 // 16 // (v59 * v60), i2 // 16 // (v69 * v70), i1 // 16 % (v59 * v60), i2 // 16 % (v69 * v70), i1 % 16, i2 % 16,), pad_value=None, assume_injective_transform=True)
        b85 = sch.cache_write(block=b41, write_buffer_index=0, storage_scope="shared.dyn")
        sch.reverse_compute_at(block=b85, loop=l83, preserve_unit_loops=True, index=-1)
        b86 = sch.cache_write(block=b41, write_buffer_index=0, storage_scope="wmma.accumulator")
        l87, l88, l89, l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b85)
        sch.reorder(l92, l90, l91, l93)
        sch.compute_at(block=b86, loop=l92, preserve_unit_loops=True, index=-1)
        l96, l97, l98, l99, l100, l101, l102, l103, l104, l105, l106 = sch.get_loops(block=b86)
        l107 = sch.fuse(l101, l102, preserve_unit_iters=True)
        sch.bind(loop=l107, thread_axis="threadIdx.y")
        sch.reverse_compute_inline(block=b16)
        l108, l109, l110, l111, l112, l113, l114, l115, l116, l117 = sch.get_loops(block=b86)
        b118 = sch.blockize(target=l116, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b118, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")
        l119, l120, l121, l122, l123, l124, l125, l126, l127 = sch.get_loops(block=b85)
        l128 = sch.fuse(l123, l124, l125, l126, l127, preserve_unit_iters=True)
        v129 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
        sch.annotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch", ann_val=v129)
        b130 = sch.cache_read(block=b41, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b41])
        sch.compute_at(block=b130, loop=l79, preserve_unit_loops=True, index=-1)
        l131, l132, l133, l134, l135, l136, l137 = sch.get_loops(block=b130)
        l138 = sch.fuse(l135, l136, l137, preserve_unit_iters=True)
        v139 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
        sch.annotate(block_or_loop=b130, ann_key="meta_schedule.cooperative_fetch", ann_val=v139)
        b140 = sch.cache_read(block=b41, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b41])
        sch.compute_at(block=b140, loop=l79, preserve_unit_loops=True, index=-1)
        l141, l142, l143, l144, l145, l146, l147 = sch.get_loops(block=b140)
        l148 = sch.fuse(l145, l146, l147, preserve_unit_iters=True)
        v149 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
        sch.annotate(block_or_loop=b140, ann_key="meta_schedule.cooperative_fetch", ann_val=v149)
        b150 = sch.cache_read(block=b41, read_buffer_index=0, storage_scope="wmma.matrix_a")
        sch.compute_at(block=b150, loop=l80, preserve_unit_loops=True, index=-1)
        l151, l152, l153, l154, l155, l156, l157, l158 = sch.get_loops(block=b150)
        l159, l160 = sch.split(loop=l158, factors=[None, 16], preserve_unit_iters=True)
        l161, l162 = sch.split(loop=l157, factors=[None, 16], preserve_unit_iters=True)
        l163, l164, l165, l166, l167, l168, l169, l170, l171, l172 = sch.get_loops(block=b150)
        sch.reorder(l171, l162, l160)
        b173 = sch.blockize(target=l162, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b173, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")
        b174 = sch.cache_read(block=b41, read_buffer_index=1, storage_scope="wmma.matrix_b")
        sch.compute_at(block=b174, loop=l80, preserve_unit_loops=True, index=-1)
        l175, l176, l177, l178, l179, l180, l181, l182 = sch.get_loops(block=b174)
        l183, l184 = sch.split(loop=l182, factors=[None, 16], preserve_unit_iters=True)
        l185, l186 = sch.split(loop=l181, factors=[None, 16], preserve_unit_iters=True)
        l187, l188, l189, l190, l191, l192, l193, l194, l195, l196 = sch.get_loops(block=b174)
        sch.reorder(l195, l186, l184)
        b197 = sch.blockize(target=l186, preserve_unit_iters=True)
        sch.annotate(block_or_loop=b197, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")
        b198, = sch.get_producers(block=b130)
        sch.compute_inline(block=b198)
        sch.storage_align(block=b130, buffer_index=0, axis=-2, factor=32, offset=8)
        b199, = sch.get_producers(block=b140)
        sch.compute_inline(block=b199)
        sch.storage_align(block=b140, buffer_index=0, axis=-2, factor=32, offset=8)
        v200 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v200)
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch")
        l201, l202, l203, l204, l205 = sch.get_loops(block=b85)
        l206, l207, l208, l209 = sch.split(loop=l205, factors=[None, 4, 32, 8], preserve_unit_iters=True)
        sch.vectorize(loop=l209)
        sch.bind(loop=l208, thread_axis="threadIdx.x")
        sch.bind(loop=l207, thread_axis="threadIdx.y")
        sch.unannotate(block_or_loop=b130, ann_key="meta_schedule.cooperative_fetch")
        l210, l211, l212, l213, l214 = sch.get_loops(block=b130)
        l215, l216, l217 = sch.split(loop=l214, factors=[None, 4, 32], preserve_unit_iters=True)
        sch.bind(loop=l217, thread_axis="threadIdx.x")
        sch.bind(loop=l216, thread_axis="threadIdx.y")
        sch.unannotate(block_or_loop=b140, ann_key="meta_schedule.cooperative_fetch")
        l218, l219, l220, l221, l222 = sch.get_loops(block=b140)
        l223, l224, l225 = sch.split(loop=l222, factors=[None, 4, 32], preserve_unit_iters=True)
        sch.bind(loop=l225, thread_axis="threadIdx.x")
        sch.bind(loop=l224, thread_axis="threadIdx.y")
        b226 = sch.get_block(name="root", func_name="main")
        sch.unannotate(block_or_loop=b226, ann_key="meta_schedule.unroll_explicit")
        b227, b228, b229, b230, b231, b232, b233, b234 = sch.get_child_blocks(b226)
        l235, l236, l237, l238, l239, l240, l241 = sch.get_loops(block=b227)
        l242, l243, l244, l245, l246, l247, l248 = sch.get_loops(block=b228)
        l249, l250, l251, l252, l253, l254, l255, l256 = sch.get_loops(block=b229)
        l257, l258, l259, l260, l261, l262, l263, l264 = sch.get_loops(block=b230)
        l265, l266, l267, l268, l269, l270, l271, l272, l273, l274, l275, l276 = sch.get_loops(block=b231)
        sch.annotate(block_or_loop=l265, ann_key="pragma_auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l265, ann_key="pragma_unroll_explicit", ann_val=1)
        l277, l278, l279, l280, l281, l282, l283, l284 = sch.get_loops(block=b232)
        l285, l286, l287, l288, l289, l290, l291, l292 = sch.get_loops(block=b233)
        l293, l294, l295 = sch.get_loops(block=b234)
        sch.annotate(block_or_loop=l293, ann_key="pragma_auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l293, ann_key="pragma_unroll_explicit", ann_val=1)
        b296 = sch.get_block(name="matmul_rf_o", func_name="main")
        l297, l298, l299, l300, l301, l302, l303, l304, l305, l306, l307, l308 = sch.get_loops(block=b296)
        b309 = sch.decompose_reduction(block=b296, loop=l300)
        sch.unannotate(block_or_loop=b309, ann_key="meta_schedule.auto_tensorize")
        sch.annotate(block_or_loop=b309, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
        sch.unannotate(block_or_loop=b296, ann_key="meta_schedule.auto_tensorize_init")
        sch.unannotate(block_or_loop=b309, ann_key="meta_schedule.auto_tensorize_init")
        b310 = sch.get_block(name="matmul", func_name="main")
        l311, l312, l313 = sch.get_loops(block=b310)
        b314 = sch.decompose_reduction(block=b310, loop=l313)
        b315 = sch.get_block(name="matmul_rf_o_init", func_name="main")
        sch.unannotate(block_or_loop=b315, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b315, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
        b316 = sch.get_block(name="lv1654_reindex_pad_shared.dyn_wmma.matrix_a_o", func_name="main")
        sch.unannotate(block_or_loop=b316, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b316, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
        b317 = sch.get_block(name="p_output0_intermediate_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
        sch.unannotate(block_or_loop=b317, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b317, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
        b318 = sch.get_block(name="matmul_rf_o_update", func_name="main")
        sch.unannotate(block_or_loop=b318, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b318, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
        b319 = sch.get_block(name="var_matmul_intermediate.rf_reindex_pad_shared.dyn_wmma.accumulator_o", func_name="main")
        sch.unannotate(block_or_loop=b319, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b319, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)

    B, N, K = 1, 22016, 4096
    GS = 128
    BLK = 8
    # func = gen_simple_qmm_dynm(N, K, GS, dim_dtype="int32")
    # func = matmul_g128_KN_sym
    func = matmul_g128_KN_sym_dynm
    mod = ms.tir_integration._normalize_mod(func)    

    USE_DLIGHT = False
    with TARGET:
        if USE_DLIGHT:
            import tvm.dlight as dl
            dl_mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
            dl_mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(dl_mod)
            dl_mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(dl_mod)
            lib = tvm.build(dl_mod)
        else:
            sch = tvm.tir.Schedule(mod)
            # apply_trace_m1(sch)
            # apply_trace_m63_v1(sch)
            apply_trace_m63_v2(sch)
            # apply_trace_m63_v3(sch)
            lib = tvm.build(sch.mod)

    # # TBR
    # mod = ms.tir_integration._normalize_mod(matmul_g128_KN_sym_m63)
    # sch = tvm.tir.Schedule(mod)
    # apply_trace_m63_v2(sch)
    # with TARGET:
    #     lib = tvm.build(sch.mod)

    # args_info = ms.arg_info.ArgInfo.from_prim_func(matmul_g128_KN_sym_m63)
    # args = [make_arg(info) for info in args_info]
    # score_s = lib.time_evaluator(lib.entry_name, dev=tvm.cuda(0), number=100, repeat=1, min_repeat_ms=2000)(*args).mean
    # print(f"M: 1 TIME: {score_s*1e6} us",)
    # exit()
    # # TBR


    # for M in range(16, 2048, 16):
    for M in range(4, 64, 4):
        args_info = [
            # == 3D case ==
            ms.arg_info.TensorInfo("uint32", [K // BLK, N]),  # WGH
            ms.arg_info.TensorInfo("float16", [K // GS, N]),  # SCL
            ms.arg_info.TensorInfo("float16", [B, M, K]),     # D_IN
            ms.arg_info.TensorInfo("float16", [B, M, N]),     # D_OUT
        ]
        args = [make_arg(info) for info in args_info]

        # from torch.profiler import profile, ProfilerActivity
        # with profile(activities=[ProfilerActivity.CUDA], with_modules=False, with_stack=True) as prof:
        if True:
            score_s = lib.time_evaluator(lib.entry_name, dev=tvm.cuda(0), number=100, repeat=1, min_repeat_ms=2000)(*args).mean
            print(f"M: {M} TIME: {score_s*1e6} us",)

        if 'prof' in locals(): 
            prof.export_chrome_trace(f"./trace_qmm_rf_tuned_m{M}.json")

    

if __name__ == "__main__":
    # main()
    # read_db()
    # read_db_topk()
    manual_sch()
