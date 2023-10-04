import numpy as np
import subprocess
from pathlib import Path

import tvm
import tvm.meta_schedule as ms
import tvm.dlight as dl
import tvm.script.tir as T
from tvm import te
from tvm.contrib import utils

from tvm.tir.schedule import BlockRV, Schedule

import tvm.tir.tensor_intrin.cuda
from tvm.contrib import nvcc

TARGET = tvm.target.Target("nvidia/nvidia-a10g")
DEV = tvm.cuda(0)

def make_arg(info):
    if info.dtype in ["float16", "float32"]:
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype(info.dtype)
    elif info.dtype in ["int32", "uint32", "int16", "int8"]:
        arr_np = np.random.randint(0, 16, size=info.shape).astype(info.dtype)
    else:
        assert False, f"Unimplemented, dtype={info.dtype}"

    return tvm.nd.array(arr_np, device=DEV)

def get_sass(cubin):
    temp = utils.tempdir()
    temp_cubin = temp.relpath("my_kernel.cubin")
    with open(temp_cubin, "wb") as out_file:
        out_file.write(cubin)
    
    cmd = [ "nvdisasm", "-c", temp_cubin]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg += "\nCompilation error:\n"
        msg += out.decode("utf-8")
        raise RuntimeError(msg)

    return out.decode("utf-8")


def cuda_dump(lib, dump_path="."):
    src = lib.imported_modules[0].get_source()
    with open(f"{dump_path}/shaders.cu", "w") as f:
        print(src, file=f)

    ptx = nvcc.compile_cuda(src, target_format="ptx")
    with open(f"{dump_path}/shaders.ptx", "wb") as f:
        f.write(ptx)

    cubin = nvcc.compile_cuda(src, target_format="cubin")
    # with open(f"{dump_path}/shaders.cubin", "wb") as f:
        # f.write(cubin)

    sass = get_sass(cubin)
    with open(f"{dump_path}/shaders.sass", "w") as f:
        f.write(sass)


# @tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code, target):
    """Use nvcc compiler for better perf."""
    with open("code.cu", "w") as f:
        f.write(code)

    ptx = nvcc.compile_cuda(code, target_format="ptx")
    return ptx


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



def ds_v1(sch: Schedule, block: BlockRV, m_pad, desicions) -> None: 
    # pad dims
    sch.pad_einsum(block, padding=[1, m_pad, 16, 16])
    b_pad_a = sch.get_producers(block)[0]
    b_pad_o = sch.get_consumers(block)[0]

    # block 16x16x16
    lb, lm, ln, lk = sch.get_loops(block)
    lm, lm_b = sch.split(lm, factors=[None, 16])
    ln, ln_b = sch.split(ln, factors=[None, 16])
    lk, lk_b = sch.split(lk, factors=[None, 16])
    sch.reorder(lm, ln, lk, lm_b, ln_b, lk_b)
    b_wmma = sch.blockize(lm_b)

    # Combined loop:
    #
    # m_b3_1, m_b3_2, m_b3_3, m_b2_1, m_b2_2, m_b1 = split(M)  
    # n_b3_1, n_b3_2, n_b3_3, n_b2_1, n_b2_2, n_b1 = split(N)
    # k_b2, k_b1 = split(K)
    #
    # for m_b3_1, n_b3_1 <bind(blk.y)>:                  | block 3
    #   for m_b3_2, n_b3_2 <bind(blk.x)>:                |  
    #     for m_b3_3, n_b3_3 <bind(thr.y)>:              |
    #         _fill_zero(acc_r)                              |
    #           for k_b2:                                    |
    #             _copy_g2s(a_g[...], a_s)                   |
    #             _copy_g2s(b_g[...], b_s)                   |
    #             for m_b1, n_b1, k_b1:                          | block 1
    #               _copy_s2r(a_s, a_r)                          |
    #               _copy_s2r(b_s, b_r)                          |
    #               wmma::mul(a_r[...], b_r[...], acc_r[...])    |
    #         _copy_r2s(acc_r, acc_s)                        |
    #         _copy_s2g(acc_s, c_g)                          |
    #

    # bad decisions
    if len(desicions) == 0:
        m_decision, n_decision, k_decision = [1, 2, 2], [1, 2, 2, 2], [1, 4]
    else:
        m_decision = desicions["m_split"]
        n_decision = desicions["n_split"]
        k_decision = desicions["k_split"]
        
    #           |              Block 3             |        Block 1        |
    #           |   Blk.y  |   Blk.x   |   Thr.y   |                       |
    #           (lm_4, ln_4, lm_3, ln_3, lm_2, ln_2, lk_2, lk_1, lm_1, ln_1)
    #           |   Mo  1  |   2    4  |   2     2 |  Ko     4     2    2  |
    lm_4, lm = sch.split(lm, factors=[None, m_pad//16])
    lm_factors = sch.sample_perfect_tile(loop=lm, n=3, max_innermost_factor=4, decision=m_decision)
    lm_3, lm_2, lm_1 = sch.split(lm, factors=lm_factors)
    ln_factors = sch.sample_perfect_tile(loop=ln, n=4, max_innermost_factor=4, decision=n_decision)
    ln_4, ln_3, ln_2, ln_1 = sch.split(ln, factors=ln_factors)
    lk_factors = sch.sample_perfect_tile(loop=lk, n=2, max_innermost_factor=4, decision=k_decision)
    lk_2, lk_1 = sch.split(lk, factors=lk_factors)
    sch.reorder(lm_4, ln_4, lm_3, ln_3, lm_2, ln_2, lk_2, lk_1, lm_1, ln_1)
    lnm_by = sch.fuse(lm_4, ln_4)
    sch.bind(lnm_by, thread_axis="blockIdx.y")
    lnm_bx = sch.fuse(lm_3, ln_3)
    sch.bind(lnm_bx, thread_axis="blockIdx.x")
    lnm_ty = sch.fuse(lm_2, ln_2)
    sch.bind(lnm_ty, thread_axis="threadIdx.y")

    # copy from/to shared on level of L1 block
    b_o_shared = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="shared.dyn")
    b_o_wmma = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="wmma.accumulator")
    sch.reverse_compute_at(b_o_wmma, loop=lnm_ty, preserve_unit_loops=True, index=-1)
    sch.reverse_compute_at(b_o_shared, loop=lnm_ty, preserve_unit_loops=True, index=-1)
    
    b_a_shared = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="shared.dyn")
    b_a_wmma = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="wmma.matrix_a")
    sch.compute_at(b_a_wmma, loop=lk_1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
    sch.compute_at(b_a_shared, loop=lk_1, preserve_unit_loops=True, index=-1)

    b_b_shared = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="shared.dyn")
    b_b_wmma = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="wmma.matrix_b")
    sch.compute_at(b_b_wmma, loop=lk_1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
    sch.compute_at(b_b_shared, loop=lk_1, preserve_unit_loops=True, index=-1)

    b_wmma_init = sch.decompose_reduction(block=b_wmma, loop=lk_2)

    # tensorize compute
    sch.tensorize(b_wmma, "wmma_sync_16x16x16_f16f16f16")
    sch.tensorize(b_wmma_init, "wmma_fill_16x16x16_f16")

    def blk_tensorize(blk, intrin_name):
        *_, lm, ln = sch.get_loops(blk)
        lm, lm_b = sch.split(lm, factors=[None, 16])
        ln, ln_b = sch.split(ln, factors=[None, 16])
        sch.reorder(lm, ln, lm_b, ln_b)
        blk_16x16 = sch.blockize(lm_b)
        # TODO: add bind to Ty???
        sch.tensorize(blk_16x16, intrin_name)
    
    # tensorize load/store
    blk_tensorize(b_o_wmma, "wmma_store_16x16x16_f16_shared_dyn")
    blk_tensorize(b_a_wmma, "wmma_load_16x16x16_f16_a_shared_dyn")
    blk_tensorize(b_b_wmma, "wmma_load_16x16x16_f16_b_shared_dyn")   # TODO: It accepts "wmma_load_16x16x16_f16_b_trans_shared_dyn" as well.. problem

    # vectorize loadstore
    def blk_vectorize(blk, vec_size=4):
        # 16x16 4*32*Ty
        # Ideally it should be 8 (128bit register containd 8 half floats) 
        ty_size = (lm_factors[-2] * ln_factors[-2])  # TODO: error "Stringifying is not supported for type: tir.Mul"
        tx_size = 32
        *_, lm, ln = sch.get_loops(blk) 
        lmn = sch.fuse(lm, ln)
        # lmn, lmn_ty, lmn_tx, lmn_v = sch.split(lmn, factors=[None, ty_size, tx_size, vec_size])
        lmn, lm_ty, ln_ty_2, lmn_tx, lmn_v = sch.split(lmn, factors=[None, lm_factors[-2], ln_factors[-2], tx_size, vec_size])
        sch.bind(lmn_tx, thread_axis="threadIdx.x")
        sch.bind(sch.fuse(lm_ty, ln_ty_2), thread_axis="threadIdx.y")
        sch.vectorize(lmn_v)

        # NB! significant impact. Looks like bank conflict. "buffer_index=0" for cache write, is it correct? 
        sch.storage_align(block=blk, buffer_index=0, axis=-2, factor=16, offset=8)   

    blk_vectorize(b_o_shared, vec_size=4)
    blk_vectorize(b_a_shared, vec_size=4)
    blk_vectorize(b_b_shared, vec_size=4)

    sch.compute_inline(b_pad_a)
    sch.reverse_compute_inline(b_pad_o)


def mds1(m_pad = 32, decisions = {}):
    if not decisions:
        decisions = {
            "m_split": [1, 2, 2], 
            "n_split": [1, 2, 2, 2], 
            "k_split": [1, 4]
        }

    def gen_ds(sch: tvm.tir.Schedule):
        mm_block = sch.get_block("matmul")
        ds_v1(sch, mm_block, m_pad, decisions)

        dec_block = sch.get_block("decode")
        sch.compute_inline(dec_block)

    return gen_ds


def tune():
    B, M, N, K, BLK, GS = 1, 63, 22016, 4096, 8, 128
    M = 64

    args_info = [
        ms.arg_info.TensorInfo("uint32", [K // BLK, N]),  # WGH
        ms.arg_info.TensorInfo("float16", [K // GS, N]),  # SCL
        ms.arg_info.TensorInfo("float16", [B, M, K]),     # D_IN
        ms.arg_info.TensorInfo("float16", [B, M, N]),     # D_OUT
    ]
    args_info = [info.as_json() for info in  args_info]
    
    func = matmul_g128_KN_sym_dynm.with_attr({"metaschedule.arg_info_hint": args_info})
    mod = tvm.IRModule({"matmul_g128_KN_sym_dynm": func})

    ms.tir_integration.tune_tir(
        mod=mod,
        target=TARGET,
        work_dir=f"__tmp/tune_sch_man_mds_{M}",
        max_trials_global=100500,
        max_trials_per_task=4096,
        num_trials_per_iter=32,
        cost_model="random",
        # cost_model="xgb",
        space=ms.space_generator.ScheduleFn(
            mds1(m_pad=M),
            sch_rules=[],
            postprocs=[],
            mutator_probs={},
        )
    )


if __name__ == "__main__":
    tune()
