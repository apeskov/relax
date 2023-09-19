import numpy as np

import tvm
import tvm.meta_schedule as ms
import tvm.dlight as dl
import tvm.script.tir as T
from tvm import te
from tvm._ffi import get_global_func


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


@tvm.register_func("tvm_callback_cuda_compile", override=True)
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



def ds_v1(sch: Schedule, block: BlockRV, attr) -> None: 
    M_PAD = 64
    # pad dims
    sch.pad_einsum(block, padding=[1, M_PAD, 16, 16])
    b_pad_a = sch.get_producers(block)[0]
    b_pad_o = sch.get_consumers(block)[0]

    # block 16x16x16
    lb, lm, ln, lk = sch.get_loops(block)
    lm, lm_b = sch.split(lm, factors=[None, 16])
    ln, ln_b = sch.split(ln, factors=[None, 16])
    lk, lk_b = sch.split(lk, factors=[None, 16])
    sch.reorder(lm, ln, lk, lm_b, ln_b, lk_b)
    b_wmma = sch.blockize(lm_b)

    # pseudocode:
    #
    # def block_1(frg a[M_b1,K_b1], 
    #             frg b[K_b1,N_b1], 
    #             frg c[M_b1,N_b1]):
    #   assert M_b1*N_b1*K_b1 < 64   # tipically
    #   for m, n, k for M_b1, N_b1, K_b1:
    #     wmma::mul(a[m,k], b[k,n], c[m,n])
    #
    # def block_2(__global__ half a[M_b2,K_b2], 
    #             __global__ half b[K_b2,N_b2],
    #             __global__ half c[M_b2,N_b2]):
    #     assert K_b2*K_b1 == K  # full redaction processing on this level
    #     m_b2_1, m_b2_2 = split(M_b2//M_b1)
    #     n_b2_1, n_b2_2 = split(N_b2//N_b1)
    #    
    #     frg_a = frg_arr(M_b1,K_b1)
    #     frg_b = frg_arr(K_b1,N_b1)
    #     frg_c = frg_arr(M_b1,N_b1]
    #
    #     for m_b2_1, n_b2_1, m_b2_2, n_b2_2:  # allow MN, NM or mixed
    # 		_fill_zero(c)
    #       	for K_b2:
    #     		read_g2regs(a[...], frg_a)  # additional bind(thr.y) is here
    #      		read_g2regs(b[...], frg_b)  # additioanl bind(thr.y) is inside
    #      		block_1(frg_a, frg_b, frg_c)
    #   		write_reg2g(c[...], frg_c)
    #
    # def block_3(__global__ half a[M,K], 
    #             __global__ half b[K,N],
    #             __global__ half c[M,N]):
    #     m_b3_1, m_b3_2, m_b3_3 = split(M//M_b2)
    #     n_b3_1, n_b3_2, n_b3_3 = split(N//N_b2)
    #
    #     for m_b3_1, n_b3_1 <bind(blk.y)>:
    #       for m_b3_2, n_b3_2 <bind(blk.x)>:
    #         for m_b3_3, n_b3_3 <bind(thr.y)>:  # NB! inefficiency here. It doesn't use thread block capabilities.
    #           block_2(a[...], b[...], c[...])
    #
    #
    # Combined loop:
    #
    # m_b3_1, m_b3_2, m_b3_3, m_b2_1, m_b2_2, m_b1 = split(M)  
    # n_b3_1, n_b3_2, n_b3_3, n_b2_1, n_b2_2, n_b1 = split(N)
    # k_b2, k_b1 = split(K)
    #
    # for m_b3_1, n_b3_1 <bind(blk.y)>:                  | block 3
    #   for m_b3_2, n_b3_2 <bind(blk.x)>:                |  
    #     for m_b3_3, n_b3_3 <bind(thr.y)>:              |
    #       for for m_b2_1, n_b2_1, m_b2_2, n_b2_2:            |  block 2
    #         _fill_zero(acc_r)                                |
    #           for k_b2:                                      |
    #             _copy_g2s(a_g[...], a_s)                     |
    #             _copy_g2s(b_g[...], b_s)                     |
    #             _copy_s2r(a_s, a_r)                          |
    #             _copy_s2r(b_s, b_r)                          |
    #             for m_b1, n_b1, k_b1:                              | block 1
    #               wmma::mul(a_r[...], b_r[...], acc_r[...])        |
    #         _copy_r2s(acc_r, acc_s)                          |
    #         _copy_s2g(acc_s, c_g)                            |
    #

    # bad decisions
    # m_decision, n_decision, k_decision = [2, 1, 1, 2, 1], [1, 4, 86, 1, 4, 1], [256, 1]
    m_decision, n_decision, k_decision = [2, 1, 1, 1, 2], [4, 4, 43, 1, 1, 2], [64, 4]
    #           |              Block 3                              |               Block 2                    |     Block 1        |
    #           |     Blk.y     |      Blk.x      |     Thr.y       |                                                               |
    #           lm_b3_1, ln_b3_1, lm_b3_2, ln_b3_2, lm_b3_3, ln_b3_3, lm_b2_1, ln_b2_1, lm_b2_2, ln_b2_2, lk_b2, lk_b1, lm_b1, ln_b1)
    #           |  Mo       1   |    2        4   |    1       86   |   1        1        2        4       256      1      1      1  |


    # good decisions
    # m_decision, n_decision, k_decision = [1, 2, 1, 1, 2], [1, 172, 4, 1, 1, 2], [1, 4]
    #           |              Block 3                              |               Block 2                    |     Block 1        |
    #           |     Blk.y     |      Blk.x      |     Thr.y       |                                                               |
    #           lm_b3_1, ln_b3_1, lm_b3_2, ln_b3_2, lm_b3_3, ln_b3_3, lm_b2_1, ln_b2_1, lm_b2_2, ln_b2_2, lk_b2, lk_b1, lm_b1, ln_b1)
    #           |  Mo       1   |    1       No   |    4        4   |   1        1        1        1       Ko      4      2      2  |

    lm_b3_1, lm = sch.split(lm, factors=[None, M_PAD//16])
    lm_factors = sch.sample_perfect_tile(loop=lm, n=5, max_innermost_factor=4, decision=m_decision)
    lm_b3_2, lm_b3_3, lm_b2_1, lm_b2_2, lm_b1 = sch.split(lm, factors=lm_factors)
    ln_factors = sch.sample_perfect_tile(loop=ln, n=6, max_innermost_factor=4, decision=n_decision)
    ln_b3_1, ln_b3_2, ln_b3_3, ln_b2_1, ln_b2_2, ln_b1 = sch.split(ln, factors=ln_factors)
    lk_factors = sch.sample_perfect_tile(loop=lk, n=2, max_innermost_factor=4, decision=k_decision)
    lk_b2, lk_b1 = sch.split(lk, factors=lk_factors)
    sch.reorder(lm_b3_1, ln_b3_1, lm_b3_2, ln_b3_2, lm_b3_3, ln_b3_3, lm_b2_1, ln_b2_1, lm_b2_2, ln_b2_2, lk_b2, lk_b1, lm_b1, ln_b1)
    lnm_by = sch.fuse(lm_b3_1, ln_b3_1)
    sch.bind(lnm_by, thread_axis="blockIdx.y")
    lnm_bx = sch.fuse(lm_b3_2, ln_b3_2)
    sch.bind(lnm_bx, thread_axis="blockIdx.x")

    # # copy from/to shared on level of L1 block
    b_o_shared = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="shared.dyn")
    b_o_wmma = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="wmma.accumulator")
    sch.reverse_compute_at(b_o_wmma, loop=ln_b2_2, preserve_unit_loops=True, index=-1)
    sch.reverse_compute_at(b_o_shared, loop=ln_b2_2, preserve_unit_loops=True, index=-1)
    
    b_a_shared = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="shared.dyn")
    b_a_wmma = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="wmma.matrix_a")
    sch.compute_at(b_a_wmma, loop=lk_b1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
    sch.compute_at(b_a_shared, loop=lk_b2, preserve_unit_loops=True, index=-1)

    b_b_shared = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="shared.dyn")
    b_b_wmma = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="wmma.matrix_b")
    sch.compute_at(b_b_wmma, loop=lk_b1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
    sch.compute_at(b_b_shared, loop=lk_b2, preserve_unit_loops=True, index=-1)

    # # TODO: Like a workaround. have to do it after cache read/write. In case of fused axis they calculate wrong shapes.
    lnm_ty = sch.fuse(lm_b3_3, ln_b3_3)
    sch.bind(lnm_ty, thread_axis="threadIdx.y")

    # # move reduction init to L1 level
    b_wmma_init = sch.decompose_reduction(block=b_wmma, loop=lk_b2)

    # # tensorize compute
    sch.tensorize(b_wmma, "wmma_sync_16x16x16_f16f16f16")  # why not trans
    sch.tensorize(b_wmma_init, "wmma_fill_16x16x16_f16")
    # sch.annotate(b_wmma, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")
    # sch.annotate(b_wmma_init, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")

    def blk_tensorize(blk, intrin_name):
        *_, lm, ln = sch.get_loops(blk)
        lm, lm_b = sch.split(lm, factors=[None, 16])
        ln, ln_b = sch.split(ln, factors=[None, 16])
        sch.reorder(lm, ln, lm_b, ln_b)
        blk_16x16 = sch.blockize(lm_b)
        sch.tensorize(blk_16x16, intrin_name)
        # sch.annotate(blk_16x16, ann_key="meta_schedule.auto_tensorize", ann_val=intrin_name)
    
    # # tensorize load/store
    blk_tensorize(b_o_wmma, "wmma_store_16x16x16_f16_shared_dyn")
    blk_tensorize(b_a_wmma, "wmma_load_16x16x16_f16_a_shared_dyn")
    blk_tensorize(b_b_wmma, "wmma_load_16x16x16_f16_b_shared_dyn")   # TODO: It accepts "wmma_load_16x16x16_f16_b_trans_shared_dyn" as well.. problem

    # vectorize loadstore
    def blk_vectorize(blk):
        *_, lm, ln = sch.get_loops(blk) 
        lmn = sch.fuse(lm, ln)
        lmn, lmn_tx, lmn_v = sch.split(lmn, factors=[None, 32, 4])
        sch.bind(lmn_tx, thread_axis="threadIdx.x")
        sch.vectorize(lmn_v)

        # NB! significant impact. Looks like bank conflict. "buffer_index=0" for cache write, is it correct? 
        sch.storage_align(block=blk, buffer_index=0, axis=-2, factor=16, offset=8)   
    
    blk_vectorize(b_o_shared)
    blk_vectorize(b_a_shared)
    blk_vectorize(b_b_shared)

    # TODO: Should be before tensorization. Otherwise compute_inline doesn't work
    # fuse pad into main block  
    sch.compute_inline(b_pad_a)
    sch.reverse_compute_inline(b_pad_o)



def apply_trace_man(sch: tvm.tir.Schedule, attr={}):
    mm_block = sch.get_block("matmul")
    # Make 2D matmul
    # sch.transform_block_layout(mm_block, index_map=lambda vb, vm, v_vn, v_vk: (vb + vm, v_vn, v_vk))

    ds_v1(sch, mm_block, attr)

    dec_block = sch.get_block("decode")
    sch.compute_inline(dec_block)


def tune():
    # mod = tvm.IRModule({"matmul_g128_KN_sym_dynm": matmul_g128_KN_sym_dynm})
    mod = tvm.IRModule({"matmul_g128_KN_sym": matmul_g128_KN_sym})

    ms.tir_integration.tune_tir(
        mod=mod,
        target=TARGET,
        work_dir="__tmp/tune_sch_man",
        max_trials_global=100500,
        max_trials_per_task=4096,
        num_trials_per_iter=32,
        cost_model="random",
        # cost_model="xgb",
        space=ms.space_generator.ScheduleFn(
            apply_trace_man,
            sch_rules=[],
            # postprocs=[],
            mutator_probs={},
        )
    )


def check():
    func = matmul_g128_KN_sym
    sch = tvm.tir.Schedule(func) 
    
    apply_trace_man(sch)
    
    with TARGET:
        lib = tvm.build(sch.mod)    

if __name__ == "__main__":
    # tune()
    check()
