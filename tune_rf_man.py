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



@T.prim_func
def simple_matmul_dynm(p_output0_intermediate: T.Buffer((T.int64(4096), T.int64(22016)), "float16"), lv1654_hdl: T.handle, var_matmul_intermediate_hdl: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    m = T.int64()
    lv1654 = T.match_buffer(lv1654_hdl, (m, T.int64(4096)), "float16")
    var_matmul_intermediate = T.match_buffer(var_matmul_intermediate_hdl, (m, T.int64(22016)), "float16")

    # with T.block("root"):
    for i1, i2, k in T.grid(m, T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i1, v_i2, v_k = T.axis.remap("SSR", [i1, i2, k])
            T.reads(lv1654[v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i1, v_i2] = var_matmul_intermediate[v_i1, v_i2] + lv1654[v_i1, v_k] * p_output0_intermediate[v_k, v_i2]


@T.prim_func
def simple_matmul_3d_dynm(p_output0_intermediate: T.Buffer((T.int64(4096), T.int64(22016)), "float16"), lv1654_hdl: T.handle, var_matmul_intermediate_hdl: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    m = T.int64()
    lv1654 = T.match_buffer(lv1654_hdl, (T.int64(1), m, T.int64(4096)), "float16")
    var_matmul_intermediate = T.match_buffer(var_matmul_intermediate_hdl, (T.int64(1), m, T.int64(22016)), "float16")

    # with T.block("root"):
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
    sch.pad_einsum(block, padding=[M_PAD, 16, 16])
    b_pad_a = sch.get_producers(block)[0]
    b_pad_o = sch.get_consumers(block)[0]

    # block 16x16x16
    lm, ln, lk = sch.get_loops(block)
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
    lm_b3_1, lm = sch.split(lm, factors=[None, M_PAD//16])
    lm_factors = sch.sample_perfect_tile(loop=lm, n=5, max_innermost_factor=4, decision= [1, 2, 1, 1, 2])
    lm_b3_2, lm_b3_3, lm_b2_1, lm_b2_2, lm_b1 = sch.split(lm, factors=lm_factors)
    ln_factors = sch.sample_perfect_tile(loop=ln, n=6, max_innermost_factor=4, decision=[1, 2752, 4, 1, 1, 2])
    ln_b3_1, ln_b3_2, ln_b3_3, ln_b2_1, ln_b2_2, ln_b1 = sch.split(ln, factors=ln_factors)
    lk_factors = sch.sample_perfect_tile(loop=lk, n=2, max_innermost_factor=4, decision=[1, 4])
    lk_b2, lk_b1 = sch.split(lk, factors=lk_factors)
    #           |              Block 3                              |               Block 2                    |     Block 1        |
    #           |     Blk.y     |      Blk.x      |     Thr.y       |                                                               |
    sch.reorder(lm_b3_1, ln_b3_1, lm_b3_2, ln_b3_2, lm_b3_3, ln_b3_3, lm_b2_1, ln_b2_1, lm_b2_2, ln_b2_2, lk_b2, lk_b1, lm_b1, ln_b1)
    #           |  Mo       1   |    1       No   |    4        4   |   1        1        1        1       Ko      4      2      2  |
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

    def blk_tensorize(blk, intrin_name):
        *_, lm, ln = sch.get_loops(blk)
        lm, lm_b = sch.split(lm, factors=[None, 16])
        ln, ln_b = sch.split(ln, factors=[None, 16])
        sch.reorder(lm, ln, lm_b, ln_b)
        blk_16x16 = sch.blockize(lm_b)
        sch.tensorize(blk_16x16, intrin_name)
    
    # # tensorize load/store
    blk_tensorize(b_o_wmma, "wmma_store_16x16x16_f16_shared_dyn")
    blk_tensorize(b_a_wmma, "wmma_load_16x16x16_f16_a_shared_dyn")
    blk_tensorize(b_b_wmma, "wmma_load_16x16x16_f16_b_trans_shared_dyn")

    # # vectorize loadstore
    # def blk_vectorize(blk):
    #     *_, lm, ln = sch.get_loops(blk) 
    #     lmn = sch.fuse(lm, ln)
    #     if attr["double_ty"] == False:
    #         lmn, lmn_tx, lmn_v = sch.split(lmn, factors=[None, 32, 4])
    #         sch.bind(lmn_tx, thread_axis="threadIdx.x")
    #         sch.vectorize(lmn_v)
    #     else:
    #         lmn, lmn_ty, lmn_tx, lmn_v = sch.split(lmn, factors=[None, 8, 32, 4])  
    #         sch.bind(lmn_ty, thread_axis="threadIdx.y") # NB! Is it correct?? What does it means, double threadIdx.y specification??
    #         sch.bind(lmn_tx, thread_axis="threadIdx.x")     
    #         sch.vectorize(lmn_v)

    #     sch.storage_align(block=blk, buffer_index=0, axis=-2, factor=16, offset=8)   # NB! significant impact
    
    # blk_vectorize(b_o_shared)
    # blk_vectorize(b_a_shared)
    # blk_vectorize(b_b_shared)

    # fuse pad into main block
    sch.compute_inline(b_pad_a)
    sch.reverse_compute_inline(b_pad_o)


def apply_trace_man(sch: tvm.tir.Schedule, attr={}):
    mm_block = sch.get_block("matmul")
    sch.transform_block_layout(mm_block, index_map=lambda vb, vm, v_vn, v_vk: (vb + vm, v_vn, v_vk))

    ds_v1(sch, mm_block, attr)

    dec_block = sch.get_block("decode")
    sch.compute_inline(dec_block)


def apply_trace_dl(sch: tvm.tir.Schedule) -> None:
    b0 = sch.get_block(name="root", func_name="main")
    b1, = sch.get_child_blocks(b0)
    b2 = sch.reindex(block=b1, buffer=("read", 0))
    sch.transform_layout(block=b2, buffer=("write", 0), index_map=lambda i0, i1: (T.int64(0), i0, i1,), pad_value=None, assume_injective_transform=False)
    b3 = sch.reindex(block=b1, buffer=("read", 1))
    sch.transform_layout(block=b3, buffer=("write", 0), index_map=lambda i0, i1: (T.int64(0), i0, i1,), pad_value=None, assume_injective_transform=False)
    b4 = sch.reindex(block=b1, buffer=("write", 0))
    sch.transform_layout(block=b4, buffer=("read", 0), index_map=lambda i0, i1: (T.int64(0), i0, i1,), pad_value=None, assume_injective_transform=False)
    sch.transform_block_layout(block=b1, index_map=lambda i0, i1, i2: (T.int64(0), i0, i1, i2,))
    sch.pad_einsum(block=b1, padding=[1, 128, 128, 64])
    l5, l6, l7, l8 = sch.get_loops(block=b1)
    l9, l10 = sch.split(loop=l6, factors=[None, 16], preserve_unit_iters=True)
    l11, l12 = sch.split(loop=l7, factors=[None, 16], preserve_unit_iters=True)
    l13, l14 = sch.split(loop=l8, factors=[None, 16], preserve_unit_iters=True)
    sch.reorder(l9, l11, l13, l10, l12, l14)
    b15 = sch.blockize(target=l10, preserve_unit_iters=True)
    l16, l17, l18, l19 = sch.split(loop=l9, factors=[None, 1, 4, 2], preserve_unit_iters=True)  # M
    l20, l21, l22, l23 = sch.split(loop=l11, factors=[1, None, 4, 2], preserve_unit_iters=True) # N
    l24, l25 = sch.split(loop=l13, factors=[None, 4], preserve_unit_iters=True)                 # K
    #           Mo   1    1    No   N4   M4   Ko   K4   2    2
    #          |   Bx   |    By   |   Ty    |  
    sch.reorder(l16, l20, l17, l21, l22, l18, l24, l25, l19, l23)
    l26 = sch.fuse(l16, l20, preserve_unit_iters=True)
    l27 = sch.fuse(l17, l21, preserve_unit_iters=True)
    l28 = sch.fuse(l22, l18, preserve_unit_iters=True)
    sch.bind(loop=l5, thread_axis="blockIdx.z")
    sch.bind(loop=l26, thread_axis="blockIdx.x")
    sch.bind(loop=l27, thread_axis="blockIdx.y")
    sch.bind(loop=l28, thread_axis="threadIdx.y")
    b29 = sch.cache_read(block=b15, read_buffer_index=0, storage_scope="shared.dyn")
    sch.compute_at(block=b29, loop=l24, preserve_unit_loops=False, index=-1)
    l30, l31, l32, l33, l34, l35, l36 = sch.get_loops(block=b29)
    l37 = sch.fuse(l35, l36, preserve_unit_iters=True)
    l38, l39, l40, l41 = sch.split(loop=l37, factors=[None, 16, 32, 4], preserve_unit_iters=True)
    sch.bind(loop=l40, thread_axis="threadIdx.x")
    sch.bind(loop=l39, thread_axis="threadIdx.y")
    sch.vectorize(loop=l41)
    sch.storage_align(block=b29, buffer_index=0, axis=-2, factor=16, offset=8)
    b42 = sch.cache_read(block=b15, read_buffer_index=1, storage_scope="shared.dyn")
    sch.compute_at(block=b42, loop=l24, preserve_unit_loops=False, index=-1)
    l43, l44, l45, l46, l47, l48, l49 = sch.get_loops(block=b42)
    l50 = sch.fuse(l48, l49, preserve_unit_iters=True)
    l51, l52, l53, l54 = sch.split(loop=l50, factors=[None, 16, 32, 4], preserve_unit_iters=True)
    sch.bind(loop=l53, thread_axis="threadIdx.x")
    sch.bind(loop=l52, thread_axis="threadIdx.y")
    sch.vectorize(loop=l54)
    sch.storage_align(block=b42, buffer_index=0, axis=-2, factor=16, offset=8)
    b55, = sch.get_producers(block=b29)
    b56, = sch.get_producers(block=b55)
    sch.get_producers(block=b56)
    sch.compute_inline(block=b55)
    sch.compute_inline(block=b56)
    sch.get_producers(block=b29)
    b57, = sch.get_producers(block=b42)
    sch.get_producers(block=b57)
    sch.compute_inline(block=b57)
    sch.get_producers(block=b42)
    b58 = sch.cache_read(block=b15, read_buffer_index=0, storage_scope="wmma.matrix_a")
    b59 = sch.cache_read(block=b15, read_buffer_index=1, storage_scope="wmma.matrix_b")
    sch.compute_at(block=b58, loop=l25, preserve_unit_loops=False, index=-1)
    sch.compute_at(block=b59, loop=l25, preserve_unit_loops=False, index=-1)
    b60 = sch.cache_write(block=b15, write_buffer_index=0, storage_scope="shared.dyn")
    sch.storage_align(block=b60, buffer_index=0, axis=-2, factor=16, offset=4)
    b61 = sch.cache_write(block=b15, write_buffer_index=0, storage_scope="wmma.accumulator")
    sch.reverse_compute_at(block=b61, loop=l28, preserve_unit_loops=False, index=-1)
    sch.reverse_compute_at(block=b60, loop=l28, preserve_unit_loops=False, index=-1)
    l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b61)
    l68, l69 = sch.split(loop=l66, factors=[None, 16], preserve_unit_iters=True)
    l70, l71 = sch.split(loop=l67, factors=[None, 16], preserve_unit_iters=True)
    sch.reorder(l68, l70, l69, l71)
    b72 = sch.decompose_reduction(block=b15, loop=l24)
    b73, = sch.get_child_blocks(b72)
    l74, l75, l76, l77, l78, l79, l80, l81 = sch.get_loops(block=b58)
    l82, l83 = sch.split(loop=l80, factors=[None, 16], preserve_unit_iters=True)
    l84, l85 = sch.split(loop=l81, factors=[None, 16], preserve_unit_iters=True)
    sch.reorder(l82, l84, l83, l85)
    sch.unroll(loop=l82)
    sch.unroll(loop=l84)
    sch.tensorize(block_or_loop=l83, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
    l86, l87, l88, l89, l90, l91, l92, l93 = sch.get_loops(block=b59)
    l94, l95 = sch.split(loop=l92, factors=[None, 16], preserve_unit_iters=True)
    l96, l97 = sch.split(loop=l93, factors=[None, 16], preserve_unit_iters=True)
    sch.reorder(l94, l96, l95, l97)
    sch.unroll(loop=l94)
    sch.unroll(loop=l96)
    sch.tensorize(block_or_loop=l95, tensor_intrin="wmma_load_16x16x16_f16_b_trans_shared_dyn", preserve_unit_iters=True)
    l98, l99 = sch.get_loops(block=b73)
    l100, l101 = sch.get_loops(block=b73)
    sch.tensorize(block_or_loop=l100, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
    l102, l103, l104, l105, l106, l107, l108, l109 = sch.get_loops(block=b61)
    sch.tensorize(block_or_loop=l108, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)
    l110, l111, l112 = sch.get_loops(block=b1)
    sch.tensorize(block_or_loop=l110, tensor_intrin="wmma_sync_16x16x16_f16f16f16_trans", preserve_unit_iters=True)
    b113, = sch.get_consumers(block=b60)
    b114, = sch.get_consumers(block=b113)
    sch.get_consumers(block=b114)
    sch.compute_inline(block=b113)
    sch.reverse_compute_inline(block=b114)
    sch.get_consumers(block=b60)
    l115, l116, l117, l118, l119, l120 = sch.get_loops(block=b60)
    l121 = sch.fuse(l119, l120, preserve_unit_iters=True)
    l122, l123, l124 = sch.split(loop=l121, factors=[None, 32, 4], preserve_unit_iters=True)
    sch.bind(loop=l123, thread_axis="threadIdx.x")
    sch.vectorize(loop=l124)


def manual_sch():
    func = matmul_g128_KN_sym_dynm
    # func = simple_matmul_dynm
    # func = simple_matmul_3d_dynm
    mod = ms.tir_integration._normalize_mod(func)    

    with TARGET:
        import tvm.dlight as dl
        # dl_mod = ms.tir_integration._normalize_mod(func)    
        # dl_mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(dl_mod)
        # lib_dl = tvm.build(dl_mod)

        dl_mod = ms.tir_integration._normalize_mod(func)    
        dl_mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(dl_mod)
        lib_fb = tvm.build(dl_mod)

        sch = tvm.tir.Schedule(mod)
        apply_trace_man(sch, attr={"double_ty": True})
        lib = tvm.build(sch.mod)    

        # sch = tvm.tir.Schedule(mod)
        # apply_trace_man(sch, attr={"double_ty": False})
        # lib2 = tvm.build(sch.mod)    


    B, N, K = 1, 22016, 4096
    GS = 128
    BLK = 8
    M = 63
    args_info = [
        # == int4 case ==
        ms.arg_info.TensorInfo("uint32", [K // BLK, N]),  # WGH
        ms.arg_info.TensorInfo("float16", [K // GS, N]),  # SCL
        ms.arg_info.TensorInfo("float16", [B, M, K]),     # D_IN
        ms.arg_info.TensorInfo("float16", [B, M, N]),     # D_OUT
        # == SIMPLE MM 2D case ==
        # ms.arg_info.TensorInfo("float16", [K, N]),     # WGH
        # ms.arg_info.TensorInfo("float16", [M, K]),     # D_IN
        # ms.arg_info.TensorInfo("float16", [M, N]),     # D_OUT
        # == SIMPLE MM 3D case ==
        # ms.arg_info.TensorInfo("float16", [K, N]),     # WGH
        # ms.arg_info.TensorInfo("float16", [1, M, K]),  # D_IN
        # ms.arg_info.TensorInfo("float16", [1, M, N]),  # D_OUT
    ]
    args = [make_arg(info) for info in args_info]

    # check correctness
    lib(*args)
    res_1 = args[-1].numpy()

    # lib2(*args)
    # res_2 = args[-1].numpy()

    # lib_dl(*args)
    # res_dl = args[-1].numpy()

    lib_fb(*args)
    res_fb = args[-1].numpy()

    # print(res_1 - res_2)
    # print(res_dl - res_2)
    # print(res_dl - res_1)
    print(res_1 - res_fb)
    print("Fallback")
    print(res_fb)
    print("dtune res")
    print(res_1)

    # score_s = lib.time_evaluator(lib.entry_name, dev=tvm.cuda(0), number=100, repeat=1, min_repeat_ms=2000)(*args).mean
    # print(f"M: {M} TIME: {score_s*1e6} us",)


    

if __name__ == "__main__":
    manual_sch()
