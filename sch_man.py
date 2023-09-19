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


def mm_gen(M, N, K) -> tvm.tir.PrimFunc:
    a = te.placeholder([M, K], dtype="float16", name="A")
    b = te.placeholder([K, N], dtype="float16", name="A")
    
    k = te.reduce_axis((0, K), name="k")
    x = te.compute(
        [M, N],
        lambda m, n: te.sum(a[m, k] * b[k, n], axis=k),
        name="matmul"
    )
    return te.create_prim_func([a, b, x])


def ds_v1(sch: Schedule, block: BlockRV, attr) -> None: 
    # M_PAD = 64
    # pad dims
    # sch.pad_einsum(block, padding=[M_PAD, 16, 16])
    # b_pad_a = sch.get_producers(block)[0]
    # b_pad_o = sch.get_consumers(block)[0]

    wmma_space_acc, wmma_space_a, wmma_space_b = "wmma.accumulator", "wmma.matrix_a", "wmma.matrix_b"
    # wmma_space_acc, wmma_space_a, wmma_space_b = ("shared.dyn",)*3 

    # Extract block 16x16x16
    lm, ln, lk = sch.get_loops(block)
    lm, lm_b = sch.split(lm, factors=[None, 16])
    ln, ln_b = sch.split(ln, factors=[None, 16])
    lk, lk_b = sch.split(lk, factors=[None, 16])
    sch.reorder(lm, ln, lk, lm_b, ln_b, lk_b)
    b_wmma = sch.blockize(lm_b)

    lm_factors = sch.sample_perfect_tile(loop=lm, n=6, max_innermost_factor=4, decision= [1, 1, 2, 1, 1, 2])
    lm_b3_1, lm_b3_2, lm_b3_3, lm_b2_1, lm_b2_2, lm_b1 = sch.split(lm, factors=lm_factors)
    ln_factors = sch.sample_perfect_tile(loop=ln, n=6, max_innermost_factor=4, decision=[1, 1, 4, 1, 1, 2])
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
    b_o_wmma = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope=wmma_space_acc)
    sch.reverse_compute_at(b_o_wmma, loop=ln_b2_2, preserve_unit_loops=True, index=-1)
    sch.reverse_compute_at(b_o_shared, loop=ln_b2_2, preserve_unit_loops=True, index=-1)
    
    b_a_shared = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="shared.dyn")
    b_a_wmma = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope=wmma_space_a)
    sch.compute_at(b_a_wmma, loop=lk_b1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
    sch.compute_at(b_a_shared, loop=lk_b2, preserve_unit_loops=True, index=-1)

    b_b_shared = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="shared.dyn")
    b_b_wmma = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope=wmma_space_b)
    sch.compute_at(b_b_wmma, loop=lk_b1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
    sch.compute_at(b_b_shared, loop=lk_b2, preserve_unit_loops=True, index=-1)

    # # TODO: Like a workaround. have to do it after cache read/write. In case of fused axis they calculate wrong shapes.
    lnm_ty = sch.fuse(lm_b3_3, ln_b3_3)
    sch.bind(lnm_ty, thread_axis="threadIdx.y")

    # # move reduction init to L1 level
    b_wmma_init = sch.decompose_reduction(block=b_wmma, loop=lk_b2)

    def blk_tensorize(blk, intrin_name):
        # sch.annotate(blk, "warp_execution", 1)
        *_, lm, ln = sch.get_loops(blk)
        lm, lm_b = sch.split(lm, factors=[None, 16])
        ln, ln_b = sch.split(ln, factors=[None, 16])
        sch.reorder(lm, ln, lm_b, ln_b)
        blk_16x16 = sch.blockize(lm_b)
        sch.tensorize(blk_16x16, intrin_name)
    
    # tensorize load/store
    blk_tensorize(b_o_wmma, "wmma_store_16x16x16_f16_shared_dyn")
    blk_tensorize(b_a_wmma, "wmma_load_16x16x16_f16_a_shared_dyn")
    blk_tensorize(b_b_wmma, "wmma_load_16x16x16_f16_b_shared_dyn")   # TODO: It accepts "wmma_load_16x16x16_f16_b_trans_shared_dyn" as well.. problem

    # tensorize compute
    sch.tensorize(b_wmma, "wmma_sync_16x16x16_f16f16f16")  # why not trans
    sch.tensorize(b_wmma_init, "wmma_fill_16x16x16_f16")

    # vectorize loadstore
    def blk_vectorize(blk):
        *_, lm, ln = sch.get_loops(blk) 
        lmn = sch.fuse(lm, ln)
        lmn, lmn_tx, lmn_v = sch.split(lmn, factors=[None, 32, 4])
        sch.bind(lmn_tx, thread_axis="threadIdx.x")
        sch.vectorize(lmn_v)

        sch.storage_align(block=blk, buffer_index=0, axis=-2, factor=16, offset=8)   # NB! significant impact
    
    # blk_vectorize(b_o_shared)
    blk_vectorize(b_a_shared)
    # blk_vectorize(b_b_shared)

    # fuse pad into main block
    # sch.compute_inline(b_pad_a)
    # sch.reverse_compute_inline(b_pad_o)


def apply_trace_man(sch: tvm.tir.Schedule, attr={}):
    mm_block = sch.get_block("matmul")
    ds_v1(sch, mm_block, attr)


def reprod_1():
    """ Reproduce wrong compute """
    M, N, K = 128, 128, 128
    func = mm_gen(M, N, K)

    with TARGET:
        dl_mod = ms.tir_integration._normalize_mod(func)    
        dl_mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(dl_mod)
        # dl_mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(dl_mod)
        dl_lib = tvm.build(dl_mod)

        man_mod = ms.tir_integration._normalize_mod(func)
        man_sch = tvm.tir.Schedule(man_mod)
        apply_trace_man(man_sch, attr={"double_ty": True})
        man_lib = tvm.build(man_sch.mod)    

    args_info = ms.arg_info.ArgInfo.from_prim_func(func)
    args = [make_arg(info) for info in args_info]

    dl_lib(*args)
    res_ref = args[-1].numpy()

    man_lib(*args)
    res_man = args[-1].numpy()

    correct = np.allclose(res_man, res_ref, rtol=0.5, atol=0.1)
    if not correct:
        print("ERROR: Result is incorrect...")
        print("DIFF:")
        print((res_ref - res_man)/res_ref)
        print("REF:")
        print(res_ref)
        print("MAN:")
        print(res_man)

    else:
        print("PASS: Results are close!!")


if __name__ == "__main__":
    reprod_1()
