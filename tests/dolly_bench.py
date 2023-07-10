
import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm.contrib import nvcc

from dolly_tune import mutate_to_dyn_m
from dolly_single import construct_te

def apply_trace_tensor_cores(sch: tvm.tir.Schedule) -> None:
    b0 = sch.get_block(name="zeros_decode", func_name="main")
    b1 = sch.get_block(name="B_decode", func_name="main")
    b2 = sch.get_block(name="matmul", func_name="main")
    b3 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b2, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
    b4 = sch.reindex(block=b2, buffer=("write", 0))
    b5 = sch.reindex(block=b2, buffer=("read", 0))
    b6 = sch.reindex(block=b2, buffer=("read", 1))
    sch.transform_layout(block=b2, buffer=("read", 0), index_map=lambda vi, vk: (vi, vk,), pad_value=None, assume_injective_transform=False)
    sch.transform_layout(block=b2, buffer=("read", 1), index_map=lambda vj, vk: (vk, vj,), pad_value=None, assume_injective_transform=False)
    sch.transform_layout(block=b2, buffer=("write", 0), index_map=lambda vi, vj: (vi, vj,), pad_value=None, assume_injective_transform=False)
    sch.transform_block_layout(block=b4, index_map=lambda vi, vj: (vi, vj,))
    sch.transform_block_layout(block=b5, index_map=lambda vi, vk: (vi, vk,))
    sch.transform_block_layout(block=b6, index_map=lambda vj, vk: (vk, vj,))
    sch.transform_block_layout(block=b2, index_map=lambda vi, vj, vk: (vi, vj, vk,))
    
    sch.pad_einsum(b2, (32, 16, 16))   # <= XXX

    l7, l8, l9 = sch.get_loops(block=b2)
    l10, l11 = sch.split(loop=l9, factors=[None, 16], preserve_unit_iters=True)
    l12, l13 = sch.split(loop=l8, factors=[None, 16], preserve_unit_iters=True)
    l14, l15 = sch.split(loop=l7, factors=[None, 16], preserve_unit_iters=True)
    l16, l17, l18, l19, l20, l21 = sch.get_loops(block=b2)
    sch.reorder(l18, l20, l15, l13, l11)
    b22 = sch.blockize(target=l15, preserve_unit_iters=True)
    sch.annotate(block_or_loop=b22, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")
    sch.annotate(block_or_loop=b22, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")
    sch.annotate(block_or_loop=b22, ann_key="warp_execution", ann_val=1)
    l23, l24, l25 = sch.get_loops(block=b22)
    v26, v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l23, n=5, max_innermost_factor=4, decision=[1, 1, 2, 1, 1])  # <= XXX  [1, 1, 2, 1, 1]
    l31, l32, l33, l34, l35 = sch.split(loop=l23, factors=[v26, v27, v28, v29, v30], preserve_unit_iters=True)
    v36, v37, v38, v39, v40 = sch.sample_perfect_tile(loop=l24, n=5, max_innermost_factor=4, decision=[1, 320, 1, 1, 1])
    l41, l42, l43, l44, l45 = sch.split(loop=l24, factors=[v36, v37, v38, v39, v40], preserve_unit_iters=True)
    v46, v47, v48 = sch.sample_perfect_tile(loop=l25, n=3, max_innermost_factor=4, decision=[40, 2, 4])
    l49, l50, l51 = sch.split(loop=l25, factors=[v46, v47, v48], preserve_unit_iters=True)
    sch.reorder(l31, l41, l32, l42, l33, l43, l49, l50, l34, l44, l51, l35, l45)
    l52 = sch.fuse(l31, l41, preserve_unit_iters=True)
    sch.bind(loop=l52, thread_axis="blockIdx.y")
    l53 = sch.fuse(l32, l42, preserve_unit_iters=True)
    sch.bind(loop=l53, thread_axis="blockIdx.x")
    l54 = sch.fuse(l33, l43, preserve_unit_iters=True)
    sch.bind(loop=l54, thread_axis="threadIdx.y")
    sch.annotate(block_or_loop=b22, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
    sch.annotate(block_or_loop=b22, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
    b55 = sch.cache_write(block=b22, write_buffer_index=0, storage_scope="shared.dyn")
    sch.reverse_compute_at(block=b55, loop=l53, preserve_unit_loops=True, index=-1)
    b56 = sch.cache_write(block=b22, write_buffer_index=0, storage_scope="wmma.accumulator")
    sch.reverse_compute_at(block=b56, loop=l54, preserve_unit_loops=True, index=-1)
    l57, l58, l59, l60 = sch.get_loops(block=b55)
    l61 = sch.fuse(l59, l60, preserve_unit_iters=True)
    v62 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
    sch.annotate(block_or_loop=b55, ann_key="meta_schedule.cooperative_fetch", ann_val=v62)
    sch.reverse_compute_inline(block=b4)
    l63, l64, l65, l66, l67 = sch.get_loops(block=b56)
    l68, l69 = sch.split(loop=l67, factors=[None, 16], preserve_unit_iters=True)
    l70, l71 = sch.split(loop=l66, factors=[None, 16], preserve_unit_iters=True)
    l72, l73, l74, l75, l76, l77, l78 = sch.get_loops(block=b56)
    sch.reorder(l77, l71, l69)
    b79 = sch.blockize(target=l71, preserve_unit_iters=True)
    sch.annotate(block_or_loop=b79, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")
    b80 = sch.cache_read(block=b22, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b22])
    sch.compute_at(block=b80, loop=l49, preserve_unit_loops=True, index=-1)
    l81, l82, l83, l84, l85, l86 = sch.get_loops(block=b80)
    l87 = sch.fuse(l85, l86, preserve_unit_iters=True)
    v88 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
    sch.annotate(block_or_loop=b80, ann_key="meta_schedule.cooperative_fetch", ann_val=v88)
    b89 = sch.cache_read(block=b22, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b22])
    sch.compute_at(block=b89, loop=l49, preserve_unit_loops=True, index=-1)
    l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b89)
    l96 = sch.fuse(l94, l95, preserve_unit_iters=True)
    v97 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
    sch.annotate(block_or_loop=b89, ann_key="meta_schedule.cooperative_fetch", ann_val=v97)
    b98 = sch.cache_read(block=b22, read_buffer_index=0, storage_scope="wmma.matrix_a")
    sch.compute_at(block=b98, loop=l50, preserve_unit_loops=True, index=-1)
    l99, l100, l101, l102, l103, l104, l105 = sch.get_loops(block=b98)
    l106, l107 = sch.split(loop=l105, factors=[None, 16], preserve_unit_iters=True)
    l108, l109 = sch.split(loop=l104, factors=[None, 16], preserve_unit_iters=True)
    l110, l111, l112, l113, l114, l115, l116, l117, l118 = sch.get_loops(block=b98)
    sch.reorder(l117, l109, l107)
    b119 = sch.blockize(target=l109, preserve_unit_iters=True)
    sch.annotate(block_or_loop=b119, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")
    b120 = sch.cache_read(block=b22, read_buffer_index=1, storage_scope="wmma.matrix_b")
    sch.compute_at(block=b120, loop=l50, preserve_unit_loops=True, index=-1)
    l121, l122, l123, l124, l125, l126, l127 = sch.get_loops(block=b120)
    l128, l129 = sch.split(loop=l127, factors=[None, 16], preserve_unit_iters=True)
    l130, l131 = sch.split(loop=l126, factors=[None, 16], preserve_unit_iters=True)
    l132, l133, l134, l135, l136, l137, l138, l139, l140 = sch.get_loops(block=b120)
    sch.reorder(l139, l131, l129)
    b141 = sch.blockize(target=l131, preserve_unit_iters=True)
    sch.annotate(block_or_loop=b141, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")
    sch.compute_inline(block=b5)
    sch.compute_inline(sch.get_block("A_reindex_pad"))  # inline pad A, new one
    sch.compute_inline(block=b6)
    sch.storage_align(block=b80, buffer_index=0, axis=-2, factor=32, offset=8)
    sch.storage_align(block=b89, buffer_index=0, axis=-2, factor=32, offset=8)
    sch.compute_inline(block=b1)
    sch.compute_inline(block=b0)
    sch.reverse_compute_inline(sch.get_block("C_reindex_pad"))  # inline pad C, new one
    v142 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=1)
    sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v142)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b55, ann_key="meta_schedule.cooperative_fetch")
    l143, l144, l145 = sch.get_loops(block=b55)
    l146, l147, l148, l149 = sch.split(loop=l145, factors=[None, 2, 32, 4], preserve_unit_iters=True)
    sch.vectorize(loop=l149)
    sch.bind(loop=l148, thread_axis="threadIdx.x")
    sch.bind(loop=l147, thread_axis="threadIdx.y")
    sch.unannotate(block_or_loop=b80, ann_key="meta_schedule.cooperative_fetch")
    l150, l151, l152, l153, l154 = sch.get_loops(block=b80)
    l155, l156, l157, l158 = sch.split(loop=l154, factors=[None, 2, 32, 2], preserve_unit_iters=True)
    sch.vectorize(loop=l158)
    sch.bind(loop=l157, thread_axis="threadIdx.x")
    sch.bind(loop=l156, thread_axis="threadIdx.y")
    sch.unannotate(block_or_loop=b89, ann_key="meta_schedule.cooperative_fetch")
    l159, l160, l161, l162, l163 = sch.get_loops(block=b89)
    l164, l165, l166 = sch.split(loop=l163, factors=[None, 2, 32], preserve_unit_iters=True)
    sch.bind(loop=l166, thread_axis="threadIdx.x")
    sch.bind(loop=l165, thread_axis="threadIdx.y")
    b167 = sch.get_block(name="root", func_name="main")
    sch.unannotate(block_or_loop=b167, ann_key="meta_schedule.unroll_explicit")
    b168, b169, b170, b171, b172, b173, b174 = sch.get_child_blocks(b167)
    l175, l176, l177, l178, l179, l180, l181, l182 = sch.get_loops(block=b168)
    l183, l184, l185, l186, l187, l188, l189 = sch.get_loops(block=b169)
    l190, l191, l192, l193, l194, l195, l196 = sch.get_loops(block=b170)
    l197, l198, l199, l200, l201, l202, l203 = sch.get_loops(block=b171)
    l204, l205, l206, l207, l208, l209, l210, l211, l212, l213 = sch.get_loops(block=b172)
    sch.annotate(block_or_loop=l204, ann_key="pragma_auto_unroll_max_step", ann_val=16)
    sch.annotate(block_or_loop=l204, ann_key="pragma_unroll_explicit", ann_val=1)
    l214, l215, l216, l217, l218 = sch.get_loops(block=b173)
    l219, l220, l221, l222, l223, l224 = sch.get_loops(block=b174)
    b225 = sch.get_block(name="matmul_o", func_name="main")
    l226, l227, l228, l229, l230, l231, l232, l233, l234, l235 = sch.get_loops(block=b225)
    b236 = sch.decompose_reduction(block=b225, loop=l229)
    sch.unannotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize")
    sch.annotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
    sch.unannotate(block_or_loop=b225, ann_key="meta_schedule.auto_tensorize_init")
    sch.unannotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize_init")
    b237 = sch.get_block(name="matmul_o_init", func_name="main")
    sch.unannotate(block_or_loop=b237, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b237, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
    b238 = sch.get_block(name="A_reindex_pad_shared.dyn_wmma.matrix_a_o", func_name="main")
    sch.unannotate(block_or_loop=b238, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b238, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
    b239 = sch.get_block(name="B_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
    sch.unannotate(block_or_loop=b239, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b239, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
    b240 = sch.get_block(name="matmul_o_update", func_name="main")
    sch.unannotate(block_or_loop=b240, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b240, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
    b241 = sch.get_block(name="C_reindex_pad_shared.dyn_wmma.accumulator_o", func_name="main")
    sch.unannotate(block_or_loop=b241, ann_key="meta_schedule.auto_tensorize")
    sch.tensorize(block_or_loop=b241, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)


def apply_trace_cuda_cores(sch: tvm.tir.Schedule) -> None:
  b0 = sch.get_block(name="zeros_decode", func_name="main")
  b1 = sch.get_block(name="B_decode", func_name="main")
  b2 = sch.get_block(name="matmul", func_name="main")
  b3 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b2, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  l4, l5, l6 = sch.get_loops(block=b2)
  v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
  l12, l13, l14, l15, l16 = sch.split(loop=l4, factors=[v7, v8, v9, v10, v11], preserve_unit_iters=True)
  v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[160, 1, 32, 1, 1])
  l22, l23, l24, l25, l26 = sch.split(loop=l5, factors=[v17, v18, v19, v20, v21], preserve_unit_iters=True)
  v27, v28, v29 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64, decision=[32, 10, 16])
  l30, l31, l32 = sch.split(loop=l6, factors=[v27, v28, v29], preserve_unit_iters=True)
  sch.reorder(l12, l22, l13, l23, l14, l24, l30, l31, l15, l25, l32, l16, l26)
  l33 = sch.fuse(l12, l22, preserve_unit_iters=True)
  sch.bind(loop=l33, thread_axis="blockIdx.x")
  l34 = sch.fuse(l13, l23, preserve_unit_iters=True)
  sch.bind(loop=l34, thread_axis="vthread.x")
  l35 = sch.fuse(l14, l24, preserve_unit_iters=True)
  sch.bind(loop=l35, thread_axis="threadIdx.x")
  sch.annotate(block_or_loop=b2, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
  sch.annotate(block_or_loop=b2, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
  b36 = sch.cache_write(block=b2, write_buffer_index=0, storage_scope="local")
  sch.reverse_compute_at(block=b36, loop=l35, preserve_unit_loops=True, index=-1)
  b37 = sch.cache_read(block=b2, read_buffer_index=0, storage_scope="shared", consumer_blocks=[b2])
  sch.compute_at(block=b37, loop=l30, preserve_unit_loops=True, index=-1)
  l38, l39, l40, l41, l42, l43 = sch.get_loops(block=b37)
  l44 = sch.fuse(l42, l43, preserve_unit_iters=True)
  v45 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
  sch.annotate(block_or_loop=b37, ann_key="meta_schedule.cooperative_fetch", ann_val=v45)
  b46 = sch.cache_read(block=b2, read_buffer_index=1, storage_scope="shared", consumer_blocks=[b2])
  sch.compute_at(block=b46, loop=l30, preserve_unit_loops=True, index=-1)
  l47, l48, l49, l50, l51, l52 = sch.get_loops(block=b46)
  l53 = sch.fuse(l51, l52, preserve_unit_iters=True)
  v54 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
  sch.annotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch", ann_val=v54)
  sch.compute_inline(block=b1)
  sch.compute_inline(block=b0)
  v55 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
  sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v55)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b37, ann_key="meta_schedule.cooperative_fetch")
  l56, l57, l58, l59, l60 = sch.get_loops(block=b37)
  l61, l62 = sch.split(loop=l60, factors=[None, 32], preserve_unit_iters=True)
  sch.bind(loop=l62, thread_axis="threadIdx.x")
  sch.unannotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch")
  l63, l64, l65, l66, l67 = sch.get_loops(block=b46)
  l68, l69 = sch.split(loop=l67, factors=[None, 32], preserve_unit_iters=True)
  sch.bind(loop=l69, thread_axis="threadIdx.x")
  b70 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b70, ann_key="meta_schedule.unroll_explicit")
  b71, b72, b73, b74 = sch.get_child_blocks(b70)
  l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b71)
  l81, l82, l83, l84, l85, l86 = sch.get_loops(block=b72)
  l87, l88, l89, l90, l91, l92, l93, l94, l95, l96 = sch.get_loops(block=b73)
  sch.annotate(block_or_loop=l87, ann_key="pragma_auto_unroll_max_step", ann_val=512)
  sch.annotate(block_or_loop=l87, ann_key="pragma_unroll_explicit", ann_val=1)
  l97, l98, l99, l100, l101 = sch.get_loops(block=b74)
  b102 = sch.get_block(name="matmul", func_name="main")
  l103, l104, l105, l106, l107, l108, l109, l110, l111, l112 = sch.get_loops(block=b102)
  b113 = sch.decompose_reduction(block=b102, loop=l106)


def generate_arg(info: ms.arg_info.ArgInfo, dev):
    if info.dtype == "float16":
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype('float16')
    elif info.dtype == "int32":
        arr_np = np.random.randint(0, 16, size=info.shape).astype('int32')
    else:
        assert False, "Unimplemented"

    return tvm.nd.array(arr_np, device=dev)

@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code, target):
    ptx = nvcc.compile_cuda(code, target_format="ptx")
    return ptx

def main_2():
    target = tvm.target.Target("nvidia/nvidia-a10g")
    dev = tvm.cuda(0)

    HS = 5120
    M, N, K, G = 1, HS, HS, HS // 128
    
    mod = get_matmul_int4(M, N, K, G)
    sch = tvm.tir.Schedule(mod)
    apply_trace_cuda_cores(sch)

    print(sch.mod)

    with target:
        lib = tvm.build(sch.mod["main"])

    print(lib.imported_modules[0].get_source())
    # print(lib.get_source("ll"))
    # print(lib.get_source("asm"))
    # print(lib.get_source(""))

    # args_info = [
    #     ms.arg_info.TensorInfo(shape=[M, K], dtype="float16"),
    #     ms.arg_info.TensorInfo(shape=[K//8, N], dtype="int32"),
    #     ms.arg_info.TensorInfo(shape=[G, N], dtype="float16"),
    #     ms.arg_info.TensorInfo(shape=[G, N//8], dtype="int32"),
    #     ms.arg_info.TensorInfo(shape=[M, N], dtype="float16"),
    # ]
    args_info = ms.arg_info.ArgInfo.from_prim_func(sch.mod["main"])
    args = [generate_arg(info, dev) for info in args_info]

    dur_us = lib.time_evaluator(lib.entry_name, dev=dev, number=2000, repeat=1)(*args).mean * 1e6
    print(f"[MxNxKxG: {M}_{N}_{K}_{G} {dur_us} us")



def find_best_dyn_m(M, N, K, G, db, top_k, target, dev) -> tvm.tir.schedule.Trace:
    static_m_mod = construct_te(M, N, K, G)
    static_m_mod = ms.tir_integration._normalize_mod(static_m_mod)

    assert db.has_workload(static_m_mod)
    workload = db.commit_workload(static_m_mod)

    best_score_us = float("inf")
    best_reported_static_score_us = None
    best_topk_pos = None
    best_trace = None
    
    top_k_recs = db.get_top_k(workload, top_k=top_k)
    
    for top_idx, rec in enumerate(top_k_recs):
        dyn_m_mod = construct_te(N, K, G)
        dyn_m_trace = mutate_to_dyn_m(rec.trace) 
        dyn_m_sch = tvm.tir.Schedule(dyn_m_mod)
        dyn_m_trace.apply_to_schedule(dyn_m_sch, remove_postproc=False)
        
        with target:
            dyn_m_lib = tvm.build(dyn_m_sch.mod["main"])

        M_ = 1
        args_info = [
            ms.arg_info.TensorInfo(shape=[M_, K], dtype="float16"),
            ms.arg_info.TensorInfo(shape=[K//8, N], dtype="int32"),
            ms.arg_info.TensorInfo(shape=[G, N], dtype="float16"),
            ms.arg_info.TensorInfo(shape=[G, N//8], dtype="int32"),
            ms.arg_info.TensorInfo(shape=[M_, N], dtype="float16"),
        ]
        args = [generate_arg(info, dev) for info in args_info]

        score_us = dyn_m_lib.time_evaluator(dyn_m_lib.entry_name, dev=dev, number=2000, repeat=1)(*args).mean * 1e6
        print(f"[TOP:{top_idx}] [M:{M}] Dyn_M Duration {score_us} us  (declared {float(rec.run_secs[0]) * 1e6} us)")
        
        if score_us < best_score_us:
            best_score_us = score_us
            best_reported_static_score_us = float(rec.run_secs[0]) * 1e6
            best_trace = dyn_m_trace
            best_topk_pos = top_idx

    print(f"Best trace found in position {best_topk_pos}")
    print(f"      dyn_m_score  : {best_score_us} us")
    print(f"     static_score  : {best_reported_static_score_us} us")
    
    return best_trace


if __name__ == "__main__":
    main()
