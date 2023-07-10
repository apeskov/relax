import tvm
from tvm.script import tir as T
import tvm.meta_schedule as ms
from tvm import te

import tvm.tir.tensor_intrin.cuda

def apply_trace(sch: tvm.tir.Schedule) -> None:
  """ From TVM """
  b0 = sch.get_block(name="decode_zp", func_name="main")
  b1 = sch.get_block(name="decode_wgh", func_name="main")
  b2 = sch.get_block(name="matmul", func_name="main")
  b3 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b2, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  b4 = sch.reindex(block=b2, buffer=("write", 0))
  b5 = sch.reindex(block=b2, buffer=("read", 0))
  b6 = sch.reindex(block=b2, buffer=("read", 1))
  sch.transform_layout(block=b2, buffer=("read", 0), index_map=lambda v_vm, v_k: (v_vm, v_k,), pad_value=None, assume_injective_transform=True)
  sch.transform_layout(block=b2, buffer=("read", 1), index_map=lambda v_vn, v_k: (v_k, v_vn,), pad_value=None, assume_injective_transform=True)
  sch.transform_layout(block=b2, buffer=("write", 0), index_map=lambda v_vm, v_vn: (v_vm, v_vn,), pad_value=None, assume_injective_transform=True)
  sch.transform_block_layout(block=b4, index_map=lambda v_vm, v_vn: (v_vm, v_vn,))
  sch.transform_block_layout(block=b5, index_map=lambda v_vm, v_k: (v_vm, v_k,))
  sch.transform_block_layout(block=b6, index_map=lambda v_vn, v_k: (v_k, v_vn,))
  sch.transform_block_layout(block=b2, index_map=lambda v_vm, v_vn, v_k: (v_vm, v_vn, v_k,))
  sch.pad_einsum(block=b2, padding=[16, 16, 16])  # [XXX]
  sch.compute_inline(b5)  # [XXX]
  sch.reverse_compute_inline(b4)  # [XXX]
  b4 = sch.get_block("matmul_reindex_pad")
  b5 = sch.get_block("A_reindex_pad")
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
  v26, v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l23, n=5, max_innermost_factor=4, decision=[1, 1, 1, 1, 1])
  l31, l32, l33, l34, l35 = sch.split(loop=l23, factors=[v26, v27, v28, v29, v30], preserve_unit_iters=True)
  v36, v37, v38, v39, v40 = sch.sample_perfect_tile(loop=l24, n=5, max_innermost_factor=4, decision=[20, 8, 2, 1, 1])
  l41, l42, l43, l44, l45 = sch.split(loop=l24, factors=[v36, v37, v38, v39, v40], preserve_unit_iters=True)
  v46, v47, v48 = sch.sample_perfect_tile(loop=l25, n=3, max_innermost_factor=4, decision=[20, 4, 4])
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
  sch.transform_layout(block=b22, buffer=("write", 0), index_map=lambda i0, i1: (i0 // 16 // (v29 * v30), i1 // 16 // (v39 * v40), i0 // 16 % (v29 * v30), i1 // 16 % (v39 * v40), i0 % 16, i1 % 16,), pad_value=None, assume_injective_transform=True)
  b55 = sch.cache_write(block=b22, write_buffer_index=0, storage_scope="shared.dyn")
  sch.reverse_compute_at(block=b55, loop=l53, preserve_unit_loops=True, index=-1)
  b56 = sch.cache_write(block=b22, write_buffer_index=0, storage_scope="wmma.accumulator")
  l57, l58, l59, l60, l61, l62, l63, l64 = sch.get_loops(block=b55)
  sch.reorder(l61, l59, l60, l62)
  sch.compute_at(block=b56, loop=l61, preserve_unit_loops=True, index=-1)
  l65, l66, l67, l68, l69, l70, l71, l72, l73 = sch.get_loops(block=b56)
  l74 = sch.fuse(l68, l69, preserve_unit_iters=True)
  sch.bind(loop=l74, thread_axis="threadIdx.y")
  sch.reverse_compute_inline(block=b4)
  l75, l76, l77, l78, l79, l80, l81, l82 = sch.get_loops(block=b56)
  b83 = sch.blockize(target=l81, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b83, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")
  l84, l85, l86, l87, l88, l89, l90, l91 = sch.get_loops(block=b55)
  l92 = sch.fuse(l87, l88, l89, l90, l91, preserve_unit_iters=True)
  v93 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b55, ann_key="meta_schedule.cooperative_fetch", ann_val=v93)
  b94 = sch.cache_read(block=b22, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b22])
  print("before sch.compute_at(block=b94,...)")
  sch.compute_at(block=b94, loop=l49, preserve_unit_loops=True, index=-1)
  l95, l96, l97, l98, l99, l100 = sch.get_loops(block=b94)
  l101 = sch.fuse(l99, l100, preserve_unit_iters=True)
  v102 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
  sch.annotate(block_or_loop=b94, ann_key="meta_schedule.cooperative_fetch", ann_val=v102)
  b103 = sch.cache_read(block=b22, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b22])
  sch.compute_at(block=b103, loop=l49, preserve_unit_loops=True, index=-1)
  l104, l105, l106, l107, l108, l109 = sch.get_loops(block=b103)
  l110 = sch.fuse(l108, l109, preserve_unit_iters=True)
  v111 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b103, ann_key="meta_schedule.cooperative_fetch", ann_val=v111)
  b112 = sch.cache_read(block=b22, read_buffer_index=0, storage_scope="wmma.matrix_a")
  sch.compute_at(block=b112, loop=l50, preserve_unit_loops=True, index=-1)
  l113, l114, l115, l116, l117, l118, l119 = sch.get_loops(block=b112)
  l120, l121 = sch.split(loop=l119, factors=[None, 16], preserve_unit_iters=True)
  l122, l123 = sch.split(loop=l118, factors=[None, 16], preserve_unit_iters=True)
  l124, l125, l126, l127, l128, l129, l130, l131, l132 = sch.get_loops(block=b112)
  sch.reorder(l131, l123, l121)
  b133 = sch.blockize(target=l123, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b133, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")
  b134 = sch.cache_read(block=b22, read_buffer_index=1, storage_scope="wmma.matrix_b")
  sch.compute_at(block=b134, loop=l50, preserve_unit_loops=True, index=-1)
  l135, l136, l137, l138, l139, l140, l141 = sch.get_loops(block=b134)
  l142, l143 = sch.split(loop=l141, factors=[None, 16], preserve_unit_iters=True)
  l144, l145 = sch.split(loop=l140, factors=[None, 16], preserve_unit_iters=True)
  l146, l147, l148, l149, l150, l151, l152, l153, l154 = sch.get_loops(block=b134)
  sch.reorder(l153, l145, l143)
  b155 = sch.blockize(target=l145, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b155, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")
  sch.compute_inline(block=b5)
  sch.compute_inline(block=b6)
  sch.storage_align(block=b94, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.storage_align(block=b103, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.compute_inline(block=b1)
  sch.compute_inline(block=b0)
  v156 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=4)
  sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v156)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b55, ann_key="meta_schedule.cooperative_fetch")
  l157, l158, l159, l160 = sch.get_loops(block=b55)
  l161, l162, l163, l164 = sch.split(loop=l160, factors=[None, 2, 32, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l164)
  sch.bind(loop=l163, thread_axis="threadIdx.x")
  sch.bind(loop=l162, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b94, ann_key="meta_schedule.cooperative_fetch")
  l165, l166, l167, l168, l169 = sch.get_loops(block=b94)
  l170, l171, l172, l173 = sch.split(loop=l169, factors=[None, 2, 32, 8], preserve_unit_iters=True)
  sch.vectorize(loop=l173)
  sch.bind(loop=l172, thread_axis="threadIdx.x")
  sch.bind(loop=l171, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b103, ann_key="meta_schedule.cooperative_fetch")
  l174, l175, l176, l177, l178 = sch.get_loops(block=b103)
  l179, l180, l181, l182 = sch.split(loop=l178, factors=[None, 2, 32, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l182)
  sch.bind(loop=l181, thread_axis="threadIdx.x")
  sch.bind(loop=l180, thread_axis="threadIdx.y")
  b183 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b183, ann_key="meta_schedule.unroll_explicit")
  b184, b185, b186, b187, b188, b189, b190 = sch.get_child_blocks(b183)
  l191, l192, l193, l194, l195, l196, l197, l198 = sch.get_loops(block=b184)
  l199, l200, l201, l202, l203, l204, l205, l206 = sch.get_loops(block=b185)
  l207, l208, l209, l210, l211, l212, l213 = sch.get_loops(block=b186)
  l214, l215, l216, l217, l218, l219, l220 = sch.get_loops(block=b187)
  l221, l222, l223, l224, l225, l226, l227, l228, l229, l230 = sch.get_loops(block=b188)
  sch.annotate(block_or_loop=l221, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
  sch.annotate(block_or_loop=l221, ann_key="pragma_unroll_explicit", ann_val=1)
  l231, l232, l233, l234, l235, l236 = sch.get_loops(block=b189)
  l237, l238, l239, l240, l241, l242, l243 = sch.get_loops(block=b190)
  b244 = sch.get_block(name="matmul_o", func_name="main")
  l245, l246, l247, l248, l249, l250, l251, l252, l253, l254 = sch.get_loops(block=b244)
  b255 = sch.decompose_reduction(block=b244, loop=l248)
  sch.unannotate(block_or_loop=b255, ann_key="meta_schedule.auto_tensorize")
  sch.annotate(block_or_loop=b255, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
  sch.unannotate(block_or_loop=b244, ann_key="meta_schedule.auto_tensorize_init")
  sch.unannotate(block_or_loop=b255, ann_key="meta_schedule.auto_tensorize_init")
  b256 = sch.get_block(name="matmul_o_init", func_name="main")
  sch.unannotate(block_or_loop=b256, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b256, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
  b257 = sch.get_block(name="A_reindex_shared.dyn_wmma.matrix_a_o", func_name="main")
  sch.unannotate(block_or_loop=b257, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b257, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
  b258 = sch.get_block(name="decode_wgh_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
  sch.unannotate(block_or_loop=b258, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b258, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
  b259 = sch.get_block(name="matmul_o_update", func_name="main")
  sch.unannotate(block_or_loop=b259, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b259, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
  b260 = sch.get_block(name="matmul_reindex_shared.dyn_wmma.accumulator_o", func_name="main")
  sch.unannotate(block_or_loop=b260, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b260, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)


def construct_te(B, M, N, K, G):
    # n = te.var("n")
    # B, M, N, K, G = 1, 1, 5120, 5120, 40
    GS = K // G
    
    if B is None:
        a = te.placeholder((M, K), name="A", dtype="float16")
    else:
        a = te.placeholder((B, M, K), name="A", dtype="float16")
    qwgh = te.placeholder((K//8, N), name="qwgh", dtype="int32")
    scl = te.placeholder((G, N), name="scl", dtype="float16")
    qzp = te.placeholder((G, N//8), name="qzp", dtype="int32")
    bias = te.placeholder((N,), name="bias", dtype="float16")

    decoded_zp = te.compute(
        (qzp.shape[0], qzp.shape[1]*8),
        lambda vg, vn: 
            (qzp[vg, vn // 8] >> (vn % 8 * 4) & 0xF).astype("int32") + tvm.tir.const(1, dtype="int32"),
        name="decode_zp"
        )

    decoded_wgh = te.compute(
        (qwgh.shape[0] * 8, qwgh.shape[1]),
        lambda vk, vn: 
            ((qwgh[vk // 8, vn] >> (vk % 8 * 4) & 0xF) - decoded_zp[vk // GS, vn]).astype("float16") * 
            scl[vk // GS, vn],
        name="decode_wgh"
        )

    k = te.reduce_axis((0, decoded_wgh.shape[0]), name="k")
    if B is None:
        x = te.compute(
            (a.shape[0], qwgh.shape[1]),
            lambda vm, vn: te.sum(a[vm, k] * decoded_wgh[k, vn], axis=k),
            name="matmul"
        )
    else:
        x = te.compute(
            (a.shape[0], a.shape[1], qwgh.shape[1]),
            lambda vb, vm, vn: te.sum(a[vb, vm, k] * decoded_wgh[k, vn], axis=k),
            name="matmul"
        )

    # x = x + bias
    
    return tvm.te.create_prim_func([a, qwgh, scl, qzp, bias, x])



def main():
    target = tvm.target.Target("nvidia/nvidia-a10g")
    
    HS = 5120
    B, M, N, K, G = None, 1, HS, HS, HS//128
    
    s_mod = construct_te(B, M, N, K, G)
    s_sch = tvm.tir.Schedule(s_mod)
    apply_trace(s_sch)

    print(s_sch.mod)


if __name__ == "__main__":
    main()
