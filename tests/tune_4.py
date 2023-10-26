from tvm.tir.schedule import Schedule, BlockRV
from tvm import meta_schedule as ms
from tvm import IRModule

from tune_com import matmul_g128_KN_sym_dynm, TARGET

"""
MDS 3 (conceptually the same as MDS2)

RFactor, final reduction is inside thread block.

DOESN'T WORK!!
Issue with arith::Analyzer::Simplify().
Is not able to simplify expression
  constrains {vk_1_i: [0:15], vk_0_2_o:[0:1], vk_0_1_o:[0:1]}
  original expr: (vk_0_1_o * 32 + vk_0_2_o * 16 + vk_1_i) % 64 // 32
  expected expr: vk_0_1_o
"""

def _ds_rule1(sch: Schedule, block: BlockRV, m_pad, decisions) -> None: 
    # Padding. Prolog
    sch.pad_einsum(block, padding=[1, m_pad, 16, 16])
    b_pad_a = sch.get_producers(block)[0]
    b_pad_o = sch.get_consumers(block)[0]

    # ========== Stage 0 ==========
    # block 16x16x16
    ls, lm, ln, lk = sch.get_loops(block)
    lm, lm_b = sch.split(lm, factors=[None, 16])
    ln, ln_b = sch.split(ln, factors=[None, 16])
    lk, lk_b = sch.split(lk, factors=[None, 16])
    sch.reorder(lm, ln, lk, lm_b, ln_b, lk_b)
    
    # Combined loop:
    #
    # lm_4, lm_3, lm_2, lm_1 = split(M)  
    # ln_4, ln_3, ln_2, ln_1 = split(N)
    # k_2, k_1 = split(K)
    #
    # for m_4, n_4 <bind(blk.y)>:                  | block 3
    #   for m_3, n_3 <bind(blk.x)>:                |  
    #     for m_2, n_2 <bind(thr.y): 
    #       for k_rf <bind(thr.z)>:                        |
    #         _fill_zero(acc_r)                            |
    #         for k_2:                                     |
    #           _copy_g2s(a_g[...], a_s)                   |
    #           _copy_g2s(b_g[...], b_s)                   |
    #           for k_1, m_1, n_1:                                       | block 1
    #             _copy_s2r(a_s, a_r)                                    |
    #             _copy_s2r(b_s, b_r)                                    |
    #             wmma::mul<16x16x16>(a_r[...], b_r[...], acc_r[...])    |
    #         _copy_r2s(acc_r, acc_s)                      |
    #
    #       for _ in [0,1] <bind(thr.z)>:            |
    #         for k_rf:    
    #           ... read from acc_s and write to acc_s 
    #           ... with reduction
    #         _copy_s2g(acc_s, c_g)                          |
    #           
    #         
    #
    m_decision, n_decision, k_decision = decisions["m_split"], decisions["n_split"], decisions["k_split"]

    lm_4, lm = sch.split(lm, factors=[None, m_pad//16])    
    lm_factors = sch.sample_perfect_tile(loop=lm, n=3, max_innermost_factor=4, decision=m_decision)
    lm_3, lm_2, lm_1 = sch.split(lm, factors=lm_factors)

    ln_factors = sch.sample_perfect_tile(loop=ln, n=4, max_innermost_factor=4, decision=n_decision)
    ln_4, ln_3, ln_2, ln_1 = sch.split(ln, factors=ln_factors)

    lk_factors = sch.sample_perfect_tile(loop=lk, n=3, max_innermost_factor=4, decision=k_decision)
    lk_rf, lk_2, lk_1 = sch.split(lk, factors=lk_factors)

    # TBD  [B, M, K,] -> [B, M, K // rf, K % rf]
    # Padding block. Change padded data tensor to form with explici RF dimension
    # stride_k = lk_factors[-2] * lk_factors[-1] * 16 
    # sch.transform_layout(block=block, buffer=("read", 0), index_map=lambda b, m, k: (b, k // stride_k, m, k % stride_k), pad_value=None, assume_injective_transform=True)
    # sch.transform_layout(block=block, buffer=("read", 1), index_map=lambda k, n: (k // stride_k, k % stride_k, n), pad_value=None, assume_injective_transform=True)
    
    s_k0 = 16 
    s_k1 = lk_factors[-1] 
    s_k2 = lk_factors[-2] 
    s_k = s_k0 * s_k1 * s_k2
    sch.transform_layout(block=block, buffer=("read", 0), 
                         index_map=lambda b, m, k: (b, k // s_k0 // s_k1 // s_k2, k // s_k0 // s_k1 % s_k2, k // s_k0 % s_k1, m, k % s_k0), 
                         pad_value=None, assume_injective_transform=True)
    sch.transform_layout(block=block, buffer=("read", 1), index_map=lambda k, n: (k // s_k, k % s_k, n), pad_value=None, assume_injective_transform=True)

    # TBD end

    # Split reduction into two stage, main matmul and final reduction 
    b_mm = sch.rfactor(lk_rf, factor_axis=0)  # TODO: Is "factor_axis=0" ok?
    b_rf = sch.get_consumers(b_mm)[0]  # same as "block" 

    # ========== Stage 1 ==========
    # Processing Main Matmul block
    _, lm_4, lm_3, lm_2, lm_1, ln_4, ln_3, ln_2, ln_1, lk_rf, lk_2, lk_1, lm_wb, ln_wb, lk_wb = sch.get_loops(b_mm)
    
    #          |  B.x     |    B.y    |       Th.y       |   Software loops      |
    sch.reorder(lm_4, ln_4, lm_3, ln_3, lm_2, ln_2, lk_rf, lk_2, lk_1, lm_1, ln_1)
    lnm_by = sch.fuse(lm_4, ln_4)
    lnm_bx = sch.fuse(lm_3, ln_3)
    lnm_ty = sch.fuse(lm_2, ln_2, lk_rf)
    sch.bind(lnm_by, thread_axis="blockIdx.y")
    sch.bind(lnm_bx, thread_axis="blockIdx.x")
    sch.bind(lnm_ty, thread_axis="threadIdx.y")

    b_wmma = sch.blockize(lm_wb)

    # copy from/to shared on level of L1 block
    b_o_wmma = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="wmma.accumulator")
    sch.reverse_compute_at(b_o_wmma, loop=lnm_ty, preserve_unit_loops=True, index=-1)
    
    b_a_shared = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="shared.dyn")
    b_a_wmma = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="wmma.matrix_a")
    sch.compute_at(b_a_wmma, loop=lk_1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
    sch.compute_at(b_a_shared, loop=lk_2, preserve_unit_loops=True, index=-1)

    b_b_shared = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="shared.dyn")
    b_b_wmma = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="wmma.matrix_b")
    sch.compute_at(b_b_wmma, loop=lk_1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
    sch.compute_at(b_b_shared, loop=lk_2, preserve_unit_loops=True, index=-1)

    b_wmma_init = sch.decompose_reduction(block=b_wmma, loop=lk_2)

    def blk_tensorize(blk, intrin_name):
        *_, lm, ln = sch.get_loops(blk)
        lm, lm_b = sch.split(lm, factors=[None, 16])
        ln, ln_b = sch.split(ln, factors=[None, 16])
        sch.reorder(lm, ln, lm_b, ln_b)
        blk_16x16 = sch.blockize(lm_b)
        # TODO: add bind to Ty???
        sch.tensorize(blk_16x16, intrin_name)
    
    # vectorize load/store
    def blk_vectorize(blk, num_loops_to_fuse=3, vec_size=4):
        # 16x16 4*32*Ty
        # Ideally it should be 8 (128bit register containd 8 half floats) 
        ty_size = (lm_factors[-2] * ln_factors[-2])  # TODO: error "Stringifying is not supported for type: tir.Mul"
        tx_size = 32
        ls = sch.get_loops(blk) 
        ls_fused = sch.fuse(*ls[-num_loops_to_fuse:])
        # lmn, lmn_ty, lmn_tx, lmn_v = sch.split(lmn, factors=[None, ty_size, tx_size, vec_size])
        lmn, lm_ty_0, ln_ty_2, ln_ty_3, lmn_tx, lmn_v = sch.split(ls_fused, factors=[None, lm_factors[-2], ln_factors[-2], lk_factors[0], tx_size, vec_size])
        sch.bind(lmn_tx, thread_axis="threadIdx.x")
        sch.bind(sch.fuse(lm_ty_0, ln_ty_2, ln_ty_3), thread_axis="threadIdx.y")
        sch.vectorize(lmn_v)

        # NB! significant impact. Looks like bank conflict. "buffer_index=0" for cache write, is it correct? 
        sch.storage_align(block=blk, buffer_index=0, axis=-2, factor=16, offset=8)   

    # blk_vectorize(b_o_shared, vec_size=4)
    blk_vectorize(b_a_shared, vec_size=4, num_loops_to_fuse=3)
    blk_vectorize(b_b_shared, vec_size=4, num_loops_to_fuse=3)
    
    # ========== Stage 2 ==========
    # TODO: Try sch.merge here
    sch.reverse_compute_at(b_rf, loop=lnm_bx, preserve_unit_loops=True)
    
    # TODO: how to prevent loop order after reverse_compute_at?
    #       using of sch.blockize(lk_rf) leads to error in reverse_compute_at.
    #       May be sch.merge will help.
    rf_loops = sch.get_loops(b_rf)
    # sch.decompose_reduction(b_rf, ax1)
    
    rf_in_shared = sch.cache_read(b_rf, read_buffer_index=0, storage_scope="shared.dyn")
    sch.reverse_compute_inline(rf_in_shared)

    rf_out_shared = sch.cache_write(b_rf, write_buffer_index=0, storage_scope="shared.dyn")
    sch.reverse_compute_at(rf_out_shared, loop=lnm_bx)
    blk_vectorize(rf_out_shared, vec_size=4, num_loops_to_fuse=2)

    l_rf, l_b, l_m, l_n = rf_loops[-4:]
    # b_rf_init = sch.decompose_reduction(block=b_rf, loop=l_rf)
    # blk_vectorize(b_rf_init, vec_size=4, num_loops_to_fuse=3)
    
    # Manual vectorization. Have
    l_bmn_fused = sch.fuse(l_b, l_m, l_n)
    l_bmn, l_bmn_ty_0, l_bmn_ty_2, l_bmn_ty_3, l_bmn_tx, l_bmn_v = sch.split(l_bmn_fused, factors=[None, lm_factors[-2], ln_factors[-2], lk_factors[0], 32, 4])
    l_bmn_ty = sch.fuse(l_bmn_ty_0, l_bmn_ty_2, l_bmn_ty_3)
    sch.reorder(l_bmn, l_bmn_ty, l_bmn_tx, l_rf, l_bmn_v)
    sch.bind(l_bmn_tx, thread_axis="threadIdx.x")
    sch.bind(l_bmn_ty, thread_axis="threadIdx.y")
    sch.vectorize(l_bmn_v)
    b_rf_init = sch.decompose_reduction(block=b_rf, loop=l_rf)

    # ========== Stage 3 ==========
    # Postprocs: Final tensorization and bind to HW threads.
    # Do it after merging with RF block. Otherwise it will not be merged.
    # TODO: formulate it as postprocs.
    sch.tensorize(b_wmma, "wmma_sync_16x16x16_f16f16f16")
    sch.tensorize(b_wmma_init, "wmma_fill_16x16x16_f16")
    blk_tensorize(b_o_wmma, "wmma_store_16x16x16_f16_shared_dyn")
    blk_tensorize(b_a_wmma, "wmma_load_16x16x16_f16_a_shared_dyn")
    blk_tensorize(b_b_wmma, "wmma_load_16x16x16_f16_b_shared_dyn")   # TODO: It accepts "wmma_load_16x16x16_f16_b_trans_shared_dyn" as well.. problem

    # Padding. Epilog
    sch.compute_inline(b_pad_a)
    sch.reverse_compute_inline(b_pad_o)


def mds3(m_pad = 32, decisions = {}):
    if not decisions:
        decisions = {
            "m_split": [1, 2, 2], 
            "n_split": [1, 2, 2, 2], 
            "k_split": [1, 2, 2]
        }

    def gen_ds(sch: Schedule):
        mm_block = sch.get_block("matmul")

        _ds_rule1(sch, mm_block, m_pad, decisions)

        dec_block = sch.get_block("decode")
        sch.compute_inline(dec_block)

    return gen_ds


def main():
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
    mod = IRModule({"matmul_g128_KN_sym_dynm": func})

    ms.tir_integration.tune_tir(
        mod=mod,
        target=TARGET,
        work_dir=f"___tmp/tune_4_mds3_m{M}",
        max_trials_global=100500,
        max_trials_per_task=4096,
        num_trials_per_iter=32,
        cost_model="random",
        # cost_model="xgb",
        space=ms.space_generator.ScheduleFn(
            mds3(m_pad=M),
            sch_rules=[],
            postprocs=[],
            mutator_probs={},
        )
    )


if __name__ == "__main__":
    main()
