from tvm.tir.schedule import Schedule, BlockRV
from tvm import meta_schedule as ms
from tvm import IRModule

from tune_com import matmul_g128_KN_sym_dynm, TARGET

"""
MDS 2

RFactor, final redaction outside of thread block.

Additional rf loop is spent to "blockIdx.y"

DOESN'T WORK!!
DynSharedMemLinearAccessPatternFinder is not bale to analize shared memory if they used 
in separate kernels (several loops with BlockIdx.x binding). 
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

    # Split reduction into two stage, main matmul and final reduction 
    b_mm = sch.rfactor(lk_rf, factor_axis=0)  # TODO: Is "factor_axis=0" ok?
    b_rf = sch.get_consumers(b_mm)[0]  # same as "block" 

    # ========== Stage 1 ==========
    # Processing Main Matmul block
    _, lm_4, lm_3, lm_2, lm_1, ln_4, ln_3, ln_2, ln_1, lk_rf, lk_2, lk_1, lm_wb, ln_wb, lk_wb = sch.get_loops(b_mm)
    
    #          |  B.x     |       B.y        |    Th.y   |   Software loops      |
    sch.reorder(lm_4, ln_4, lm_3, ln_3, lk_rf, lm_2, ln_2, lk_2, lk_1, lm_1, ln_1)
    lnm_by = sch.fuse(lm_4, ln_4)
    lnm_bx = sch.fuse(lm_3, ln_3, lk_rf)
    lnm_ty = sch.fuse(lm_2, ln_2)
    sch.bind(lnm_by, thread_axis="blockIdx.y")
    sch.bind(lnm_bx, thread_axis="blockIdx.x")
    sch.bind(lnm_ty, thread_axis="threadIdx.y")

    b_wmma = sch.blockize(lm_wb)

    # copy from/to shared on level of L1 block
    b_o_shared = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="shared.dyn")
    b_o_wmma = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="wmma.accumulator")
    sch.reverse_compute_at(b_o_wmma, loop=lnm_ty, preserve_unit_loops=True, index=-1)
    sch.reverse_compute_at(b_o_shared, loop=lnm_ty, preserve_unit_loops=True, index=-1)
    
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
        lmn, lm_ty_0, ln_ty_2, lmn_tx, lmn_v = sch.split(ls_fused, factors=[None, lm_factors[-2], ln_factors[-2], tx_size, vec_size])
        sch.bind(lmn_tx, thread_axis="threadIdx.x")
        sch.bind(sch.fuse(lm_ty_0, ln_ty_2), thread_axis="threadIdx.y")
        sch.vectorize(lmn_v)

        # NB! significant impact. Looks like bank conflict. "buffer_index=0" for cache write, is it correct? 
        sch.storage_align(block=blk, buffer_index=0, axis=-2, factor=16, offset=8)   

    # blk_vectorize(b_o_shared, vec_size=4)
    blk_vectorize(b_a_shared, vec_size=4, num_loops_to_fuse=3)
    blk_vectorize(b_b_shared, vec_size=4, num_loops_to_fuse=2)
    blk_vectorize(b_o_shared, vec_size=4, num_loops_to_fuse=4)
    
    # ========== Stage 2 ==========
    lb, lm_4, lm_3, lm_2, lm_1, ln_4, ln_3, ln_2, ln_1, lk_rf, lm_wb, ln_wb = sch.get_loops(b_rf)
    sch.reorder(lm_4, ln_4, lm_3, ln_3, lm_2, ln_2, lk_rf, lm_1, ln_1, lm_wb, ln_wb) 
    lnm_2 = sch.fuse(lm_2, ln_2)
    sch.bind(lnm_2, thread_axis="threadIdx.x")
    lnm_o = sch.fuse(lm_4, ln_4, lm_3, ln_3)
    sch.bind(loop=lnm_o, thread_axis="blockIdx.x")

    b_rf_o_shared = sch.cache_write(b_rf, write_buffer_index=0, storage_scope="shared.dyn")
    sch.reverse_compute_at(b_rf_o_shared, loop=lnm_2, preserve_unit_loops=True, index=-1)
    b_rf_init = sch.decompose_reduction(b_rf, lk_rf)

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


def mds2(m_pad = 32, decisions = {}):
    if not decisions:
        decisions = {
            "m_split": [1, 2, 2], 
            "n_split": [1, 2, 2, 2], 
            "k_split": [1, 2, 2]
        }

    def gen_ds(sch: Schedule):
        mm_block = sch.get_block("matmul")

        # normalize_mm_block(sch, mm_block)

        _ds_rule1(sch, mm_block, m_pad, decisions)

        dec_block = sch.get_block("decode")
        sch.compute_inline(dec_block)

    return gen_ds


def main():
    M = 64
    
    func = matmul_g128_KN_sym_dynm.with_attr({"metaschedule.hint.dyn_var_value": {"m": M}})
    mod = IRModule({"matmul_g128_KN_sym_dynm": func})

    ms.tir_integration.tune_tir(
        mod=mod,
        target=TARGET,
        work_dir=f"___tmp/tune_3_mds2_m{M}",
        max_trials_global=100500,
        max_trials_per_task=4096,
        num_trials_per_iter=32,
        cost_model="random",
        # cost_model="xgb",
        space=ms.space_generator.ScheduleFn(
            mds2(m_pad=M),
            sch_rules=[],
            postprocs=[],
            mutator_probs={},
        )
    )


if __name__ == "__main__":
    main()
