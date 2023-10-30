
from tvm.tir.schedule import Schedule, BlockRV
from tvm import meta_schedule as ms
from tvm import IRModule

from tune_com import TARGET, matmul_g128_KN_sym_dynm, fused_fused_decode2_NT_matmul

"""
MDS 1

NO RFactor

Design Space with support of dynamic shape.
Trivial application of wmma 16x16 intrinsics.  
"""


def ds_rule1(sch: Schedule, block: BlockRV, m_pad, decisions) -> None: 
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

    m_decision = decisions["m_split"]
    n_decision = decisions["n_split"]
    k_decision = decisions["k_split"]
        
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
    sch.compute_at(b_a_shared, loop=lk_2, preserve_unit_loops=True, index=-1)

    b_b_shared = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="shared.dyn")
    b_b_wmma = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="wmma.matrix_b")
    sch.compute_at(b_b_wmma, loop=lk_1, preserve_unit_loops=True, index=-1)      # NB! Not lk_b2, to reduce wmma::fragments count
    sch.compute_at(b_b_shared, loop=lk_2, preserve_unit_loops=True, index=-1)

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
            "k_split": [1, 2]
        }

    def gen_ds(sch: Schedule):
        mm_block = sch.get_block("matmul")
        ds_rule1(sch, mm_block, m_pad, decisions)

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
        work_dir=f"___tmp/tune_2_mds1_m{M}",
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
    main()
