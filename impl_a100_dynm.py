from tvm.script import ir as I
from tvm.script import tir as T


# [TOP:0] [M:32] Dyn_M Duration 181.8956755 us  (declared 152.35255123339658 us)
# [TOP:1] [M:32] Dyn_M Duration 143.28627 us  (declared 152.35449146110057 us)
# [TOP:2] [M:32] Dyn_M Duration 143.290878 us  (declared 152.39283634633682 us)
# [TOP:3] [M:32] Dyn_M Duration 143.272964 us  (declared 152.4064784688995 us)
# [TOP:4] [M:32] Dyn_M Duration 143.3128965 us  (declared 152.41234832535883 us)
# [TOP:5] [M:32] Dyn_M Duration 147.9316405 us  (declared 152.45264722753348 us)
# [TOP:6] [M:32] Dyn_M Duration 147.9813075 us  (declared 152.46286952380953 us)
# [TOP:7] [M:32] Dyn_M Duration 147.938812 us  (declared 152.4737031398668 us)
# [TOP:8] [M:32] Dyn_M Duration 147.977722 us  (declared 152.48585358851673 us)
# [TOP:9] [M:32] Dyn_M Duration 147.9608305 us  (declared 152.49124167459564 us)
# [TOP:10] [M:32] Dyn_M Duration 147.9685055 us  (declared 152.50195623215987 us)
# [TOP:11] [M:32] Dyn_M Duration 147.935745 us  (declared 152.50312713472485 us)
# [TOP:12] [M:32] Dyn_M Duration 162.22105399999998 us  (declared 153.36762884615385 us)
# [TOP:13] [M:32] Dyn_M Duration 162.19648700000002 us  (declared 153.40989377990434 us)
# [TOP:14] [M:32] Dyn_M Duration 143.58476250000004 us  (declared 153.4233808574277 us)
# [TOP:15] [M:32] Dyn_M Duration 165.06163 us  (declared 153.49417416267943 us)
# [TOP:16] [M:32] Dyn_M Duration 162.2492215 us  (declared 153.55955780346824 us)
# [TOP:17] [M:32] Dyn_M Duration 162.2205505 us  (declared 153.59114409221903 us)
# [TOP:18] [M:32] Dyn_M Duration 162.2947845 us  (declared 153.7167458937198 us)
# [TOP:19] [M:32] Dyn_M Duration 143.45677149999997 us  (declared 153.75654722492698 us)
# [TOP:20] [M:32] Dyn_M Duration 162.222076 us  (declared 153.76836249999997 us)
# [TOP:21] [M:32] Dyn_M Duration 162.1729275 us  (declared 153.82098554913293 us)
# [TOP:22] [M:32] Dyn_M Duration 164.8814085 us  (declared 153.82106001936106 us)
# [TOP:23] [M:32] Dyn_M Duration 147.942398 us  (declared 153.89087246376812 us)
# [TOP:24] [M:32] Dyn_M Duration 143.2919005 us  (declared 153.8939865125241 us)
# [TOP:25] [M:32] Dyn_M Duration 147.9766995 us  (declared 153.90286512524085 us)
# [TOP:26] [M:32] Dyn_M Duration 147.98439 us  (declared 153.9064187380497 us)
# [TOP:27] [M:32] Dyn_M Duration 147.9659575 us  (declared 153.9137138728324 us)
# [TOP:28] [M:32] Dyn_M Duration 143.2801205 us  (declared 153.95190513068732 us)
# [TOP:29] [M:32] Dyn_M Duration 147.9562225 us  (declared 153.96212587412586 us)
# [TOP:30] [M:32] Dyn_M Duration 143.45318599999996 us  (declared 153.97084053794427 us)
# [TOP:31] [M:32] Dyn_M Duration 144.5841975 us  (declared 153.99460982658957 us)
# [TOP:32] [M:32] Dyn_M Duration 144.57650749999996 us  (declared 154.01630732177267 us)
# [TOP:33] [M:32] Dyn_M Duration 148.158981 us  (declared 154.07291976516635 us)
# [TOP:34] [M:32] Dyn_M Duration 144.59750349999996 us  (declared 154.0801844660194 us)
# [TOP:35] [M:32] Dyn_M Duration 148.1661375 us  (declared 154.08379710144928 us)
# [TOP:36] [M:32] Dyn_M Duration 148.16921950000003 us  (declared 154.08516213592233 us)
# [TOP:37] [M:32] Dyn_M Duration 148.162567 us  (declared 154.08776231884056 us)
# [TOP:38] [M:32] Dyn_M Duration 148.18457 us  (declared 154.09226300578032 us)
# [TOP:39] [M:32] Dyn_M Duration 157.26387 us  (declared 154.09226300578032 us)
# Best trace found in position 3
#       dyn_m_score  : 143.272964 us
#      static_score  : 152.4064784688995 us
# [TOP:0] [M:32] Dyn_M Duration 142.7240905 us  (declared 75.53586415633939 us)
# [TOP:1] [M:32] Dyn_M Duration 122.56614649999999 us  (declared 75.53814559386973 us)
# [TOP:2] [M:32] Dyn_M Duration 122.50624049999999 us  (declared 75.53928217349856 us)
# [TOP:3] [M:32] Dyn_M Duration 122.58969849999998 us  (declared 75.5412316491897 us)
# [TOP:4] [M:32] Dyn_M Duration 122.532867 us  (declared 75.54546482290151 us)
# [TOP:5] [M:32] Dyn_M Duration 122.5487365 us  (declared 75.54574335863377 us)
# [TOP:6] [M:32] Dyn_M Duration 123.1001585 us  (declared 75.64996264367817 us)
# [TOP:7] [M:32] Dyn_M Duration 123.0888975 us  (declared 75.65055958055291 us)
# [TOP:8] [M:32] Dyn_M Duration 123.083267 us  (declared 75.65290038314176 us)
# [TOP:9] [M:32] Dyn_M Duration 123.1155165 us  (declared 75.65928017241379 us)
# [TOP:10] [M:32] Dyn_M Duration 123.070976 us  (declared 75.65928017241379 us)
# [TOP:11] [M:32] Dyn_M Duration 123.12217700000001 us  (declared 75.6607490421456 us)
# [TOP:12] [M:32] Dyn_M Duration 123.09401700000001 us  (declared 75.66276358436606 us)
# [TOP:13] [M:32] Dyn_M Duration 123.08992 us  (declared 75.66325071496664 us)
# [TOP:14] [M:32] Dyn_M Duration 123.0684125 us  (declared 75.66369444444443 us)
# [TOP:15] [M:32] Dyn_M Duration 123.08531149999999 us  (declared 75.66565277777777 us)
# [TOP:16] [M:32] Dyn_M Duration 123.04946850000002 us  (declared 75.66663936781609 us)
# [TOP:17] [M:32] Dyn_M Duration 123.081726 us  (declared 75.67106196377503 us)
# [TOP:18] [M:32] Dyn_M Duration 123.0807035 us  (declared 75.67252383222117 us)
# [TOP:19] [M:32] Dyn_M Duration 123.1083525 us  (declared 75.67497030651342 us)
# [TOP:20] [M:32] Dyn_M Duration 123.11756850000002 us  (declared 75.67502837902838 us)
# [TOP:21] [M:32] Dyn_M Duration 123.11347150000002 us  (declared 75.67545977011494 us)
# [TOP:22] [M:32] Dyn_M Duration 82.34649650000001 us  (declared 75.73922078544062 us)
# [TOP:23] [M:32] Dyn_M Duration 82.4581145 us  (declared 75.74103222703224 us)
# [TOP:24] [M:32] Dyn_M Duration 82.34649650000001 us  (declared 75.74264846743296 us)
# [TOP:25] [M:32] Dyn_M Duration 82.4775695 us  (declared 75.74281029551955 us)
# [TOP:26] [M:32] Dyn_M Duration 82.3347165 us  (declared 75.74363457854405 us)
# [TOP:27] [M:32] Dyn_M Duration 82.4780805 us  (declared 75.74398268398268 us)
# [TOP:28] [M:32] Dyn_M Duration 82.3434215 us  (declared 75.7455933908046 us)
# [TOP:29] [M:32] Dyn_M Duration 82.477058 us  (declared 75.74706226053638 us)
# [TOP:30] [M:32] Dyn_M Duration 82.3403545 us  (declared 75.74915252621544 us)
# [TOP:31] [M:32] Dyn_M Duration 122.57587399999998 us  (declared 77.40146598139991 us)
# [TOP:32] [M:32] Dyn_M Duration 122.526718 us  (declared 77.40247430249633 us)
# [TOP:33] [M:32] Dyn_M Duration 122.60761249999999 us  (declared 77.40547674987762 us)
# [TOP:34] [M:32] Dyn_M Duration 122.49549099999999 us  (declared 77.40652118850464 us)
# [TOP:35] [M:32] Dyn_M Duration 122.55538900000002 us  (declared 77.40735675675674 us)
# [TOP:36] [M:32] Dyn_M Duration 122.55897499999999 us  (declared 77.41750856583457 us)
# [TOP:37] [M:32] Dyn_M Duration 123.086334 us  (declared 77.54030347528145 us)
# [TOP:38] [M:32] Dyn_M Duration 123.0658565 us  (declared 77.5423127753304 us)
# [TOP:39] [M:32] Dyn_M Duration 123.086845 us  (declared 77.54623783783784 us)
# Best trace found in position 26
#       dyn_m_score  : 82.3347165 us
#      static_score  : 75.74363457854405 us
# [TOP:0] [M:32] Dyn_M Duration 154.8441615 us  (declared 141.39203108348133 us)
# [TOP:1] [M:32] Dyn_M Duration 149.23263499999996 us  (declared 141.44587367491167 us)
# [TOP:2] [M:32] Dyn_M Duration 150.71948199999997 us  (declared 141.4630600706714 us)
# [TOP:3] [M:32] Dyn_M Duration 150.5361935 us  (declared 141.53693179805137 us)
# [TOP:4] [M:32] Dyn_M Duration 150.4394225 us  (declared 141.57945169946333 us)
# [TOP:5] [M:32] Dyn_M Duration 150.55513 us  (declared 141.7237812223206 us)
# [TOP:6] [M:32] Dyn_M Duration 144.442886 us  (declared 142.40482258064515 us)
# [TOP:7] [M:32] Dyn_M Duration 145.2328945 us  (declared 142.50605861456484 us)
# [TOP:8] [M:32] Dyn_M Duration 142.621185 us  (declared 142.82014183764497 us)
# [TOP:9] [M:32] Dyn_M Duration 155.74117999999999 us  (declared 143.7047274368231 us)
# [TOP:10] [M:32] Dyn_M Duration 157.4799345 us  (declared 143.95123619909504 us)
# [TOP:11] [M:32] Dyn_M Duration 161.8872375 us  (declared 144.02911462093863 us)
# [TOP:12] [M:32] Dyn_M Duration 145.738754 us  (declared 144.08374570135743 us)
# [TOP:13] [M:32] Dyn_M Duration 145.71929899999998 us  (declared 144.09274167416743 us)
# [TOP:14] [M:32] Dyn_M Duration 145.733627 us  (declared 144.0960242587601 us)
# [TOP:15] [M:32] Dyn_M Duration 153.1038665 us  (declared 144.14833393501806 us)
# [TOP:16] [M:32] Dyn_M Duration 165.3734435 us  (declared 144.1889478885894 us)
# [TOP:17] [M:32] Dyn_M Duration 165.38316300000002 us  (declared 144.2353575268817 us)
# [TOP:18] [M:32] Dyn_M Duration 165.404678 us  (declared 144.24036561085975 us)
# [TOP:19] [M:32] Dyn_M Duration 161.9727325 us  (declared 144.4577299729973 us)
# [TOP:20] [M:32] Dyn_M Duration 161.86419650000002 us  (declared 144.46566129032257 us)
# [TOP:21] [M:32] Dyn_M Duration 156.68582150000003 us  (declared 144.5161561371841 us)
# [TOP:22] [M:32] Dyn_M Duration 152.1684415 us  (declared 144.88399187725634 us)
# [TOP:23] [M:32] Dyn_M Duration 150.2494655 us  (declared 145.2093318223028 us)
# [TOP:24] [M:32] Dyn_M Duration 150.336517 us  (declared 145.3495104261106 us)
# [TOP:25] [M:32] Dyn_M Duration 145.2620845 us  (declared 145.51617759562842 us)
# [TOP:26] [M:32] Dyn_M Duration 171.2834625 us  (declared 145.80549727272728 us)
# [TOP:27] [M:32] Dyn_M Duration 171.5563505 us  (declared 145.83240713632205 us)
# [TOP:28] [M:32] Dyn_M Duration 162.05772349999998 us  (declared 145.852200913242 us)
# [TOP:29] [M:32] Dyn_M Duration 156.9843135 us  (declared 145.97386825251604 us)
# [TOP:30] [M:32] Dyn_M Duration 145.952255 us  (declared 146.01958721461187 us)
# [TOP:31] [M:32] Dyn_M Duration 145.91027799999998 us  (declared 146.02631876138432 us)
# [TOP:32] [M:32] Dyn_M Duration 145.9542995 us  (declared 146.0356989935956 us)
# [TOP:33] [M:32] Dyn_M Duration 162.153976 us  (declared 146.05538334858187 us)
# [TOP:34] [M:32] Dyn_M Duration 162.0776975 us  (declared 146.06728675799087 us)
# [TOP:35] [M:32] Dyn_M Duration 171.43705699999998 us  (declared 146.1182048957389 us)
# [TOP:36] [M:32] Dyn_M Duration 153.1048885 us  (declared 146.12945545454548 us)
# [TOP:37] [M:32] Dyn_M Duration 152.973312 us  (declared 146.15085909090908 us)
# [TOP:38] [M:32] Dyn_M Duration 144.3793945 us  (declared 146.17107701019253 us)
# [TOP:39] [M:32] Dyn_M Duration 145.817596 us  (declared 146.22494693504117 us)
# Best trace found in position 8
#       dyn_m_score  : 142.621185 us
#      static_score  : 142.82014183764497 us
# [TOP:0] [M:32] Dyn_M Duration 1094.040039 us  (declared 432.5436397849462 us)
# [TOP:1] [M:32] Dyn_M Duration 1069.0656735 us  (declared 432.6207150537635 us)
# [TOP:2] [M:32] Dyn_M Duration 1067.4094235 us  (declared 432.7280994623656 us)
# [TOP:3] [M:32] Dyn_M Duration 1073.1781005 us  (declared 432.7423918918919 us)
# [TOP:4] [M:32] Dyn_M Duration 1084.662231 us  (declared 432.9097688172043 us)
# [TOP:5] [M:32] Dyn_M Duration 1084.1231685000002 us  (declared 435.3958782608696 us)
# [TOP:6] [M:32] Dyn_M Duration 1085.5086665 us  (declared 435.59298108108106 us)
# [TOP:7] [M:32] Dyn_M Duration 1084.223022 us  (declared 435.86336086956516 us)
# [TOP:8] [M:32] Dyn_M Duration 1084.525024 us  (declared 435.9485353260869 us)
# [TOP:9] [M:32] Dyn_M Duration 1085.4774165 us  (declared 436.7561135371179 us)
# [TOP:10] [M:32] Dyn_M Duration 1085.3605955 us  (declared 437.2658864628821 us)
# [TOP:11] [M:32] Dyn_M Duration 495.852539 us  (declared 443.410110619469 us)
# [TOP:12] [M:32] Dyn_M Duration 495.92678800000004 us  (declared 443.4871504424778 us)
# [TOP:13] [M:32] Dyn_M Duration 495.901184 us  (declared 443.5136464088398 us)
# [TOP:14] [M:32] Dyn_M Duration 495.9078365 us  (declared 443.53910773480663 us)
# [TOP:15] [M:32] Dyn_M Duration 495.89453100000003 us  (declared 443.5687079646018 us)
# [TOP:16] [M:32] Dyn_M Duration 495.9324035 us  (declared 443.57586187845305 us)
# [TOP:17] [M:32] Dyn_M Duration 495.9805295 us  (declared 443.5815110497237 us)
# [TOP:18] [M:32] Dyn_M Duration 1084.915771 us  (declared 444.4508435374149 us)
# [TOP:19] [M:32] Dyn_M Duration 495.9129635 us  (declared 444.9392445054945 us)
# [TOP:20] [M:32] Dyn_M Duration 1084.6018065 us  (declared 448.6669103641456 us)
# [TOP:21] [M:32] Dyn_M Duration 1086.617065 us  (declared 448.72876601671305 us)
# [TOP:22] [M:32] Dyn_M Duration 482.3961485 us  (declared 448.75159052924795 us)
# [TOP:23] [M:32] Dyn_M Duration 495.873016 us  (declared 448.78011142061285 us)
# [TOP:24] [M:32] Dyn_M Duration 482.4284055 us  (declared 448.82006406685235 us)
# [TOP:25] [M:32] Dyn_M Duration 1085.0964355 us  (declared 448.9626657381615 us)
# [TOP:26] [M:32] Dyn_M Duration 496.140808 us  (declared 449.0168579387187 us)
# [TOP:27] [M:32] Dyn_M Duration 496.0056455 us  (declared 449.01970028011203 us)
# [TOP:28] [M:32] Dyn_M Duration 495.9503475 us  (declared 449.0197047353761 us)
# [TOP:29] [M:32] Dyn_M Duration 496.01535000000007 us  (declared 449.051243697479 us)
# [TOP:30] [M:32] Dyn_M Duration 482.4043575 us  (declared 449.0584260089686 us)
# [TOP:31] [M:32] Dyn_M Duration 495.9406125 us  (declared 449.0630134529148 us)
# [TOP:32] [M:32] Dyn_M Duration 496.1576840000001 us  (declared 449.1410852017937 us)
# [TOP:33] [M:32] Dyn_M Duration 482.223602 us  (declared 449.1410852017937 us)
# [TOP:34] [M:32] Dyn_M Duration 496.0081785 us  (declared 449.1927436619718 us)
# [TOP:35] [M:32] Dyn_M Duration 482.2097775 us  (declared 449.23077715877446 us)
# [TOP:36] [M:32] Dyn_M Duration 482.293762 us  (declared 449.2559013452915 us)
# [TOP:37] [M:32] Dyn_M Duration 496.017395 us  (declared 449.2821225626741 us)
# [TOP:38] [M:32] Dyn_M Duration 482.429962 us  (declared 449.3661031390135 us)
# [TOP:39] [M:32] Dyn_M Duration 496.0122985 us  (declared 449.4019387186629 us)
# Best trace found in position 35


@I.ir_module
class ImplModuleA100:
    @T.prim_func
    def matmul_1_15360_5120_40(A: T.Buffer((T.int64(1), T.int64(5120)), "float16"), B_pack: T.Buffer((T.int64(640), T.int64(15360)), "int32"), scales: T.Buffer((T.int64(40), T.int64(15360)), "float16"), zeros_pack: T.Buffer((T.int64(40), T.int64(1920)), "int32"), C: T.Buffer((T.int64(1), T.int64(15360)), "float16")):
        T.func_attr({"global_symbol": "matmul_1_15360_5120_40", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(960), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(960), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="wmma.accumulator")
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(5120), T.int64(15360)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(5120), T.int64(15360)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(T.int64(1), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(960), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(960), ax0_0_1_ax1_0_1_fused + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(40)):
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                                        with T.block("A_reindex_shared.dyn"):
                                            v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(128))
                                            v1 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(128))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_shared_dyn[v0, v1] = T.if_then_else(v0 < T.int64(1), A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(64)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(16))
                                        v1 = T.axis.spatial(T.int64(15360), ax0_0_1_ax1_0_1_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(16))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(4)):
                                with T.block("A_reindex_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0)
                                    v1_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(4) + ax1_0)
                                    T.reads(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, T.Cast("int64", A_1.strides[0]) * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(4) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(960), ax0_0_1_ax1_0_1_fused + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, T.Cast("int64", A_1.strides[0]) * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(960), ax0_0_1_ax1_0_1_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(4) + ax2_0_2)
                                    T.reads(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), A_1.data, A_1.elem_offset // T.Cast("int64", A_1.strides[0]) // T.int64(16) * (T.Cast("int64", A_1.strides[0]) // T.int64(16)) + A_1.elem_offset % T.Cast("int64", A_1.strides[0]) // T.int64(16), B.data, B.elem_offset // T.Cast("int64", B.strides[0]) // T.int64(16) * (T.Cast("int64", B.strides[0]) // T.int64(16)) + B.elem_offset % T.Cast("int64", B.strides[0]) // T.int64(16), C_1.data, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16))
                for ax2 in range(T.int64(1)):
                    for ax0_ax1_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(T.int64(1), T.int64(1)):
                            with T.block("C_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o, v2_o, v3_o = T.axis.remap("SSS", [ax0_0_1_ax1_0_1_fused, ax2_1, ax3])
                                v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                T.writes(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                A_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="wmma.accumulator", offset_factor=16)
                                C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // T.Cast("int64", A_1.strides[0]) // T.int64(16) * (T.Cast("int64", A_1.strides[0]) // T.int64(16)) + A_1.elem_offset % T.Cast("int64", A_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, T.Cast("int64", C_1.strides[0]) * T.int64(16), 2), C_1.strides[0], "row_major")
                    for ax0_ax1_ax3_ax4_ax5_fused_0 in range(T.int64(8)):
                        for ax0_ax1_ax3_ax4_ax5_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                            for ax0_ax1_ax3_ax4_ax5_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                with T.block("C_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1, v2 = T.axis.remap("SS", [ax0_0_1_ax1_0_1_fused, ax2])
                                    v3 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v4 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) // T.int64(16))
                                    v5 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) % T.int64(16))
                                    T.where(((ax0_ax1_ax3_ax4_ax5_fused_0 + ax0_ax1_ax3_ax4_ax5_fused_1) * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) % T.int64(256) // T.int64(16) < T.int64(1))
                                    T.reads(C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5])
                                    T.writes(C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)])
                                    C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)] = C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5]

    @T.prim_func
    def matmul_1_20480_5120_40(A: T.Buffer((T.int64(1), T.int64(5120)), "float16"), B_pack: T.Buffer((T.int64(640), T.int64(20480)), "int32"), scales: T.Buffer((T.int64(40), T.int64(20480)), "float16"), zeros_pack: T.Buffer((T.int64(40), T.int64(2560)), "int32"), C: T.Buffer((T.int64(1), T.int64(20480)), "float16")):
        T.func_attr({"global_symbol": "matmul_1_20480_5120_40", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="wmma.accumulator")
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(5120), T.int64(20480)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(5120), T.int64(20480)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(T.int64(1), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(1280), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(1280), ax0_0_1_ax1_0_1_fused + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(80)):
                        for ax0_ax1_fused_0 in range(T.int64(16)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(2)):
                                        with T.block("A_reindex_shared.dyn"):
                                            v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(64) + ax0_ax1_fused_1 * T.int64(64) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) // T.int64(64))
                                            v1 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(64) + ax0_ax1_fused_1 * T.int64(64) + ax0_ax1_fused_2 * T.int64(2) + ax0_ax1_fused_3) % T.int64(64))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_shared_dyn[v0, v1] = T.if_then_else(v0 < T.int64(1), A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(32)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(16))
                                        v1 = T.axis.spatial(T.int64(20480), ax0_0_1_ax1_0_1_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(16))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(2)):
                                with T.block("A_reindex_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0)
                                    v1_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(4) + ax2_0_1 * T.int64(2) + ax1_0)
                                    T.reads(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, T.Cast("int64", A_1.strides[0]) * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(4) + ax2_0_1 * T.int64(2) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(1280), ax0_0_1_ax1_0_1_fused + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, T.Cast("int64", A_1.strides[0]) * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(1280), ax0_0_1_ax1_0_1_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(320), ax2_0_0 * T.int64(4) + ax2_0_1 * T.int64(2) + ax2_0_2)
                                    T.reads(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), A_1.data, A_1.elem_offset // T.Cast("int64", A_1.strides[0]) // T.int64(16) * (T.Cast("int64", A_1.strides[0]) // T.int64(16)) + A_1.elem_offset % T.Cast("int64", A_1.strides[0]) // T.int64(16), B.data, B.elem_offset // T.Cast("int64", B.strides[0]) // T.int64(16) * (T.Cast("int64", B.strides[0]) // T.int64(16)) + B.elem_offset % T.Cast("int64", B.strides[0]) // T.int64(16), C_1.data, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16))
                for ax2 in range(T.int64(1)):
                    for ax0_ax1_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(T.int64(1), T.int64(1)):
                            with T.block("C_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o, v2_o, v3_o = T.axis.remap("SSS", [ax0_0_1_ax1_0_1_fused, ax2_1, ax3])
                                v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                T.writes(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                A_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="wmma.accumulator", offset_factor=16)
                                C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // T.Cast("int64", A_1.strides[0]) // T.int64(16) * (T.Cast("int64", A_1.strides[0]) // T.int64(16)) + A_1.elem_offset % T.Cast("int64", A_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, T.Cast("int64", C_1.strides[0]) * T.int64(16), 2), C_1.strides[0], "row_major")
                    for ax0_ax1_ax3_ax4_ax5_fused_0 in range(T.int64(2)):
                        for ax0_ax1_ax3_ax4_ax5_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                            for ax0_ax1_ax3_ax4_ax5_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax3_ax4_ax5_fused_3 in T.vectorized(T.int64(4)):
                                    with T.block("C_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1, v2 = T.axis.remap("SS", [ax0_0_1_ax1_0_1_fused, ax2])
                                        v3 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v4 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) // T.int64(16))
                                        v5 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(16))
                                        T.where((((ax0_ax1_ax3_ax4_ax5_fused_0 + ax0_ax1_ax3_ax4_ax5_fused_1) * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(256) // T.int64(16) < T.int64(1))
                                        T.reads(C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5])
                                        T.writes(C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)])
                                        C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)] = C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5]

    @T.prim_func
    def matmul_1_5120_20480_160(A: T.Buffer((T.int64(1), T.int64(20480)), "float16"), B_pack: T.Buffer((T.int64(2560), T.int64(5120)), "int32"), scales: T.Buffer((T.int64(160), T.int64(5120)), "float16"), zeros_pack: T.Buffer((T.int64(160), T.int64(640)), "int32"), C: T.Buffer((T.int64(1), T.int64(5120)), "float16")):
        T.func_attr({"global_symbol": "matmul_1_5120_20480_160", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="wmma.accumulator")
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(16), T.int64(20480)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(20480), T.int64(5120)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(16), T.int64(20480)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(20480), T.int64(5120)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(T.int64(320), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(80)):
                        for ax0_ax1_fused_0 in range(T.int64(16)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                                        with T.block("A_reindex_shared.dyn"):
                                            v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(256))
                                            v1 = T.axis.spatial(T.int64(20480), ax2_0_0 * T.int64(256) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(256))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_shared_dyn[v0, v1] = T.if_then_else(v0 < T.int64(1), A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(128)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(20480), ax2_0_0 * T.int64(256) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(16))
                                        v1 = T.axis.spatial(T.int64(5120), ax0_0_0_ax1_0_0_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(16))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(8)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(2)):
                                with T.block("A_reindex_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0)
                                    v1_o = T.axis.spatial(T.int64(1280), ax2_0_0 * T.int64(16) + ax2_0_1 * T.int64(2) + ax1_0)
                                    T.reads(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, T.Cast("int64", A_1.strides[0]) * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(1280), ax2_0_0 * T.int64(16) + ax2_0_1 * T.int64(2) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, T.Cast("int64", A_1.strides[0]) * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(1280), ax2_0_0 * T.int64(16) + ax2_0_1 * T.int64(2) + ax2_0_2)
                                    T.reads(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), A_1.data, A_1.elem_offset // T.Cast("int64", A_1.strides[0]) // T.int64(16) * (T.Cast("int64", A_1.strides[0]) // T.int64(16)) + A_1.elem_offset % T.Cast("int64", A_1.strides[0]) // T.int64(16), B.data, B.elem_offset // T.Cast("int64", B.strides[0]) // T.int64(16) * (T.Cast("int64", B.strides[0]) // T.int64(16)) + B.elem_offset % T.Cast("int64", B.strides[0]) // T.int64(16), C_1.data, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16))
                for ax2 in range(T.int64(1)):
                    for ax0_ax1_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(T.int64(1), T.int64(1)):
                            with T.block("C_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o, v2_o, v3_o = T.axis.remap("SSS", [ax0_0_0_ax1_0_0_fused, ax2_1, ax3])
                                v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                T.writes(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                A_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="wmma.accumulator", offset_factor=16)
                                C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // T.Cast("int64", A_1.strides[0]) // T.int64(16) * (T.Cast("int64", A_1.strides[0]) // T.int64(16)) + A_1.elem_offset % T.Cast("int64", A_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, T.Cast("int64", C_1.strides[0]) * T.int64(16), 2), C_1.strides[0], "row_major")
                    for ax0_ax1_ax3_ax4_ax5_fused_0 in range(T.int64(2)):
                        for ax0_ax1_ax3_ax4_ax5_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                            for ax0_ax1_ax3_ax4_ax5_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax3_ax4_ax5_fused_3 in T.vectorized(T.int64(4)):
                                    with T.block("C_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1, v2 = T.axis.remap("SS", [ax0_0_0_ax1_0_0_fused, ax2])
                                        v3 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v4 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) // T.int64(16))
                                        v5 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(128) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(16))
                                        T.where((((ax0_ax1_ax3_ax4_ax5_fused_0 + ax0_ax1_ax3_ax4_ax5_fused_1) * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(256) // T.int64(16) < T.int64(1))
                                        T.reads(C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5])
                                        T.writes(C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)])
                                        C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)] = C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5]

    @T.prim_func
    def matmul_1_5120_5120_40(A: T.Buffer((T.int64(1), T.int64(5120)), "float16"), B_pack: T.Buffer((T.int64(640), T.int64(5120)), "int32"), scales: T.Buffer((T.int64(40), T.int64(5120)), "float16"), zeros_pack: T.Buffer((T.int64(40), T.int64(640)), "int32"), C: T.Buffer((T.int64(1), T.int64(5120)), "float16")):
        T.func_attr({"global_symbol": "matmul_1_5120_5120_40", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="shared.dyn")
        C_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float16", scope="wmma.accumulator")
        A_reindex_shared_dyn = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(5120), T.int64(5120)), "float16", scope="shared.dyn")
        A_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(16), T.int64(5120)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(5120), T.int64(5120)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding(T.int64(2), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(1024), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(40), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(4) + ax0_0_2_ax1_0_2_fused + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(40)):
                        for ax0_ax1_fused_0 in range(T.int64(2)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                                        with T.block("A_reindex_shared.dyn"):
                                            v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(128))
                                            v1 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(128))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_shared_dyn[v0, v1] = T.if_then_else(v0 < T.int64(1), A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(64)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(64))
                                        v1 = T.axis.spatial(T.int64(5120), ax0_0_0_ax1_0_0_fused * T.int64(2560) + ax0_0_1_ax1_0_1_fused * T.int64(64) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(64))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(4)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(2)):
                                with T.block("A_reindex_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0)
                                    v1_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(2) + ax1_0)
                                    T.reads(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_shared_dyn[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, T.Cast("int64", A_1.strides[0]) * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(2) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(4) + ax0_0_2_ax1_0_2_fused + ax1_0)
                                    T.reads(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, T.Cast("int64", A_1.strides[0]) * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(4) + ax0_0_2_ax1_0_2_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(2) + ax2_0_2)
                                    T.reads(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o, T.int64(0), T.int64(0), T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16), A_1.data, A_1.elem_offset // T.Cast("int64", A_1.strides[0]) // T.int64(16) * (T.Cast("int64", A_1.strides[0]) // T.int64(16)) + A_1.elem_offset % T.Cast("int64", A_1.strides[0]) // T.int64(16), B.data, B.elem_offset // T.Cast("int64", B.strides[0]) // T.int64(16) * (T.Cast("int64", B.strides[0]) // T.int64(16)) + B.elem_offset % T.Cast("int64", B.strides[0]) // T.int64(16), C_1.data, C_1.elem_offset // T.Cast("int64", C_1.strides[0]) // T.int64(16) * (T.Cast("int64", C_1.strides[0]) // T.int64(16)) + C_1.elem_offset % T.Cast("int64", C_1.strides[0]) // T.int64(16))
                for ax2 in range(T.int64(1)):
                    for ax0_ax1_fused in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                        for ax2_1, ax3 in T.grid(T.int64(1), T.int64(1)):
                            with T.block("C_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(4) + ax0_ax1_fused)
                                v2_o, v3_o = T.axis.remap("SS", [ax2_1, ax3])
                                v4_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v5_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                T.writes(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                                A_1 = T.match_buffer(C_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_1_s0", "A_1_s1"), scope="wmma.accumulator", offset_factor=16)
                                C_1 = T.match_buffer(C_reindex_shared_dyn[v0_o, v1_o, v2_o, v3_o, T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_1_s0", "C_1_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // T.Cast("int64", A_1.strides[0]) // T.int64(16) * (T.Cast("int64", A_1.strides[0]) // T.int64(16)) + A_1.elem_offset % T.Cast("int64", A_1.strides[0]) // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, T.Cast("int64", C_1.strides[0]) * T.int64(16), 2), C_1.strides[0], "row_major")
                    for ax0_ax1_ax3_ax4_ax5_fused_0 in range(T.int64(1)):
                        for ax0_ax1_ax3_ax4_ax5_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                            for ax0_ax1_ax3_ax4_ax5_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_ax1_ax3_ax4_ax5_fused_3 in T.vectorized(T.int64(8)):
                                    with T.block("C_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(4) + (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(1024) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(256) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(8) + ax0_ax1_ax3_ax4_ax5_fused_3) // T.int64(256))
                                        v2 = T.axis.spatial(T.int64(1), ax2)
                                        v3 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v4 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(1024) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(256) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(8) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(256) // T.int64(16))
                                        v5 = T.axis.spatial(T.int64(16), (ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(1024) + ax0_ax1_ax3_ax4_ax5_fused_1 * T.int64(256) + ax0_ax1_ax3_ax4_ax5_fused_2 * T.int64(8) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(16))
                                        T.where((((ax0_ax1_ax3_ax4_ax5_fused_0 * T.int64(4) + ax0_ax1_ax3_ax4_ax5_fused_1) * T.int64(32) + ax0_ax1_ax3_ax4_ax5_fused_2) * T.int64(8) + ax0_ax1_ax3_ax4_ax5_fused_3) % T.int64(256) // T.int64(16) < T.int64(1))
                                        T.reads(C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5])
                                        T.writes(C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)])
                                        C[v4 + v0 * T.int64(16), v5 + v1 * T.int64(16)] = C_reindex_shared_dyn[v0, v1, v2, v3, v4, v5]

    @T.prim_func
    def matmul_dynm_15360_5120_40(a: T.handle, B_pack: T.Buffer((T.int64(640), T.int64(15360)), "int32"), scales: T.Buffer((T.int64(40), T.int64(15360)), "float16"), zeros_pack: T.Buffer((T.int64(40), T.int64(1920)), "int32"), c: T.handle):
        T.func_attr({"global_symbol": "matmul_dynm_15360_5120_40", "tir.noalias": T.bool(True)})
        m = T.int64()
        A = T.match_buffer(a, (m, T.int64(5120)), "float16")
        C = T.match_buffer(c, (m, T.int64(15360)), "float16")
        # with T.block("root"):
        C_reindex_pad_shared_dyn = T.alloc_buffer(((m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(15360)), "float16", scope="shared.dyn")
        C_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(32), T.int64(32)), "float16", scope="wmma.accumulator")
        A_reindex_pad_shared_dyn = T.alloc_buffer(((m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(5120)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(5120), T.int64(15360)), "float16", scope="shared.dyn")
        A_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(32), T.int64(64)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(64), T.int64(32)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding((m + T.int64(31)) // T.int64(32) * T.int64(480), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(64), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(2), ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(2) + ax0_0_2_ax1_0_2_fused // T.int64(2) + ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(960), ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(2) + ax0_0_2_ax1_0_2_fused % T.int64(2) + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32):v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32):v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(80)):
                        for ax0_ax1_fused_0 in range(T.int64(2)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                                        with T.block("A_reindex_pad_shared.dyn"):
                                            v0 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(64))
                                            v1 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(64))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_pad_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_pad_shared_dyn[v0, v1] = T.if_then_else(v0 < m, A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(16)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(32))
                                        v1 = T.axis.spatial(T.int64(15360), ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(32))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(4)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                                with T.block("A_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(2), ax0_0_2_ax1_0_2_fused // T.int64(2) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(4), ax2_0_1 + ax1_0)
                                    T.reads(A_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + v0_o * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + v0_o * T.int64(16) + T.int64(16), ax2_0_0 * T.int64(64) + v1_o * T.int64(16):ax2_0_0 * T.int64(64) + v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + v0_o * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + v0_o * T.int64(16) + T.int64(16), ax2_0_0 * T.int64(64) + v1_o * T.int64(16):ax2_0_0 * T.int64(64) + v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(4), ax2_0_1 + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(2), ax0_0_2_ax1_0_2_fused % T.int64(2) + ax1_0)
                                    T.reads(B_reindex_shared_dyn[ax2_0_0 * T.int64(64) + v0_o * T.int64(16):ax2_0_0 * T.int64(64) + v0_o * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + v1_o * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[ax2_0_0 * T.int64(64) + v0_o * T.int64(16):ax2_0_0 * T.int64(64) + v0_o * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + v1_o * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(2), ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(2) + ax0_0_2_ax1_0_2_fused // T.int64(2) + ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(960), ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(2) + ax0_0_2_ax1_0_2_fused % T.int64(2) + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(320), ax2_0_0 * T.int64(4) + ax2_0_1 + ax2_0_2)
                                    T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32):v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + T.int64(16)], A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + T.int64(16), v2_o * T.int64(16) - ax2_0_0 * T.int64(64):v2_o * T.int64(16) - ax2_0_0 * T.int64(64) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16) - ax2_0_0 * T.int64(64):v2_o * T.int64(16) - ax2_0_0 * T.int64(64) + T.int64(16), v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32):v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + T.int64(16)])
                                    T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32):v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + T.int64(16), v2_o * T.int64(16) - ax2_0_0 * T.int64(64):v2_o * T.int64(16) - ax2_0_0 * T.int64(64) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16) - ax2_0_0 * T.int64(64):v2_o * T.int64(16) - ax2_0_0 * T.int64(64) + T.int64(16), v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32):v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32):v1_o * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16))
                    for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                        with T.block("C_reindex_pad_shared.dyn_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(T.int64(2), ax0_0_2_ax1_0_2_fused // T.int64(2) + ax0_0)
                            v1_o = T.axis.spatial(T.int64(2), ax0_0_2_ax1_0_2_fused % T.int64(2) + ax1_0)
                            T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                            T.writes(C_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + v0_o * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + v0_o * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + v1_o * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + v1_o * T.int64(16) + T.int64(16)])
                            A_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                            C_1 = T.match_buffer(C_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + v0_o * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + v0_o * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + v1_o * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                            T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(16), 2), C_1.strides[0], "row_major")
                for ax0_ax1_fused_0 in range(T.int64(8)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            with T.block("C_reindex_pad_shared.dyn"):
                                v0 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax0_0_0_ax1_0_0_fused // T.int64(480) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(32))
                                v1 = T.axis.spatial(T.int64(15360), ax0_0_0_ax1_0_0_fused % T.int64(480) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(32))
                                T.reads(C_reindex_pad_shared_dyn[v0, v1])
                                T.writes(C[v0, v1])
                                if v0 < m:
                                    C[v0, v1] = C_reindex_pad_shared_dyn[v0, v1]

    @T.prim_func
    def matmul_dynm_20480_5120_40(a: T.handle, B_pack: T.Buffer((T.int64(640), T.int64(20480)), "int32"), scales: T.Buffer((T.int64(40), T.int64(20480)), "float16"), zeros_pack: T.Buffer((T.int64(40), T.int64(2560)), "int32"), c: T.handle):
        T.func_attr({"global_symbol": "matmul_dynm_20480_5120_40", "tir.noalias": T.bool(True)})
        m = T.int64()
        A = T.match_buffer(a, (m, T.int64(5120)), "float16")
        C = T.match_buffer(c, (m, T.int64(20480)), "float16")
        # with T.block("root"):
        C_reindex_pad_shared_dyn = T.alloc_buffer(((m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(20480)), "float16", scope="shared.dyn")
        C_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(32), T.int64(64)), "float16", scope="wmma.accumulator")
        A_reindex_pad_shared_dyn = T.alloc_buffer(((m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(5120)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(5120), T.int64(20480)), "float16", scope="shared.dyn")
        A_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(32), T.int64(128)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(128), T.int64(64)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding((m + T.int64(31)) // T.int64(32) * T.int64(2), thread="blockIdx.y"):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(160), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(2), ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(2) + ax0_0_2_ax1_0_2_fused // T.int64(4) + ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(1280), ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(640) + ax0_0_1_ax1_0_1_fused * T.int64(4) + ax0_0_2_ax1_0_2_fused % T.int64(4) + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(40)):
                        for ax0_ax1_fused_0 in range(T.int64(4)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                        with T.block("A_reindex_pad_shared.dyn"):
                                            v0 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(128))
                                            v1 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(128))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_pad_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_pad_shared_dyn[v0, v1] = T.if_then_else(v0 < m, A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(32)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(64))
                                        v1 = T.axis.spatial(T.int64(20480), ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + ax0_0_1_ax1_0_1_fused * T.int64(64) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(64))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(4)):
                                with T.block("A_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                    v0_o = T.axis.spatial(T.int64(2), ax0_0_2_ax1_0_2_fused // T.int64(4) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(8), ax2_0_1 * T.int64(4) + ax1_0)
                                    T.reads(A_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + v0_o * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + v0_o * T.int64(16) + T.int64(16), ax2_0_0 * T.int64(128) + v1_o * T.int64(16):ax2_0_0 * T.int64(128) + v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + v0_o * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + v0_o * T.int64(16) + T.int64(16), ax2_0_0 * T.int64(128) + v1_o * T.int64(16):ax2_0_0 * T.int64(128) + v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(8), ax2_0_1 * T.int64(4) + ax0_0)
                                    v1_o = T.axis.spatial(T.int64(4), ax0_0_2_ax1_0_2_fused % T.int64(4) + ax1_0)
                                    T.reads(B_reindex_shared_dyn[ax2_0_0 * T.int64(128) + v0_o * T.int64(16):ax2_0_0 * T.int64(128) + v0_o * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + ax0_0_1_ax1_0_1_fused * T.int64(64) + v1_o * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + ax0_0_1_ax1_0_1_fused * T.int64(64) + v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[ax2_0_0 * T.int64(128) + v0_o * T.int64(16):ax2_0_0 * T.int64(128) + v0_o * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + ax0_0_1_ax1_0_1_fused * T.int64(64) + v1_o * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + ax0_0_1_ax1_0_1_fused * T.int64(64) + v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(2), ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(2) + ax0_0_2_ax1_0_2_fused // T.int64(4) + ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(1280), ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(640) + ax0_0_1_ax1_0_1_fused * T.int64(4) + ax0_0_2_ax1_0_2_fused % T.int64(4) + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(4) + ax2_0_2)
                                    T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + T.int64(16)], A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + T.int64(16), v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + T.int64(16)])
                                    T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + T.int64(16), v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(64) - ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16))
                    for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                        with T.block("C_reindex_pad_shared.dyn_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(T.int64(2), ax0_0_2_ax1_0_2_fused // T.int64(4) + ax0_0)
                            v1_o = T.axis.spatial(T.int64(4), ax0_0_2_ax1_0_2_fused % T.int64(4) + ax1_0)
                            T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                            T.writes(C_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + v0_o * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + v0_o * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + ax0_0_1_ax1_0_1_fused * T.int64(64) + v1_o * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + ax0_0_1_ax1_0_1_fused * T.int64(64) + v1_o * T.int64(16) + T.int64(16)])
                            A_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                            C_1 = T.match_buffer(C_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + v0_o * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + v0_o * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + ax0_0_1_ax1_0_1_fused * T.int64(64) + v1_o * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + ax0_0_1_ax1_0_1_fused * T.int64(64) + v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                            T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(16), 2), C_1.strides[0], "row_major")
                for ax0_ax1_fused_0 in range(T.int64(1)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                                with T.block("C_reindex_pad_shared.dyn"):
                                    v0 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax0_0_0_ax1_0_0_fused // T.int64(2) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(64))
                                    v1 = T.axis.spatial(T.int64(20480), ax0_0_0_ax1_0_0_fused % T.int64(2) * T.int64(10240) + ax0_0_1_ax1_0_1_fused * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(64))
                                    T.reads(C_reindex_pad_shared_dyn[v0, v1])
                                    T.writes(C[v0, v1])
                                    if v0 < m:
                                        C[v0, v1] = C_reindex_pad_shared_dyn[v0, v1]

    @T.prim_func
    def matmul_dynm_5120_20480_160(a: T.handle, B_pack: T.Buffer((T.int64(2560), T.int64(5120)), "int32"), scales: T.Buffer((T.int64(160), T.int64(5120)), "float16"), zeros_pack: T.Buffer((T.int64(160), T.int64(640)), "int32"), c: T.handle):
        T.func_attr({"global_symbol": "matmul_dynm_5120_20480_160", "tir.noalias": T.bool(True)})
        m = T.int64()
        A = T.match_buffer(a, (m, T.int64(20480)), "float16")
        C = T.match_buffer(c, (m, T.int64(5120)), "float16")
        # with T.block("root"):
        C_reindex_pad_shared_dyn = T.alloc_buffer(((m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(5120)), "float16", scope="shared.dyn")
        C_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(16), T.int64(16)), "float16", scope="wmma.accumulator")
        A_reindex_pad_shared_dyn = T.alloc_buffer(((m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(20480)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(5120), T.int64(20480)), "float16", scope="shared.dyn")
        A_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(16), T.int64(128)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(16), T.int64(128)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding((m + T.int64(31)) // T.int64(32) * T.int64(64), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(10), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(2), ax0_0_0_ax1_0_0_fused // T.int64(32) + ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(10) + ax0_0_1_ax1_0_1_fused + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(160)):
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                                        with T.block("A_reindex_pad_shared.dyn"):
                                            v0 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(128))
                                            v1 = T.axis.spatial(T.int64(20480), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(128))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_pad_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_pad_shared_dyn[v0, v1] = T.if_then_else(v0 < m, A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(64)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(5120), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(128))
                                        v1 = T.axis.spatial(T.int64(20480), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(128))
                                        T.reads(B_pack[v1 // T.int64(8), v0], zeros_pack[v1 // T.int64(128), v0 // T.int64(8)], scales[v1 // T.int64(128), v0])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v1 // T.int64(8), v0]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v1 // T.int64(128), v0 // T.int64(8)]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v1 // T.int64(128), v0]
                        for ax2_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(4)):
                                with T.block("A_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                    v1_o = T.axis.spatial(T.int64(8), ax2_0_1 * T.int64(4) + ax1_0)
                                    T.reads(A_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), ax2_0_0 * T.int64(128) + v1_o * T.int64(16):ax2_0_0 * T.int64(128) + v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), ax2_0_0 * T.int64(128) + v1_o * T.int64(16):ax2_0_0 * T.int64(128) + v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(4)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v1_o = T.axis.spatial(T.int64(8), ax2_0_1 * T.int64(4) + ax1_0)
                                    T.reads(B_reindex_shared_dyn[ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(16) + T.int64(16), ax2_0_0 * T.int64(128) + v1_o * T.int64(16):ax2_0_0 * T.int64(128) + v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(16) + T.int64(16), ax2_0_0 * T.int64(128) + v1_o * T.int64(16):ax2_0_0 * T.int64(128) + v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "col_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(2), ax0_0_0_ax1_0_0_fused // T.int64(32) + ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(10) + ax0_0_1_ax1_0_1_fused + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(1280), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(4) + ax2_0_2)
                                    T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)], A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16), v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16)])
                                    T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16), v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):v0_o * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16))
                    for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                        with T.block("C_reindex_pad_shared.dyn_wmma.accumulator_o"):
                            T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            T.writes(C_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(16) + T.int64(16)])
                            A_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                            C_1 = T.match_buffer(C_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                            T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(16), 2), C_1.strides[0], "row_major")
                for ax0_ax1_fused_0 in range(T.int64(8)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            with T.block("C_reindex_pad_shared.dyn"):
                                v0 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(16))
                                v1 = T.axis.spatial(T.int64(5120), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(16))
                                T.reads(C_reindex_pad_shared_dyn[v0, v1])
                                T.writes(C[v0, v1])
                                if v0 < m:
                                    C[v0, v1] = C_reindex_pad_shared_dyn[v0, v1]

    @T.prim_func
    def matmul_dynm_5120_5120_40(a: T.handle, B_pack: T.Buffer((T.int64(640), T.int64(5120)), "int32"), scales: T.Buffer((T.int64(40), T.int64(5120)), "float16"), zeros_pack: T.Buffer((T.int64(40), T.int64(640)), "int32"), c: T.handle):
        T.func_attr({"global_symbol": "matmul_dynm_5120_5120_40", "tir.noalias": T.bool(True)})
        m = T.int64()
        A = T.match_buffer(a, (m, T.int64(5120)), "float16")
        C = T.match_buffer(c, (m, T.int64(5120)), "float16")
        # with T.block("root"):
        C_reindex_pad_shared_dyn = T.alloc_buffer(((m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(5120)), "float16", scope="shared.dyn")
        C_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(16), T.int64(16)), "float16", scope="wmma.accumulator")
        A_reindex_pad_shared_dyn = T.alloc_buffer(((m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(5120)), "float16", scope="shared.dyn")
        B_reindex_shared_dyn = T.alloc_buffer((T.int64(5120), T.int64(5120)), "float16", scope="shared.dyn")
        A_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(16), T.int64(128)), "float16", scope="wmma.matrix_a")
        B_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(128), T.int64(16)), "float16", scope="wmma.matrix_b")
        for ax0_0_0_ax1_0_0_fused in T.thread_binding((m + T.int64(31)) // T.int64(32) * T.int64(32), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": T.int64(512), "pragma_unroll_explicit": T.int64(1)}):
            for ax0_0_1_ax1_0_1_fused in T.thread_binding(T.int64(20), thread="blockIdx.x"):
                for ax0_0_2_ax1_0_2_fused in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                    for ax0_0_3_init, ax1_0_3_init, ax0_0_4_init, ax1_0_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("matmul_o_init"):
                            v0_o = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(2), ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(2) + ax0_0_1_ax1_0_1_fused // T.int64(10) + ax0_0_3_init + ax0_0_4_init)
                            v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(10) + ax0_0_1_ax1_0_1_fused % T.int64(10) + ax1_0_3_init + ax1_0_4_init)
                            T.reads()
                            T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32):v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                            C_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32):v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                            T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.float32(0))
                    for ax2_0_0 in range(T.int64(40)):
                        for ax0_ax1_fused_0 in range(T.int64(8)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(8)):
                                        with T.block("A_reindex_pad_shared.dyn"):
                                            v0 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(128))
                                            v1 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(128))
                                            T.reads(A[v0, v1])
                                            T.writes(A_reindex_pad_shared_dyn[v0, v1])
                                            T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                            A_reindex_pad_shared_dyn[v0, v1] = T.if_then_else(v0 < m, A[v0, v1], T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(64)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("B_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(T.int64(5120), ax2_0_0 * T.int64(128) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(16))
                                        v1 = T.axis.spatial(T.int64(5120), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(16))
                                        T.reads(B_pack[v0 // T.int64(8), v1], zeros_pack[v0 // T.int64(128), v1 // T.int64(8)], scales[v0 // T.int64(128), v1])
                                        T.writes(B_reindex_shared_dyn[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        B_reindex_shared_dyn[v0, v1] = T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("int64", B_pack[v0 // T.int64(8), v1]), v0 % T.int64(8) * T.int64(4)), T.int64(15)) - T.Cast("int64", T.Cast("int32", T.bitwise_and(T.shift_right(T.Cast("int64", zeros_pack[v0 // T.int64(128), v1 // T.int64(8)]), v1 % T.int64(8) * T.int64(4)), T.int64(15)) + T.int64(1)))) * scales[v0 // T.int64(128), v1]
                        for ax2_0_1 in range(T.int64(2)):
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(4)):
                                with T.block("A_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                    v1_o = T.axis.spatial(T.int64(8), ax2_0_1 * T.int64(4) + ax1_0)
                                    T.reads(A_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) + T.int64(16), ax2_0_0 * T.int64(128) + v1_o * T.int64(16):ax2_0_0 * T.int64(128) + v1_o * T.int64(16) + T.int64(16)])
                                    T.writes(A_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)])
                                    A_1 = T.match_buffer(A_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) + T.int64(16), ax2_0_0 * T.int64(128) + v1_o * T.int64(16):ax2_0_0 * T.int64(128) + v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(A_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0):T.int64(16), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0, ax1_0 in T.grid(T.int64(4), T.int64(1)):
                                with T.block("B_reindex_shared.dyn_wmma.matrix_b_o"):
                                    v0_o = T.axis.spatial(T.int64(8), ax2_0_1 * T.int64(4) + ax0_0)
                                    T.reads(B_reindex_shared_dyn[ax2_0_0 * T.int64(128) + v0_o * T.int64(16):ax2_0_0 * T.int64(128) + v0_o * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) + T.int64(16)])
                                    T.writes(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), T.int64(0):T.int64(16)])
                                    A_1 = T.match_buffer(B_reindex_shared_dyn[ax2_0_0 * T.int64(128) + v0_o * T.int64(16):ax2_0_0 * T.int64(128) + v0_o * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                    C_1 = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v0_o * T.int64(16):v0_o * T.int64(16) + T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0_3, ax1_0_3, ax2_0_2, ax0_0_4, ax1_0_4 in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(1), T.int64(1)):
                                with T.block("matmul_o_update"):
                                    v0_o = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(2), ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(2) + ax0_0_1_ax1_0_1_fused // T.int64(10) + ax0_0_3 + ax0_0_4)
                                    v1_o = T.axis.spatial(T.int64(320), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(10) + ax0_0_1_ax1_0_1_fused % T.int64(10) + ax1_0_3 + ax1_0_4)
                                    v2_o = T.axis.reduce(T.int64(320), ax2_0_0 * T.int64(8) + ax2_0_1 * T.int64(4) + ax2_0_2)
                                    T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32):v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)], A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32):v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + T.int64(16), v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16)], B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)])
                                    T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32):v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": T.int64(1024), "meta_schedule.thread_extent_low_inclusive": T.int64(32), "warp_execution": T.int64(1)})
                                    A_1 = T.match_buffer(A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32):v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + T.int64(16), v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                    B = T.match_buffer(B_reindex_shared_dyn_wmma_matrix_b[v2_o * T.int64(16) - ax2_0_0 * T.int64(128):v2_o * T.int64(16) - ax2_0_0 * T.int64(128) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                    C_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32):v0_o * T.int64(16) - ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + T.int64(16), v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160):v1_o * T.int64(16) - ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) - ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C_1.data, C_1.elem_offset // C_1.strides[0] // T.int64(16) * (C_1.strides[0] // T.int64(16)) + C_1.elem_offset % C_1.strides[0] // T.int64(16))
                    for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                        with T.block("C_reindex_pad_shared.dyn_wmma.accumulator_o"):
                            T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0):T.int64(16), T.int64(0):T.int64(16)])
                            T.writes(C_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) + T.int64(16)])
                            A_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0):T.int64(16), T.int64(0):T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                            C_1 = T.match_buffer(C_reindex_pad_shared_dyn[ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16):ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) + T.int64(16), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16):ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                            T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * T.int64(16), 2), C_1.strides[0], "row_major")
                for ax0_ax1_fused_0 in range(T.int64(8)):
                    for ax0_ax1_fused_1 in T.thread_binding(T.int64(1), thread="threadIdx.y"):
                        for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            with T.block("C_reindex_pad_shared.dyn"):
                                v0 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax0_0_0_ax1_0_0_fused // T.int64(32) * T.int64(32) + ax0_0_1_ax1_0_1_fused // T.int64(10) * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) // T.int64(16))
                                v1 = T.axis.spatial(T.int64(5120), ax0_0_0_ax1_0_0_fused % T.int64(32) * T.int64(160) + ax0_0_1_ax1_0_1_fused % T.int64(10) * T.int64(16) + (ax0_ax1_fused_0 * T.int64(32) + ax0_ax1_fused_1 * T.int64(32) + ax0_ax1_fused_2) % T.int64(16))
                                T.reads(C_reindex_pad_shared_dyn[v0, v1])
                                T.writes(C[v0, v1])
                                if v0 < m:
                                    C[v0, v1] = C_reindex_pad_shared_dyn[v0, v1]
