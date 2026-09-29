# Continuous warmup investigation results

Generated: 2026-07-18T02:23:35.043

| exp | variant | model | nrmse | nrmse_global | vpt_lyap | notes |
|-----|---------|-------|-------|--------------|----------|-------|
| E1 | cold_package_predict | ContinuousESN | 1.4782807405733622 | 0.8686309038147971 | 0.21734399999999998 | wall_train_s=18.117923542; wall_ar_s=2.060436125; mode=full |
| E1 | warm_train_terminal_u0 | ContinuousESN | 1.2292196160902893 | 0.7208150983122181 | 4.129536 | u0_norm=13.724870901821692; wall_train_s=18.117923542; wall_ar_s=0.952867458; wall_warmup_collect_s=16.314653625; mode=full |
| E1 | seeded_zero_matches_cold_maxabs | ContinuousESN |  |  |  | max_abs_diff=0.0; match_ok=true; mode=full |
| E1 | cold_horizon | ContinuousESN | 0.7535572065523274 |  |  | horizon_steps=28; horizon_lyap=0.507136; mode=full |
| E1 | warm_horizon | ContinuousESN | 0.21463110240826375 |  |  | horizon_steps=28; horizon_lyap=0.507136; mode=full |
| E1 | cold_horizon | ContinuousESN | 0.6853049713043267 |  |  | horizon_steps=55; horizon_lyap=0.99616; mode=full |
| E1 | warm_horizon | ContinuousESN | 0.14675113554137212 |  |  | horizon_steps=55; horizon_lyap=0.99616; mode=full |
| E1 | cold_horizon | ContinuousESN | 1.3393780477247423 |  |  | horizon_steps=110; horizon_lyap=1.99232; mode=full |
| E1 | warm_horizon | ContinuousESN | 0.08497214024585852 |  |  | horizon_steps=110; horizon_lyap=1.99232; mode=full |
| E1 | cold_horizon | ContinuousESN | 1.506054216154576 |  |  | horizon_steps=166; horizon_lyap=3.006592; mode=full |
| E1 | warm_horizon | ContinuousESN | 0.093388739272057 |  |  | horizon_steps=166; horizon_lyap=3.006592; mode=full |
| E1 | cold_horizon | ContinuousESN | 1.6554797621233028 |  |  | horizon_steps=221; horizon_lyap=4.002752; mode=full |
| E1 | warm_horizon | ContinuousESN | 0.13715717405603614 |  |  | horizon_steps=221; horizon_lyap=4.002752; mode=full |
| E1 | cold_horizon | ContinuousESN | 1.5117492481722878 |  |  | horizon_steps=331; horizon_lyap=5.9950719999999995; mode=full |
| E1 | warm_horizon | ContinuousESN | 0.6692777566540231 |  |  | horizon_steps=331; horizon_lyap=5.9950719999999995; mode=full |
| E2 | cold_package_predict | SciMLProblemReservoir | 1.470005354266804 | 0.8610435641262816 | 0.018112 | mode=full |
| E2 | warm_train_terminal_u0 | SciMLProblemReservoir | 1.312672661950845 | 0.7709100283139908 | 2.843584 | u0_norm=13.604509626860247; mode=full |
| E2 | remake_prob_u0_package_predict | SciMLProblemReservoir | 1.312672661950845 | 0.7709100283139908 | 2.843584 | mode=full |
| E3 | K=0_cold | ContinuousESN | 1.4782807405733622 | 0.8686309038147971 | 0.21734399999999998 | warmup_len=0; mode=full |
| E3 | K=10 | ContinuousESN | 1.1420303622679457 | 0.6695443570238542 | 4.872128 | warmup_len=10; mode=full |
| E3 | K=50 | ContinuousESN | 1.272365316049519 | 0.7447275884595839 | 4.129536 | warmup_len=50; mode=full |
| E3 | K=100 | ContinuousESN | 1.2309111911851687 | 0.7277626070587372 | 4.129536 | warmup_len=100; mode=full |
| E3 | K=250 | ContinuousESN | 1.2163625936067015 | 0.7141959888660526 | 4.129536 | warmup_len=250; mode=full |
| E3 | K=500 | ContinuousESN | 1.2843248580360316 | 0.7517730578199671 | 4.129536 | warmup_len=500; mode=full |
| E3 | K=1000 | ContinuousESN | 1.1789022279431667 | 0.6993686867334836 | 4.129536 | warmup_len=1000; mode=full |
| E3 | K=2000 | ContinuousESN | 1.1835926940405561 | 0.692584966096159 | 4.129536 | warmup_len=2000; mode=full |
| E4 | shuffled_train_terminal | ContinuousESN | 13.54940655959303 | 9.826713305822715 | 0.0 | u0_norm=13.724870901821683; mode=full |
| E4 | randn | ContinuousESN | 13.558170985119219 | 9.833322066308458 | 0.0 | u0_norm=17.851901593047117; mode=full |
| E4 | zeros | ContinuousESN | 1.4782807405733622 | 0.8686309038147971 | 0.21734399999999998 | u0_norm=0.0; mode=full |
| E4 | train_terminal | ContinuousESN | 1.2292196160902893 | 0.7208150983122181 | 4.129536 | u0_norm=13.724870901821692; mode=full |
| E4 | oracle_test_prefix | ContinuousESN | 1.4509837707185447 | 0.8508405432103581 | 0.959936 | u0_norm=15.374881292468654; mode=full |
| E5 | cold | ContinuousESN | 0.7535572065523274 |  |  | horizon_steps=28; horizon_lyap=0.507136; mode=full |
| E5 | warm_train_terminal | ContinuousESN | 0.21463110240826375 |  |  | horizon_steps=28; horizon_lyap=0.507136; mode=full |
| E5 | cold | ContinuousESN | 0.6853049713043267 |  |  | horizon_steps=55; horizon_lyap=0.99616; mode=full |
| E5 | warm_train_terminal | ContinuousESN | 0.14675113554137212 |  |  | horizon_steps=55; horizon_lyap=0.99616; mode=full |
| E5 | cold | ContinuousESN | 1.3393780477247423 |  |  | horizon_steps=110; horizon_lyap=1.99232; mode=full |
| E5 | warm_train_terminal | ContinuousESN | 0.08497214024585852 |  |  | horizon_steps=110; horizon_lyap=1.99232; mode=full |
| E5 | cold | ContinuousESN | 1.506054216154576 |  |  | horizon_steps=166; horizon_lyap=3.006592; mode=full |
| E5 | warm_train_terminal | ContinuousESN | 0.093388739272057 |  |  | horizon_steps=166; horizon_lyap=3.006592; mode=full |
| E5 | cold | ContinuousESN | 1.6554797621233028 |  |  | horizon_steps=221; horizon_lyap=4.002752; mode=full |
| E5 | warm_train_terminal | ContinuousESN | 0.13715717405603614 |  |  | horizon_steps=221; horizon_lyap=4.002752; mode=full |
| E5 | cold | ContinuousESN | 1.5117492481722878 |  |  | horizon_steps=331; horizon_lyap=5.9950719999999995; mode=full |
| E5 | warm_train_terminal | ContinuousESN | 0.6692777566540231 |  |  | horizon_steps=331; horizon_lyap=5.9950719999999995; mode=full |
| E6 | continuous_st0 | ContinuousESN |  |  |  | has_carry=false; mode=full |
| E6 | continuous_st_after_train | ContinuousESN |  |  |  | has_carry=false; mode=full |
| E6 | continuous_st_after_collectstates | ContinuousESN |  |  |  | has_carry=false; mode=full |
| E6 | discrete_st_after_train | ESN |  |  |  | has_carry=true; mode=full |
| E7 | cold_fresh_st | ESN | 1.55491741812422 |  | 0.0 | mode=full |
| E7 | warm_st_after_train | ESN | 1.1363799866710107 |  | 4.1114239999999995 | mode=full |
| E7 | rewarm_collectstates_then_ar | ESN | 1.1363799866710107 |  | 4.1114239999999995 | mode=full |
| E8 | cold | ContinuousESN | 1.4840906014114033 |  |  | washout=200; mode=full |
| E8 | warm_full_train | ContinuousESN | 1.1839060794299072 |  |  | washout=200; mode=full |
| E8 | warm_post_washout_tail | ContinuousESN | 1.1280934497932662 |  |  | washout=200; mode=full |
| META | suite_wall_s | harness |  |  |  | wall_s=275.113651291; mode=full |
