open HolKernel boolLib bossLib Parse binderLib

open chap3Theory
open pred_setTheory
open termTheory
open boolSimps
open normal_orderTheory
open churchboolTheory
open reductionEval

fun Store_thm(n,t,tac) = store_thm(n,t,tac) before export_rewrites [n]

val _ = new_theory "churchnum"

val _ = set_trace "Unicode" 1

val church_def = Define`
  church n = LAM "z" (LAM "s" (FUNPOW ((@@) (VAR "s")) n (VAR "z")))
`

val FUNPOW_SUC = arithmeticTheory.FUNPOW_SUC

val size_funpow = store_thm(
  "size_funpow",
  ``size (FUNPOW ((@@) f) n x) = (size f + 1) * n + size x``,
  Induct_on `n` THEN
  SRW_TAC [ARITH_ss][FUNPOW_SUC, arithmeticTheory.LEFT_ADD_DISTRIB,
                     arithmeticTheory.MULT_CLAUSES]);

val church_11 = store_thm(
  "church_11",
  ``(church m = church n) ⇔ (m = n)``,
  SRW_TAC [][church_def, EQ_IMP_THM] THEN
  POP_ASSUM (MP_TAC o Q.AP_TERM `size`) THEN
  SRW_TAC [ARITH_ss][size_funpow, arithmeticTheory.LEFT_ADD_DISTRIB]);
val _ = export_rewrites ["church_11"]

val bnf_church = store_thm(
  "bnf_church",
  ``∀n. bnf (church n)``,
  SRW_TAC [][church_def] THEN
  Induct_on `n` THEN SRW_TAC [][] THEN
  SRW_TAC [][FUNPOW_SUC]);
val _ = export_rewrites ["bnf_church"]

val is_abs_church = Store_thm(
  "is_abs_church",
  ``is_abs (church n)``,
  SRW_TAC [][church_def]);

val church_lameq_11 = store_thm(
  "church_lameq_11",
  ``(church m == church n) ⇔ (m = n)``,
  SRW_TAC [][EQ_IMP_THM, chap2Theory.lam_eq_rules] THEN
  `∃Z. church m -β->* Z ∧ church n -β->* Z`
     by METIS_TAC [beta_CR, theorem3_13, prop3_18] THEN
  `church m = church n`
     by METIS_TAC [corollary3_2_1, beta_normal_form_bnf, bnf_church] THEN
  FULL_SIMP_TAC (srw_ss()) []);

val FV_church = store_thm(
  "FV_church",
  ``FV (church n) = {}``,
  SRW_TAC [][church_def] THEN
  `(n = 0) ∨ (∃m. n = SUC m)`
    by METIS_TAC [TypeBase.nchotomy_of ``:num``] THEN
  SRW_TAC [][] THENL [
    SRW_TAC [CONJ_ss] [EXTENSION],
    Q_TAC SUFF_TAC
          `FV (FUNPOW ((@@) (VAR "s")) (SUC m) (VAR "z")) = {"s"; "z"}`
          THEN1 SRW_TAC [CONJ_ss][pred_setTheory.EXTENSION] THEN
    Induct_on `m` THEN SRW_TAC [][] THENL [
      SRW_TAC [][EXTENSION],
      SRW_TAC [][Once FUNPOW_SUC] THEN
      SRW_TAC [][EXTENSION] THEN METIS_TAC []
    ]
  ]);
val _ = export_rewrites ["FV_church"]

val csuc_def = Define`
  csuc = LAM "n" (LAM "z" (LAM "s"
            (VAR "s" @@ (VAR "n" @@ VAR "z" @@ VAR "s"))))
`;

val tpm_funpow_app = store_thm(
  "tpm_funpow_app",
  ``tpm pi (FUNPOW ($@@ f) n x) = FUNPOW ($@@ (tpm pi f)) n (tpm pi x)``,
  Induct_on `n` THEN SRW_TAC [][FUNPOW_SUC]);
val _  = export_rewrites ["tpm_funpow_app"]

val FV_funpow_app = store_thm(
  "FV_funpow_app",
  ``FV (FUNPOW ($@@ f) n x) ⊆ FV f ∪ FV x``,
  Induct_on `n` THEN SRW_TAC [][FUNPOW_SUC]);

val FV_funpow_app_I = store_thm(
  "FV_funpow_app_I",
  ``v ∈ FV x ⇒ v ∈ FV (FUNPOW ((@@) f) n x)``,
  Induct_on `n` THEN SRW_TAC [][FUNPOW_SUC]);

val FV_funpow_app_E = store_thm(
  "FV_funpow_app_E",
  ``v ∈ FV (FUNPOW ((@@) f) n x) ⇒ v ∈ FV f ∨ v ∈ FV x``,
  MATCH_ACCEPT_TAC (REWRITE_RULE [IN_UNION, SUBSET_DEF] FV_funpow_app));

val fresh_funpow_app_I = store_thm(
  "fresh_funpow_app_I",
  ``v ∉ FV f ∧ v ∉ FV x ⇒ v ∉ FV (FUNPOW ((@@) f) n x)``,
  METIS_TAC [FV_funpow_app_E]);
val _ = export_rewrites ["fresh_funpow_app_I"]

val FV_funpow_app_vars = store_thm(
  "FV_funpow_app_vars",
  ``FV (FUNPOW ($@@ (VAR f)) n (VAR x)) ⊆ {f; x}``,
  Q_TAC SUFF_TAC `FV (VAR f) ∪ FV (VAR x) = {f; x}`
        THEN1 METIS_TAC [FV_funpow_app] THEN
  SRW_TAC [][EXTENSION]);

val bnf_FUNPOW = store_thm(
  "bnf_FUNPOW",
  ``∀x. bnf (FUNPOW ((@@) (VAR v)) n x) ⇔ bnf x``,
  Induct_on `n` THEN SRW_TAC [][FUNPOW_SUC]);
val _ = export_rewrites ["bnf_FUNPOW"]


val SUB_funpow_app = store_thm(
  "SUB_funpow_app",
  ``[M/v] (FUNPOW ($@@ f) n x) = FUNPOW ($@@ ([M/v]f)) n ([M/v]x)``,
  Induct_on `n` THEN SRW_TAC [][FUNPOW_SUC]);
val _ = export_rewrites ["SUB_funpow_app"]

val RTC1_step = CONJUNCT2 (SPEC_ALL relationTheory.RTC_RULES)

val ccbeta_church = store_thm(
  "ccbeta_church",
  ``church n -β-> M ⇔ F``,
  METIS_TAC [beta_normal_form_bnf, corollary3_2_1, bnf_church]);
val _ = export_rewrites ["ccbeta_church"]

val normorder_church = store_thm(
  "normorder_church",
  ``church n -n-> M ⇔ F``,
  METIS_TAC [normorder_ccbeta, ccbeta_church])

val church_eq = store_thm(
  "church_eq",
  ``(∀s. church n ≠ VAR s) ∧ (∀M N. church n ≠ M @@ N)``,
  SRW_TAC [][church_def]);
val _ = export_rewrites ["church_eq"]


val normorder_funpow_var = store_thm(
  "normorder_funpow_var",
  ``∀M. FUNPOW ((@@) (VAR v)) n x -n-> M ⇔
        ∃y. (M = FUNPOW ((@@) (VAR v)) n y) ∧ x -n-> y``,
  Induct_on `n`  THEN SRW_TAC [DNF_ss][FUNPOW_SUC, normorder_rwts]);
val normorderstar_funpow_var = store_thm(
  "normorderstar_funpow_var",
  ``FUNPOW ((@@) (VAR v)) n x -n->* M ⇔
        ∃y. (M = FUNPOW ((@@) (VAR v)) n y) ∧ x -n->* y``,
  EQ_TAC THENL [
    Q_TAC SUFF_TAC `∀M N. M -n->* N ⇒
                           ∀v n x. (M = FUNPOW ((@@) (VAR v)) n x) ⇒
                                    ∃y. (N = FUNPOW ((@@) (VAR v)) n y) ∧
                                        x -n->* y`
          THEN1 METIS_TAC [] THEN
    HO_MATCH_MP_TAC relationTheory.RTC_INDUCT THEN SRW_TAC [][] THENL [
      METIS_TAC [relationTheory.RTC_RULES],
      FULL_SIMP_TAC (srw_ss()) [normorder_funpow_var] THEN
      METIS_TAC [normorder_funpow_var, relationTheory.RTC_RULES]
    ],
    Q_TAC SUFF_TAC
      `∀x y. x -n->* y ⇒
              FUNPOW ((@@) (VAR v)) n x -n->* FUNPOW ((@@) (VAR v)) n y`
      THEN1 METIS_TAC [] THEN
    HO_MATCH_MP_TAC relationTheory.RTC_INDUCT THEN SRW_TAC [][] THEN
    METIS_TAC [relationTheory.RTC_RULES, normorder_funpow_var]
  ]);


val ccbeta_funpow_var = store_thm(
  "ccbeta_funpow_var",
  ``∀M. FUNPOW ((@@) (VAR v)) n x -β-> M ⇔
        ∃y. (M = FUNPOW ((@@) (VAR v)) n y) ∧ x -β-> y``,
  Induct_on `n`  THEN SRW_TAC [DNF_ss][FUNPOW_SUC, ccbeta_rwt]);

val ccbeta_funpow = store_thm(
  "ccbeta_funpow",
  ``M -β-> N ⇒ FUNPOW ((@@)P) n M -β-> FUNPOW ((@@)P) n N``,
  Induct_on `n` THEN SRW_TAC [][FUNPOW_SUC] THEN
  METIS_TAC [cc_beta_thm]);

val betastar_funpow_cong = store_thm(
  "betastar_funpow_cong",
  ``M -β->* N ⇒ FUNPOW ((@@) P) n M -β->* FUNPOW ((@@)P) n N``,
  MAP_EVERY Q.ID_SPEC_TAC [`N`, `M`] THEN
  HO_MATCH_MP_TAC relationTheory.RTC_INDUCT THEN SRW_TAC [][] THEN
  METIS_TAC [relationTheory.RTC_RULES, ccbeta_funpow]);


val church_behaviour = store_thm(
  "church_behaviour",
  ``church n @@ x @@ f -n->* FUNPOW ($@@ f) n x``,
  SRW_TAC [][church_def] THEN FRESH_TAC THEN
  SRW_TAC [NORMSTAR_ss][SUB_funpow_app]);

val csuc_behaviour = store_thm(
  "csuc_behaviour",
  ``∀n. (csuc @@ (church n)) -n->* church (SUC n)``,
  SIMP_TAC (betafy (srw_ss())) [csuc_def, church_behaviour, FUNPOW_SUC,
                                Q.SPEC `SUC n` church_def]);

val cplus_def = Define`
  cplus = LAM "m" (LAM "n" (LAM "z" (LAM "s"
             (VAR "m" @@ (VAR "n" @@ VAR "z" @@ VAR "s") @@ VAR "s"))))
`;

fun bsrw_ss() = betafy(srw_ss())

val cplus_behaviour = store_thm(
  "cplus_behaviour",
  ``cplus @@ church m @@ church n -n->* church (m + n)``,
  SIMP_TAC (bsrw_ss()) [cplus_def, church_behaviour,
                        Cong betastar_funpow_cong] THEN
  SRW_TAC [][arithmeticTheory.FUNPOW_ADD, church_def]);

(* λn.λz.λs. n (λu. z) (λg.λh. h (g s))  (λu. u) *)
val cpred_def = Define`
  cpred =
    LAM "n"
     (LAM "z"
       (LAM "s"
          (VAR "n" @@ (LAM "u" (VAR "z")) @@
           (LAM "g" (LAM "h" (VAR "h" @@ (VAR "g" @@ VAR "s")))) @@
           (LAM "u" (VAR "u")))))
`;

val cpred_bnf = store_thm(
  "cpred_bnf",
  ``∀M. cpred -n-> M ⇔ F``,
  SRW_TAC [][cpred_def, normorder_rwts]);
val _ = export_rewrites ["cpred_bnf"]

val bnf_cpred = Store_thm(
  "bnf_cpred",
  ``bnf cpred``,
  SRW_TAC [][cpred_def]);

val FV_cpred = store_thm(
  "FV_cpred",
  ``FV cpred = {}``,
  SRW_TAC [][cpred_def, EXTENSION] THEN
  METIS_TAC []);
val _ = export_rewrites ["FV_cpred"]

val cpred_0 = store_thm(
  "cpred_0",
  ``cpred @@ church 0 -n->* church 0``,
  SIMP_TAC (bsrw_ss()) [church_def, cpred_def]);

val cpred_funpow = store_thm(
  "cpred_funpow",
  ``g ≠ h ∧ g ≠ s ∧ h ≠ s ∧ g ∉ FV f ∧ h ∉ FV f ⇒
      FUNPOW ((@@) (LAM g (LAM h (VAR h @@ (VAR g @@ VAR s)))))
             (SUC n)
             f
    -β->*
      LAM h (VAR h @@ FUNPOW ((@@) (VAR s)) n (f @@ VAR s))``,
  STRIP_TAC THEN unvarify_tac THEN Induct_on `n` THENL [
    ASM_SIMP_TAC (bsrw_ss()) [FUNPOW_SUC],

    CONV_TAC (LAND_CONV (ONCE_REWRITE_CONV [FUNPOW_SUC])) THEN
    ASM_SIMP_TAC (bsrw_ss()) [] THEN
    SRW_TAC [][FUNPOW_SUC]
  ]);

val cpred_beta_SUC = store_thm(
  "cpred_beta_SUC",
  ``cpred @@ church (SUC n) -β->* church n``,
  SIMP_TAC (bsrw_ss()) [cpred_def, church_behaviour, cpred_funpow]

  SRW_TAC [][cpred_def] THEN
  SRW_TAC [][Once relationTheory.RTC_CASES1, ccbeta_rwt] THEN
  CONV_TAC (RAND_CONV (ONCE_REWRITE_CONV [church_def])) THEN
  SRW_TAC [][] THEN
  ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
  Q.ABBREV_TAC `hgs = VAR "h" @@ (VAR "g" @@ VAR "s")` THEN
  Q.ABBREV_TAC `ID = LAM "u" (VAR "u")` THEN
  Q.ABBREV_TAC `Kz = LAM "u" (VAR "z")` THEN
  Q.EXISTS_TAC `FUNPOW ((@@) (LAM "g" (LAM "h" hgs))) (SUC n) Kz @@ ID` THEN
  CONJ_TAC THEN1
     SRW_TAC [][betastar_APPl, church_behaviour, nstar_betastar] THEN

  ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
  Q.EXISTS_TAC
    `(LAM "h"
       (VAR "h" @@ FUNPOW ((@@) (VAR "s")) n (Kz @@ VAR "s"))) @@ ID`  THEN
  CONJ_TAC THENL [
    MATCH_MP_TAC betastar_APPl THEN
    SRW_TAC [][Abbr`hgs`, cpred_funpow, Abbr`Kz`],

    SRW_TAC [DNF_ss]
            [Once relationTheory.RTC_CASES1, ccbeta_rwt, ccbeta_funpow_var,
             Abbr`ID`, Abbr`Kz`] THEN
    SRW_TAC [DNF_ss]
            [Once relationTheory.RTC_CASES1, ccbeta_rwt, ccbeta_funpow_var]
  ]);

val cpred_SUC = store_thm(
  "cpred_SUC",
  ``cpred @@ church (SUC n) -n->* church n``,
  METIS_TAC [bnf_church, normal_finds_bnf, cpred_beta_SUC]);

val cpred_behaviour = store_thm(
  "cpred_behaviour",
  ``cpred @@ church n -n->* church (PRE n)``,
  Cases_on `n` THEN SRW_TAC [][cpred_SUC, cpred_0]);

val cminus_def = Define`
  cminus = LAM "m" (LAM "n" (VAR "n" @@ VAR "m" @@ cpred))
`;

val cminus_behaviour = store_thm(
  "cminus_behaviour",
  ``cminus @@ church m @@ church n -n->* church (m - n)``,
  SRW_TAC [DNF_ss][cminus_def, Once relationTheory.RTC_CASES1, normorder_rwts,
                   lemma14b] THEN
  SRW_TAC [][Once relationTheory.RTC_CASES1, normorder_rwts, lemma14b] THEN
  ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
  Q.EXISTS_TAC `FUNPOW ((@@) cpred) n (church m)` THEN
  SRW_TAC [][church_behaviour] THEN
  SRW_TAC [][nstar_betastar_bnf] THEN
  Q.ID_SPEC_TAC `m` THEN Induct_on `n` THEN
  SRW_TAC [][FUNPOW_SUC] THEN
  ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
  Q.EXISTS_TAC `cpred @@ church (m - n)` THEN SRW_TAC [][betastar_APPr] THEN
  Q_TAC SUFF_TAC `m - SUC n = PRE (m - n)`
    THEN1 SRW_TAC [][cpred_behaviour, nstar_betastar] THEN
  DECIDE_TAC);

val cmult_def = Define`
  cmult = LAM "m" (LAM "n" (VAR "m" @@ church 0 @@ (cplus @@ VAR "n")))
`;

val church_to_bnf = store_thm(
  "church_to_bnf",
  ``bnf M ⇒ (church n @@ x @@ f -n->* M  ⇔ FUNPOW ((@@) f) n x -n->* M)``,
  SRW_TAC [][church_def] THEN FRESH_TAC THEN
  CONV_TAC (LAND_CONV
                (ONCE_REWRITE_CONV [relationTheory.RTC_CASES1])) THEN
  SRW_TAC [][normorder_rwts] THEN
  MATCH_MP_TAC (DECIDE ``¬p ∧ (q ⇔ r) ⇒ (p ∨ q ⇔ r)``) THEN CONJ_TAC THEN1
    (STRIP_TAC THEN SRW_TAC [][] THEN FULL_SIMP_TAC (srw_ss()) []) THEN
  CONV_TAC (LAND_CONV
                (ONCE_REWRITE_CONV [relationTheory.RTC_CASES1])) THEN
  SRW_TAC [][normorder_rwts, lemma14b] THEN
  MATCH_MP_TAC (DECIDE ``¬p ∧ (q ⇔ r) ⇒ (p ∨ q ⇔ r)``) THEN CONJ_TAC THEN1
    (STRIP_TAC THEN SRW_TAC [][] THEN FULL_SIMP_TAC (srw_ss()) []) THEN
  REWRITE_TAC []);
val _ = export_rewrites ["church_to_bnf"]


val FV_cplus = store_thm(
  "FV_cplus",
  ``FV cplus = {}``,
  SRW_TAC [][cplus_def, EXTENSION] THEN METIS_TAC []);
val _ = export_rewrites ["FV_cplus"]

val cmult_behaviour = store_thm(
  "cmult_behaviour",
  ``cmult @@ church m @@ church n -n->* church (m * n)``,
  SRW_TAC [][cmult_def, Once relationTheory.RTC_CASES1, normorder_rwts,
             lemma14b] THEN
  SRW_TAC [][Once relationTheory.RTC_CASES1, normorder_rwts, lemma14b] THEN
  Induct_on `m` THEN SRW_TAC [][FUNPOW_SUC] THEN
  SRW_TAC [][nstar_betastar_bnf] THEN
  ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
  Q.EXISTS_TAC `cplus @@ church n @@ church (m * n)` THEN
  SRW_TAC [][betastar_APPr, nstar_betastar, arithmeticTheory.MULT_CLAUSES] THEN
  METIS_TAC [nstar_betastar, cplus_behaviour, arithmeticTheory.ADD_COMM])

(* predicates/relations *)
val cis_zero_def = Define`
  cis_zero = LAM "n" (VAR "n" @@ cB T @@ (LAM "x" (cB F)))
`;
val FV_cis_zero = Store_thm(
  "FV_cis_zero",
  ``FV cis_zero = {}``,
  SRW_TAC [][cis_zero_def, EXTENSION]);
val bnf_cis_zero = Store_thm(
  "bnf_cis_zero",
  ``bnf cis_zero``,
  SRW_TAC [][cis_zero_def]);

val cis_zero_behaviour = store_thm(
  "cis_zero_behaviour",
  ``cis_zero @@ church n -n->* cB (n = 0)``,
  Cases_on `n = 0` THEN SRW_TAC [][cis_zero_def] THEN
  SRW_TAC [][Once relationTheory.RTC_CASES1, normorder_rwts, lemma14b] THEN
  `∃m. n = SUC m` by (Cases_on `n` THEN FULL_SIMP_TAC (srw_ss()) []) THEN
  SRW_TAC [][FUNPOW_SUC] THEN
  SRW_TAC [][Once relationTheory.RTC_CASES1, normorder_rwts, lemma14b])

val ceqnat_def = Define`
  ceqnat = LAM "n"
             (VAR "n" @@ cis_zero @@
                (LAM "r" (LAM "m" (cand @@ (cnot @@ (cis_zero @@ (VAR "m"))) @@
                                           (VAR "r" @@ (cpred @@ (VAR "m")))))))
`;
val FV_ceqnat = Store_thm(
  "FV_ceqnat",
  ``FV ceqnat = {}``,
  SRW_TAC [][ceqnat_def, EXTENSION] THEN METIS_TAC []);

val ceqnat_behaviour = store_thm(
  "ceqnat_behaviour",
  ``ceqnat @@ church n @@ church m -n->* cB (n = m)``,
  SRW_TAC [][ceqnat_def, Once relationTheory.RTC_CASES1, normorder_rwts,
             lemma14b] THEN DISJ2_TAC THEN
  Q.MATCH_ABBREV_TAC
      `church n @@ cis_zero @@ ff @@ church m -n->* cB(n = m)` THEN
  SRW_TAC [][nstar_betastar_bnf] THEN
  ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
  Q.EXISTS_TAC `FUNPOW ((@@) ff) n cis_zero @@ church m` THEN CONJ_TAC
    THEN1 SRW_TAC [][betastar_APPl, nstar_betastar, church_behaviour] THEN
  Q.ID_SPEC_TAC `m` THEN Induct_on `n` THENL [
    ONCE_REWRITE_TAC [EQ_SYM_EQ] THEN
    SRW_TAC [][nstar_betastar, cis_zero_behaviour],

    ALL_TAC
  ] THEN SRW_TAC [][FUNPOW_SUC] THEN
  SRW_TAC [][Abbr`ff`, Once relationTheory.RTC_CASES1, GSYM nstar_betastar_bnf,
             normorder_rwts, lemma14b] THEN DISJ2_TAC THEN
  SRW_TAC [][Once relationTheory.RTC_CASES1, normorder_rwts, lemma14b] THEN
  DISJ2_TAC THEN
  `(m = 0) ∨ ∃m0. m = SUC m0` by (Cases_on `m` THEN SRW_TAC [][]) THEN
  SRW_TAC [][] THENL [
    SRW_TAC [][nstar_betastar_bnf] THEN
    Q.MATCH_ABBREV_TAC
        `cand @@ (cnot @@ (cis_zero @@ church 0)) @@ X -β->* cB F` THEN
    ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
    Q.EXISTS_TAC `cand @@ cB F @@ X` THEN CONJ_TAC THENL [
      ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
      Q.EXISTS_TAC `cand @@ (cnot @@ cB T) @@ X` THEN CONJ_TAC THENL [
        MATCH_MP_TAC betastar_APPl THEN
        MATCH_MP_TAC betastar_APPr THEN
        MATCH_MP_TAC betastar_APPr THEN
        MATCH_MP_TAC nstar_betastar THEN
        ONCE_REWRITE_TAC [DECIDE ``T = (0 = 0)``] THEN
        SRW_TAC [][cis_zero_behaviour],

        MATCH_MP_TAC betastar_APPl THEN
        MATCH_MP_TAC betastar_APPr THEN
        MATCH_MP_TAC nstar_betastar THEN
        ONCE_REWRITE_TAC [DECIDE ``F = ¬T``] THEN
        SRW_TAC [][cnot_behaviour]
      ],

      SRW_TAC [][nstar_betastar, cand_F1]
    ],

    SRW_TAC [][nstar_betastar_bnf] THEN
    Q.MATCH_ABBREV_TAC `cand @@ Cond1 @@ Cond2 -β->* cB bool` THEN
    ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
    Q.EXISTS_TAC `cand @@ cB T @@ Cond2` THEN CONJ_TAC THENL [
      MATCH_MP_TAC betastar_APPl THEN
      MATCH_MP_TAC betastar_APPr THEN
      SRW_TAC [][Abbr`Cond1`] THEN
      Q.MATCH_ABBREV_TAC `cnot @@ Cond -β->* cB T` THEN
      ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
      Q.EXISTS_TAC `cnot @@ cB F` THEN CONJ_TAC THENL [
        SRW_TAC [][Abbr`Cond`] THEN
        MATCH_MP_TAC betastar_APPr THEN
        MATCH_MP_TAC nstar_betastar THEN
        MATCH_ACCEPT_TAC
            (SIMP_RULE (srw_ss()) [] (Q.INST [`n` |-> `SUC N`]
                                             cis_zero_behaviour)),

        MP_TAC (REWRITE_RULE [] (Q.INST [`p` |-> `F`] cnot_behaviour)) THEN
        SRW_TAC [][nstar_betastar]
      ],

      ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
      Q.EXISTS_TAC `Cond2` THEN
      SRW_TAC [][cand_T1, nstar_betastar] THEN
      UNABBREV_ALL_TAC THEN
      Q.MATCH_ABBREV_TAC `FF @@ Arg -β->* bool` THEN
      ONCE_REWRITE_TAC [relationTheory.RTC_CASES_RTC_TWICE] THEN
      Q.EXISTS_TAC `FF @@ church m0` THEN CONJ_TAC THEN1
        SRW_TAC [][betastar_APPr, cpred_beta_SUC, Abbr`Arg`] THEN
      SRW_TAC [][Abbr`bool`]
    ]
  ]);


val _ = export_theory()

