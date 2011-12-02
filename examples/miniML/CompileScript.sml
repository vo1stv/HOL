(* generated by Lem from compile.lem *)
open bossLib Theory Parse res_quanTheory
open finite_mapTheory listTheory pairTheory pred_setTheory
open set_relationTheory sortingTheory stringTheory wordsTheory

val _ = new_theory "Compile"

open MiniMLTheory

(*open MiniML*)

(* Remove all ML data constructors and replace them with untyped tuples with
 * numeric indices *)
(*val remove_ctors : (conN -> num) -> exp -> exp*)

 val pat_remove_ctors_defn = Hol_defn "pat_remove_ctors" `
 
(pat_remove_ctors cnmap (Pvar vn) = Pvar vn)
/\
(pat_remove_ctors cnmap (Plit l) = Plit l)
/\
(pat_remove_ctors cnmap (Pcon NONE ps) = 
  Pcon NONE (MAP (pat_remove_ctors cnmap) ps))
/\
(pat_remove_ctors cnmap (Pcon (SOME cn) ps) = 
  Pcon NONE ((Plit (Num (cnmap cn))) :: MAP (pat_remove_ctors cnmap) ps))`;

val _ = Defn.save_defn pat_remove_ctors_defn;

 val remove_ctors_defn = Hol_defn "remove_ctors" `

(remove_ctors cnmap (Raise err) = Raise err)
/\
(remove_ctors cnmap (Val v) = Val (v_remove_ctors cnmap v))
/\
(remove_ctors cnmap (Con NONE es) = Con NONE (MAP (remove_ctors cnmap) es))
/\
(remove_ctors cnmap (Con (SOME cn) es) = 
  Con NONE (Val (Lit (Num (cnmap cn))) :: MAP (remove_ctors cnmap) es))
/\
(remove_ctors cnmap (Var vn) = Var vn)
/\
(remove_ctors cnmap (Fun vn e) = Fun vn (remove_ctors cnmap e))
/\ 
(remove_ctors cnmap (App op e1 e2) = 
  App op (remove_ctors cnmap e1) (remove_ctors cnmap e2))
/\
(remove_ctors cnmap (Log op' e1 e2) = 
  Log op' (remove_ctors cnmap e1) (remove_ctors cnmap e2))
/\
(remove_ctors cnmap (If e1 e2 e3) = 
  If (remove_ctors cnmap e1) (remove_ctors cnmap e2) (remove_ctors cnmap e3))
/\
(remove_ctors cnmap (Mat e pes) = 
  Mat (remove_ctors cnmap e) (match_remove_ctors cnmap pes))
/\
(remove_ctors cnmap (Let vn e1 e2) = 
  Let vn (remove_ctors cnmap e1) (remove_ctors cnmap e2))
/\
(remove_ctors cnmap (Letrec funs e) =
  Letrec (funs_remove_ctors cnmap funs) (remove_ctors cnmap e))
/\
(remove_ctors cnmap (Proj e n) = Proj (remove_ctors cnmap e) n)
/\
(v_remove_ctors cnmap (Lit l) = Lit l)
/\
(v_remove_ctors cnmap (Conv NONE vs) = 
  Conv NONE (MAP (v_remove_ctors cnmap) vs))
/\ 
(v_remove_ctors cnmap (Closure envE vn e) =
  Closure (env_remove_ctors cnmap envE) vn (remove_ctors cnmap e))
/\
(v_remove_ctors cnmap (Recclosure envE funs vn) =
  Recclosure (env_remove_ctors cnmap envE) (funs_remove_ctors cnmap funs) vn)
/\
(env_remove_ctors cnmap [] = [])
/\
(env_remove_ctors cnmap ((vn,v)::env) = 
  ((vn, v_remove_ctors cnmap v)::env_remove_ctors cnmap env))
/\
(funs_remove_ctors cnmap [] = [])
/\
(funs_remove_ctors cnmap ((vn1,vn2,e)::funs) = 
  ((vn1,vn2,remove_ctors cnmap e)::funs_remove_ctors cnmap funs))
/\
(match_remove_ctors cnmap [] = [])
/\
(match_remove_ctors cnmap ((p,e)::pes) =
  (pat_remove_ctors cnmap p, remove_ctors cnmap e)::match_remove_ctors cnmap pes)`;

val _ = Defn.save_defn remove_ctors_defn; 
val _ = export_theory()
