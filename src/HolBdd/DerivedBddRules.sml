
(*****************************************************************************)
(* DerivedBddRules.sml                                                       *)
(* -------------------                                                       *)
(*                                                                           *)
(* Some BDD utilities and derived rules using MuDDy and PrimitiveBddRules    *)
(* (builds on some of Ken Larsen's code                                      *)
(*****************************************************************************)
(*                                                                           *)
(* Revision history:                                                         *)
(*                                                                           *)
(*   Mon Oct  8 10:27:40 BST 2001 -- created file                            *)
(*                                                                           *)
(*****************************************************************************)

(*
load "muddyLib";
load "pairLib";
load "Pair_basic";
load "numLib";
load "PrimitiveBddRules";
load "HolBddTheory";

val _ = if not(bdd.isRunning()) then bdd.init 1000000 10000 else ();
*)

local

open pairSyntax;
open pairTools;
open Pair_basic;
open numLib;
open muddyLib;
open PrimitiveBddRules;
open bdd;
open Varmap;

open Globals HolKernel Parse boolLib bdd;
infixr 3 -->;
infix ## |-> THEN THENL THENC ORELSE ORELSEC THEN_TCL ORELSE_TCL;

fun hol_err msg func = 
 (print "DerivedBddRules: hol_err \""; print msg; 
  print "\" \""; print func; print "\"\n";
  raise mk_HOL_ERR "HolBdd" func msg);

in

(*****************************************************************************)
(* Test if a term is constructed using a BuDDY BDD binary operation (bddop)  *)
(*****************************************************************************)

(*****************************************************************************)
(* Destruct a term corresponding to a BuDDY BDD binary operation (bddop).    *)
(* Fail if not such a term.                                                  *)
(*****************************************************************************)

exception dest_BddOpError;

fun dest_BddOp tm =
 if is_neg tm
  then
   let val t = dest_neg tm
   in
    if is_conj t 
     then let val (t1,t2) = dest_conj t in (Nand, t1, t2) end else
    if is_disj t
     then let val (t1,t2) = dest_disj t in (Nor, t1, t2) end
     else raise dest_BddOpError
   end 
  else
   case strip_comb tm of
      (opr, [t1,t2]) => (case fst(dest_const opr) of
                            "/\\"  => if is_neg t1 
                                       then (Lessth, dest_neg t1, t2) else
                                      if is_neg t2
                                       then (Diff, t1, dest_neg t2)
                                       else (And, t1, t2)
                          | "\\/"  => (Or, t1, t2)
                          | "==>"  => (Imp, t1, t2)
                          | "="    => (Biimp, t1, t2)
                          | _      => raise dest_BddOpError)
    | _              => raise dest_BddOpError;

(*****************************************************************************)
(* Scan a term and construct a term_bdd using the primitive operations       *)
(* when applicable, and a supplied function otherwise                        *)
(*****************************************************************************)

local
fun fn3(f1,f2,f3)(x1,x2,x3) = (f1 x1, f2 x2, f3 x3)
in
fun GenTermToTermBdd leaffn vm tm =
 let fun recfn tm = 
  if tm = T 
   then BddT vm else
  if tm = F 
   then BddF vm else
  if is_var tm 
   then BddVar true vm tm else
  if is_neg tm andalso is_var(dest_neg tm) 
   then BddVar false vm tm else
  if is_cond tm 
   then (BddIte o fn3(recfn,recfn,recfn) o dest_cond) tm else
  if is_forall tm 
   then let val (vars,bdy) = strip_forall tm
        in
         (BddAppall vars o fn3(I,recfn,recfn) o dest_BddOp) bdy
          handle dest_BddOpError => (BddForall vars o recfn) bdy
        end else
  if is_exists tm 
   then let val (vars,bdy) = strip_exists tm
        in
         (BddAppex  vars o fn3(I,recfn,recfn) o dest_BddOp) bdy
         handle dest_BddOpError => (BddExists vars o recfn) bdy
        end
   else ((BddOp o fn3(I,recfn,recfn) o dest_BddOp) tm
         handle dest_BddOpError => leaffn tm)
 in
  recfn tm
 end
end;

(*****************************************************************************)
(* Extend a varmap with a list of variables                                  *)
(*****************************************************************************)

fun extendVarmap [] vm = vm
 |  extendVarmap (v::vl) vm =
     case Varmap.peek vm (name v) of
        SOME _ => extendVarmap vl vm
      | NONE   => let val n   = getVarnum()
                      val _   = bdd.setVarnum(n+1)
                  in 
                   extendVarmap vl (Varmap.insert (name v, n) vm)
                  end;

(*****************************************************************************)
(* Convert the BDD part of a term_bdd to a term                              *)
(*****************************************************************************)

exception bddToTermError;

fun bddToTerm varmap =
 let val pairs = Binarymap.listItems varmap
     fun get_node_name n =
      case assoc2 n pairs of
         SOME(str,_) => str
       | NONE        => (print("Node "^(Int.toString n)^" has no name");
                         raise bddToTermError)
     fun bddToTerm_aux bdd =
      if (bdd.equal bdd bdd.TRUE)
       then T
       else
        if (bdd.equal bdd bdd.FALSE)
         then F
         else Psyntax.mk_cond(mk_var(get_node_name(bdd.var bdd),bool),
                              bddToTerm_aux(bdd.high bdd),
                              bddToTerm_aux(bdd.low bdd))
 in
  bddToTerm_aux
 end;

(*****************************************************************************)
(* Global assignable varmap                                                  *)
(*****************************************************************************)

val global_varmap = ref(Varmap.empty);

fun showVarmap () = Binarymap.listItems(!global_varmap);

(*****************************************************************************)
(* Add variables to global_varmap and then call GenTermToTermBdd             *)
(* using the global function !termToTermBddFun on leaves                     *)
(*****************************************************************************)

exception termToTermBddError;

val termToTermBddFun = 
 ref(fn (tm:term) => (raise termToTermBddError):term_bdd);

fun termToTermBdd tm =
 let val vl = rev(all_vars tm)     (* all_vars returns vars in reverse order *)
     val vm = extendVarmap vl (!global_varmap)
     val _  = global_varmap := vm
 in
  GenTermToTermBdd (!termToTermBddFun) vm tm
 end;

(*****************************************************************************)
(* Iterate a function f : int -> term_bdd -> term_bdd from an initial        *)
(* term_bdd and applied successively to 0,1,2,... until a fixed point is     *)
(* reached. The fixedpoint is returned.                                      *)
(*                                                                           *)
(* The reference                                                             *)
(*                                                                           *)
(*  iterateReport : (int -> term_bdd -> unit) ref                            *)
(*                                                                           *)
(* contains a function that is applied to the current iteration level        *)
(* and term_bdd, and can be used to trace the iteration. The default         *)
(* just prints out the iteration num.                                        *)
(*                                                                           *)
(*****************************************************************************)

val iterateReport = 
 ref(fn n => fn (tb:term_bdd) => (print(Int.toString n); print " "));

fun iterateToFixedpoint f =
 let fun iter n tb =
      let val _    = (!iterateReport) n tb
          val tb'  = f n tb
      in
       if BddEqualTest tb tb' then tb else iter (n+1) tb'
      end
 in
  iter 0
 end;

(*****************************************************************************)
(* Flatten a varstruct term into a list of variables (also in StateEnum).    *)
(*****************************************************************************)

fun flatten_pair t =
if is_pair t
 then foldr (fn(t,l) => (flatten_pair t) @ l) [] (strip_pair t)
 else [t];

(*****************************************************************************)
(* MkIterThms ReachBy_rec``R((v1,...,vn),(v1',...,vn'))`` ``B(v1,...,vn)`` = *)
(*  ([|- ReachBy R B 0 (v1,...,vn) = B(v1,...,vn),                           *)
(*    |- !n. ReachBy R B (SUC n) (v1,...,vn) =                               *)
(*                ReachBy R B n (v1,...,vn)                                  *)
(*                \/                                                         *)
(*                ?v1'...vn'. ReachBy R B n (v1',...,vn')                    *)
(*                            /\                                             *)
(*                            R ((v1',...,vn'),(v1,...,vn))]                 *)
(*                                                                           *)
(*                                                                           *)
(* MkIterThms ReachIn_rec``R((v1,...,vn),(v1',...,vn'))`` ``B(v1,...,vn)`` = *)
(*  ([|- ReachIn R B 0 (v1,...,vn) = B(v1,...,vn),                           *)
(*    |- !n. ReachIn R B (SUC n) (v1,...,vn) =                               *)
(*                ?v1'...vn'. ReachIn R B n (v1',...,vn')                    *)
(*                            /\                                             *)
(*                            R ((v1',...,vn'),(v1,...,vn))]                 *)
(*****************************************************************************)

fun MkIterThms reachth Rtm Btm =
 let val (R,st_st') = dest_comb Rtm
     val (st,st') = dest_pair st_st'
     val (B,st0) = dest_comb Btm
     val _ = Term.aconv st st0 
             orelse hol_err "R and B vars not consistent" "MkReachByIterThms"
     val ty     = type_of st
     val th = INST_TYPE[(``:'a`` |-> ty),(``:'b`` |-> ty)]reachth
     val (th1,th2) = (CONJUNCT1 th, CONJUNCT2 th)
     val ntm = mk_var("n",num)
     val th3 = SPECL[R,B,st]th1
     val th4 = CONV_RULE 
                (RHS_CONV
                 (ONCE_DEPTH_CONV
                  (Ho_Rewrite.REWRITE_CONV[pairTheory.EXISTS_PROD]
                    THENC RENAME_VARS_CONV (map (fst o dest_var) (flatten_pair st')))))
                (SPECL[R,B,ntm,st]th2)

 in
  (th3, GEN ntm th4)
 end;

(*****************************************************************************)
(*  |- t1 = t2   vm t1' |--> b                                               *)
(*  -------------------------                                                *)
(*       vm t2' |--> b'                                                      *)
(*                                                                           *)
(* where t1 can be instantiated to t1' and t2' is the corresponding          *)
(* instance of t2                                                            *)
(*****************************************************************************)

exception BddApThmError;

fun BddApThm th tb =
 let val (vm,t1',b) = dest_term_bdd tb
 in
  BddEqMp (REWR_CONV th t1') tb 
   handle HOL_ERR _ => hol_err "REWR_CONV failed" "BddApthm"
 end;

(*****************************************************************************)
(*   vm t |--> b                                                             *)
(*  -------------                                                            *)
(*  vm tm |--> b'                                                            *)
(*                                                                           *)
(* where boolean variables in t can be renamed to get tm and b' is           *)
(* the corresponding replacement of BDD variables in b                       *)
(*****************************************************************************)

exception BddApReplaceError;

fun BddApReplace tb tm =
 let val (vm,t,b)  = dest_term_bdd tb
     val (tml,tyl) = match_term t tm
     val _         = if null tyl then () else raise BddApReplaceError
     val tbl       = (List.map 
                       (fn{redex=old,residue=new}=> 
                         (BddVar true vm old, BddVar true vm new))
                       tml 
                      handle BddVarError => raise BddApReplaceError)
 in
   BddReplace tbl tb
 end;

(*****************************************************************************)
(*     |- t1 = t2                                                            *)
(*   ---------------                                                         *)
(*     vm t1 |--> b                                                          *)
(*                                                                           *)
(* Fails if t2 is not built from variables using bddops                      *)
(*****************************************************************************)

fun eqToTermBdd leaffn vm th =
 let val th' = SPEC_ALL th
     val tm  = rhs(concl th')
 in
  BddEqMp (SYM th') (GenTermToTermBdd leaffn vm tm)
 end;

(*****************************************************************************)
(* Convert an ml positive integer to a HOL numeral                           *)
(*****************************************************************************)

fun intToTerm n = numSyntax.mk_numeral(Arbnum.fromInt n);

(*****************************************************************************)
(*  vm tm |--> b   conv tm = |= tm = tm'                                     *)
(*  ------------------------------------                                     *)
(*           vm tm' |--> b                                                   *)
(*****************************************************************************)

fun BddApConv conv tb = BddEqMp (conv(getTerm tb)) tb;

(*****************************************************************************)
(*   |- f (SUC n) s = ... f n ... s ...    |- f 0 s = ... s ...              *)
(*   ----------------------------------------------------------              *)
(*       |- f (SUC i) s = f i s       vm ``f i s`` |--> bi                   *)
(*                                                                           *)
(* where i is the first number such that |- f (SUC i) s = f i s              *)
(*****************************************************************************)

exception computeFixedpointError;

fun computeFixedpoint vm (thsuc, th0) =
 let val tb0 =  eqToTermBdd (fn tm => raise computeFixedpointError) vm th0
     fun f n tb =  
      BddApConv
       computeLib.EVAL_CONV
       (eqToTermBdd (BddApReplace tb) vm (SPEC (intToTerm n) thsuc))
 in
  iterateToFixedpoint f tb0
 end;

end;
