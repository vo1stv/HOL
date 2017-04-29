load "tautLib";
load "stringTheory";
load "numTheory";
load "ptopTheory";
load "imperativeLib";
load "imperativeTheory";
open tautLib ptopTheory imperativeLib imperativeTheory;

val _ = set_trace "Unicode" 0;

val _ = show_types:=true;
val _ = show_assums:=true;

val simpleTruthForgettableName1 = TAUT_PROVE (``!(a:bool) (b:bool). (  (~ (a       ==> b) ) <=> (   a  /\  ~ b  ) )``);
val simpleTruthForgettableName2 = TAUT_PROVE (``!(a:bool) (b:bool). (  (~ ( a       /\ b) ) <=> ( (~a) \/ (~ b) ) )``);

val simpleTruths = [
	simpleTruthForgettableName1,
	simpleTruthForgettableName2
];

val simpleTruthForgettableName3 = prove(
	``!(z:num)(z':num)(x:'a)(x':'a).(
		((if x = x' then ((if x = x' then 1 else 0) = z) else ((if x = x' then 1 else 0) = z'))
	<=>
		(if x = x' then (1 = z) else (0 = z'))))
	``,(METIS_TAC [])
);

val simpleTruthForgettableName4 = prove(
	``!(z:num) (z':num)(v:bool).( 
		((~v) ==> ( (if v then z else z' ) <> z) )
	<=> 
		( (~v) ==>(z' <> z ) ))
	``,(METIS_TAC [])
);

(*
val polymorphicTruth1 = prove (``!(a:bool) (b:bool). (  (  (?(s:'c).a)/\b  ) <=> (?s:'c.( a /\ b)) )``, (METIS_TAC []));
fun pmt1Applies tg = ( 
	if ( not (is_conj tg) ) then
		false
	else let val lr = (dest_conj(tg)) in (
		if ( not (is_exists (#1(lr) ) ) ) then 
			false
		else 
			true
	) end
);

fun applyPolymorphicTruth1 lhs rhs sType = ( 
	INST_TYPE [gamma |-> sType] (SPECL [lhs,rhs] polymorphicTruth1) 
);

fun pmt1tac sType = let val tg = (#2(top_goal())) in (
	if(pmt1Applies tg) then 
		(REWRITE_TAC [applyPolymorphicTruth1 (#2(dest_exists(#1(dest_conj tg)))) (#2(dest_conj tg)) sType])
	else
		ALL_TAC
	)
end;
*)

val goodImplementation = ``(sc (assign y (\ (s:'a->num).1)) (assign x (\ (s:'a->num).(s y ))))``; 
val badImplementation = ``(sc (assign y (\ (s:'a->num).(s x))) (assign x (\ (s:'a->num).1 )))``;

val lhsSpec = ``(\ (s:'a->num) (s':'a->num). (((s' (x:'a)) = 1 ) /\ ((s' (y:'a)) = 1)))``;

fun rhsProgLhsNotSpec rhsProg = mk_icomb(mk_icomb(REFINEMENT_NOT_RATOR,lhsSpec),rhsProg);

fun rhsProgLhsSpec rhsProg = mk_icomb(mk_icomb(REFINEMENT_RATOR,lhsSpec),rhsProg);

fun testIt asl b = if b then 
	set_goal([], mk_imp ( list_mk_conj(asl), (rhsProgLhsSpec goodImplementation ) ) ) 
else 
	set_goal([], mk_imp ( list_mk_conj(asl), (rhsProgLhsNotSpec badImplementation) ) ) 
;

testIt (DECL_STATEVARS ``v:'a`` [``x``,``y``,``t``]) true;
testIt (DECL_STATEVARS ``v:'a`` [``x``,``y``,``t``]) false;

val theTac=(
	(REPEAT STRIP_TAC)
THEN
	(REFINEMENT_TAC)
THEN
	(REWRITE_TAC simpleTruths)
THEN
	(REWRITE_TAC [FORWARD_SUB])
THEN
	(REP_EVAL_TAC)
);

e theTac;

val matchMe=dest_conj(#2(dest_exists(#2(dest_exists(#2(top_goal()))))));
val Px=(#2(dest_forall(#1(matchMe))));
val P=mk_abs(``v:'a``,Px);
val Q=(#2(matchMe));
 
e (REWRITE_TAC [BETA_RULE (SPECL [P,Q] LEFT_AND_FORALL_THM)]);

e ((EVAL_FOR_STATEVARS [``x:'a``,``y:'a``,``t:'a``]) THEN REP_EVAL_TAC);

val matchMeToo = (let val gl = strip_exists(#2(top_goal())) in 
	mk_abs((hd(#1(gl))),(mk_abs(hd(tl(#1(gl))),(#2(gl)))))
end);


e (REWRITE_TAC [BETA_RULE (SPEC matchMeToo (INST_TYPE [alpha |-> ``:'a->num``,beta |-> ``:'a->num``] SWAP_EXISTS_THM))]);

e  (EXISTS_TAC ``\(v:'a). ( if (x:'a)=v then 1 else ( ( \(v':'a).0) (v) ) )``);
e  (EXISTS_TAC ``\(v':'a).0``);

e (GEN_TAC THEN (REPEAT STRIP_TAC) THEN REP_EVAL_TAC);

e (REWRITE_TAC [SPECL [``1``,``0``] simpleTruthForgettableName3]);

e (CHANGED_TAC REP_EVAL_TAC);

e (UNDISCH_TAC ``  (x :'a) <> (y :'a)``);
e ((REWRITE_TAC [SPECL [``1``,``0``] simpleTruthForgettableName4]) THEN DISCH_TAC THEN EVAL_TAC);
