(* tests for string and character parsing *)
open HolKernel Parse boolLib bossLib testutils
fun q (QUOTE s) = "Q\"" ^ String.toString s ^ "\""
  | q (ANTIQUOTE a) = "AQ"

fun printq [] = ""
  | printq [x] = q x
  | printq (x::xs) = q x ^ " " ^ printq xs

open stringSyntax
val testdata = [(`#"("`, fromMLchar #"("),
                (`"\n^`)"`, fromMLstring "\n`)"),
                (`"foo\
    \bar"`, fromMLstring "foobar"),
                (`"foo\n\
\bar"`, fromMLstring "foo\nbar"),
                (`[#"c"]`, listSyntax.mk_list ([fromMLchar #"c"], ``:char``))]

fun do_test (q,res) = let
  val l_s = StringCvt.padRight #" " 40 (printq q)
  val r_s = StringCvt.padLeft #" " 15 ("``" ^ term_to_string res ^ "``")
  val _ = tprint (l_s ^ " = " ^ r_s)
in
  if aconv (Term q) res then OK() else die "FAILED!"
end

val _ = app do_test testdata

val foo =
 Define
  `foo = [#"\n"; #" "; #"!"; #"\""; #"#";
          #"$"; #"%"; #"&"; #"'"; #"("; #")";
          #"*"; #"+"; #";"; #"-"; #"."; #"/";
          #"0"; #"1"; #"2"; #"3"; #"4"; #"5";
          #"6"; #"7"; #"8"; #"9"; #":"; #";";
          #"<"; #"="; #">"; #"?"; #"@"; #"A";
          #"B"; #"C"; #"D"; #"E"; #"F"; #"G";
          #"H"; #"I"; #"J"; #"K"; #"L"; #"M";
          #"N"; #"O"; #"P"; #"Q"; #"R"; #"S";
          #"T"; #"U"; #"V"; #"W"; #"X"; #"Y";
          #"Z"; #"["; #"\\"; #"]"; #"^^"; #"_";
          #"^`"; #"a"; #"b"; #"c"; #"d"; #"e";
          #"f"; #"g"; #"h"; #"i"; #"j"; #"k";
          #"l"; #"m"; #"n"; #"o"; #"p"; #"q";
          #"r"; #"s"; #"t"; #"u"; #"v"; #"w";
          #"x"; #"y"; #"z"; #"{"; #"|"; #"}";
          #"~"]`;

val bar = Define`
  bar = EXPLODE "\n !\"#$%&'()*+;-./0123456789:;<=>?@\
                \ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^^_^`\
                \abcdefghijklmnopqrstuvwxyz{|}~"
`

val testthm = prove(``foo = bar``, SRW_TAC [][foo,bar]);

(* ----------------------------------------------------------------------
    string_eq_conv
   ---------------------------------------------------------------------- *)

open boolSyntax
val sec_data = [(``"" = ""``, T),
                (``"" = "a"``, F),
                (``"a" = "b"``, F),
                (``"a" = "a"``, T),
                (``"abc" = "ab"``, F)]

fun sectest (t1, rest) = let
  val _ = tprint (StringCvt.padRight #" " 40 (term_to_string t1) ^ " = " ^
                  StringCvt.padLeft #" " 15 (term_to_string rest))
  val (actual, ok) = let
    val res = rhs (concl (stringLib.string_EQ_CONV t1))
  in
    (term_to_string res, aconv res rest)
  end handle _ => ("EXN", false)
in
  if ok then OK() else die ("FAILED!\n  Got "^actual)
end

val _ = app sectest sec_data

val _ = set_trace "Unicode" 0

val _ = app tpp ["P \"a\" /\\ Q",
                 "P (STRCAT a \"b\") /\\ Q",
                 "#\"a\"",
                 "\"(*\"",
                 "\"*)\""]

val _ = set_trace "paranoid string literal printing" 1

val t = ``"*)"``
val _ = tprint "Paranoid printing of ``\"*)\"``"
val s = term_to_string t
val _ = if s = "\"\\042)\"" then OK()
        else die "FAILED!"


val _ = OS.Process.exit OS.Process.success
