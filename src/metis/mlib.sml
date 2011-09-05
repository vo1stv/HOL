(* ========================================================================= *)
(* ML COMPILER SPECIFIC FUNCTIONS                                            *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibPortable =
sig

(* ------------------------------------------------------------------------- *)
(* The ML implementation.                                                    *)
(* ------------------------------------------------------------------------- *)

val ml : string

(* ------------------------------------------------------------------------- *)
(* Pointer equality using the run-time system.                               *)
(* ------------------------------------------------------------------------- *)

val pointerEqual : 'a * 'a -> bool

(* ------------------------------------------------------------------------- *)
(* Marking critical sections of code.                                        *)
(* ------------------------------------------------------------------------- *)

val critical : (unit -> 'a) -> unit -> 'a

(* ------------------------------------------------------------------------- *)
(* Generating random values.                                                 *)
(* ------------------------------------------------------------------------- *)

val randomBool : unit -> bool

val randomInt : int -> int  (* n |-> [0,n) *)

val randomReal : unit -> real  (* () |-> [0,1] *)

val randomWord : unit -> Word.word

(* ------------------------------------------------------------------------- *)
(* Timing function applications.                                             *)
(* ------------------------------------------------------------------------- *)

val time : ('a -> 'b) -> 'a -> 'b

end
structure mlibPortable =
struct
  fun critical f () = f ();
  val pointerEqual = Portable.pointer_eq
  fun randomBool _ = raise (Fail "Unimplemented: randomBool")
  fun randomInt  _ = raise (Fail "Unimplemented: randomInt")
  fun randomReal _ = raise (Fail "Unimplemented: randomReal")
  fun randomWord _ = raise (Fail "Unimplemented: randomWord")
  fun time       _ = raise (Fail "Unimplemented: randomWord")
end
(* ========================================================================= *)
(* ML UTILITY FUNCTIONS                                                      *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibUseful =
sig

(* ------------------------------------------------------------------------- *)
(* Exceptions.                                                               *)
(* ------------------------------------------------------------------------- *)

exception Error of string

exception Bug of string

val total : ('a -> 'b) -> 'a -> 'b option

val can : ('a -> 'b) -> 'a -> bool
val assert : bool -> exn -> unit

(* ------------------------------------------------------------------------- *)
(* Tracing.                                                                  *)
(* ------------------------------------------------------------------------- *)

val tracePrint : (string -> unit) ref

val trace : string -> unit

(* ------------------------------------------------------------------------- *)
(* Combinators.                                                              *)
(* ------------------------------------------------------------------------- *)

val C : ('a -> 'b -> 'c) -> 'b -> 'a -> 'c

val I : 'a -> 'a

val K : 'a -> 'b -> 'a

val S : ('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c

val W : ('a -> 'a -> 'b) -> 'a -> 'b

val funpow : int -> ('a -> 'a) -> 'a -> 'a

val exp : ('a * 'a -> 'a) -> 'a -> int -> 'a -> 'a

(* ------------------------------------------------------------------------- *)
(* Pairs.                                                                    *)
(* ------------------------------------------------------------------------- *)

val fst : 'a * 'b -> 'a

val snd : 'a * 'b -> 'b

val pair : 'a -> 'b -> 'a * 'b

val swap : 'a * 'b -> 'b * 'a

val curry : ('a * 'b -> 'c) -> 'a -> 'b -> 'c

val uncurry : ('a -> 'b -> 'c) -> 'a * 'b -> 'c

val ## : ('a -> 'c) * ('b -> 'd) -> 'a * 'b -> 'c * 'd

(* ------------------------------------------------------------------------- *)
(* State transformers.                                                       *)
(* ------------------------------------------------------------------------- *)

val unit : 'a -> 's -> 'a * 's

val bind : ('s -> 'a * 's) -> ('a -> 's -> 'b * 's) -> 's -> 'b * 's

val mmap : ('a -> 'b) -> ('s -> 'a * 's) -> 's -> 'b * 's

val mjoin : ('s -> ('s -> 'a * 's) * 's) -> 's -> 'a * 's

val mwhile : ('a -> bool) -> ('a -> 's -> 'a * 's) -> 'a -> 's -> 'a * 's

(* ------------------------------------------------------------------------- *)
(* Equality.                                                                 *)
(* ------------------------------------------------------------------------- *)

val equal : ''a -> ''a -> bool

val notEqual : ''a -> ''a -> bool

val listEqual : ('a -> 'a -> bool) -> 'a list -> 'a list -> bool

(* ------------------------------------------------------------------------- *)
(* Comparisons.                                                              *)
(* ------------------------------------------------------------------------- *)

val mapCompare : ('a -> 'b) -> ('b * 'b -> order) -> 'a * 'a -> order

val revCompare : ('a * 'a -> order) -> 'a * 'a -> order

val prodCompare :
    ('a * 'a -> order) -> ('b * 'b -> order) -> ('a * 'b) * ('a * 'b) -> order

val lexCompare : ('a * 'a -> order) -> 'a list * 'a list -> order

val optionCompare : ('a * 'a -> order) -> 'a option * 'a option -> order

val boolCompare : bool * bool -> order  (* false < true *)

(* ------------------------------------------------------------------------- *)
(* Lists: note we count elements from 0.                                     *)
(* ------------------------------------------------------------------------- *)

val cons : 'a -> 'a list -> 'a list

val hdTl : 'a list -> 'a * 'a list

val append : 'a list -> 'a list -> 'a list

val singleton : 'a -> 'a list

val first : ('a -> 'b option) -> 'a list -> 'b option

val maps : ('a -> 's -> 'b * 's) -> 'a list -> 's -> 'b list * 's

val mapsPartial : ('a -> 's -> 'b option * 's) -> 'a list -> 's -> 'b list * 's

val zipWith : ('a -> 'b -> 'c) -> 'a list -> 'b list -> 'c list

val zip : 'a list -> 'b list -> ('a * 'b) list

val unzip : ('a * 'b) list -> 'a list * 'b list

val cartwith : ('a -> 'b -> 'c) -> 'a list -> 'b list -> 'c list

val cart : 'a list -> 'b list -> ('a * 'b) list

val takeWhile : ('a -> bool) -> 'a list -> 'a list

val dropWhile : ('a -> bool) -> 'a list -> 'a list

val divideWhile : ('a -> bool) -> 'a list -> 'a list * 'a list

val groups : ('a * 's -> bool * 's) -> 's -> 'a list -> 'a list list

val groupsBy : ('a * 'a -> bool) -> 'a list -> 'a list list

val groupsByFst : (''a * 'b) list -> (''a * 'b list) list

val groupsOf : int -> 'a list -> 'a list list

val index : ('a -> bool) -> 'a list -> int option

val enumerate : 'a list -> (int * 'a) list

val divide : 'a list -> int -> 'a list * 'a list  (* Subscript *)

val revDivide : 'a list -> int -> 'a list * 'a list  (* Subscript *)

val updateNth : int * 'a -> 'a list -> 'a list  (* Subscript *)

val deleteNth : int -> 'a list -> 'a list  (* Subscript *)

(* ------------------------------------------------------------------------- *)
(* Sets implemented with lists.                                              *)
(* ------------------------------------------------------------------------- *)

val mem : ''a -> ''a list -> bool

val insert : ''a -> ''a list -> ''a list

val delete : ''a -> ''a list -> ''a list

val setify : ''a list -> ''a list  (* removes duplicates *)

val union : ''a list -> ''a list -> ''a list

val intersect : ''a list -> ''a list -> ''a list

val difference : ''a list -> ''a list -> ''a list

val subset : ''a list -> ''a list -> bool

val distinct : ''a list -> bool

(* ------------------------------------------------------------------------- *)
(* Sorting and searching.                                                    *)
(* ------------------------------------------------------------------------- *)

val minimum : ('a * 'a -> order) -> 'a list -> 'a * 'a list  (* Empty *)

val maximum : ('a * 'a -> order) -> 'a list -> 'a * 'a list  (* Empty *)

val merge : ('a * 'a -> order) -> 'a list -> 'a list -> 'a list

val sort : ('a * 'a -> order) -> 'a list -> 'a list

val sortMap : ('a -> 'b) -> ('b * 'b -> order) -> 'a list -> 'a list

(* ------------------------------------------------------------------------- *)
(* Integers.                                                                 *)
(* ------------------------------------------------------------------------- *)

val interval : int -> int -> int list

val divides : int -> int -> bool

val gcd : int -> int -> int

val primes : int -> int list

val primesUpTo : int -> int list

(* ------------------------------------------------------------------------- *)
(* Strings.                                                                  *)
(* ------------------------------------------------------------------------- *)

val rot : int -> char -> char

val charToInt : char -> int option

val charFromInt : int -> char option

val nChars : char -> int -> string

val chomp : string -> string

val trim : string -> string

val join : string -> string list -> string

val split : string -> string -> string list

val capitalize : string -> string

val mkPrefix : string -> string -> string

val destPrefix : string -> string -> string

val isPrefix : string -> string -> bool

val stripPrefix : (char -> bool) -> string -> string

val mkSuffix : string -> string -> string

val destSuffix : string -> string -> string

val isSuffix : string -> string -> bool

val stripSuffix : (char -> bool) -> string -> string

(* ------------------------------------------------------------------------- *)
(* Tables.                                                                   *)
(* ------------------------------------------------------------------------- *)

type columnAlignment = {leftAlign : bool, padChar : char}

val alignColumn : columnAlignment -> string list -> string list -> string list

val alignTable : columnAlignment list -> string list list -> string list

(* ------------------------------------------------------------------------- *)
(* Reals.                                                                    *)
(* ------------------------------------------------------------------------- *)

val percentToString : real -> string

val pos : real -> real

val log2 : real -> real  (* Domain *)

(* ------------------------------------------------------------------------- *)
(* Sum datatype.                                                             *)
(* ------------------------------------------------------------------------- *)

datatype ('a,'b) sum = Left of 'a | Right of 'b

val destLeft : ('a,'b) sum -> 'a

val isLeft : ('a,'b) sum -> bool

val destRight : ('a,'b) sum -> 'b

val isRight : ('a,'b) sum -> bool

(* ------------------------------------------------------------------------- *)
(* mlibUseful impure features.                                                   *)
(* ------------------------------------------------------------------------- *)

val newInt : unit -> int

val newInts : int -> int list

val withRef : 'r ref * 'r -> ('a -> 'b) -> 'a -> 'b

val cloneArray : 'a Array.array -> 'a Array.array

(* ------------------------------------------------------------------------- *)
(* The environment.                                                          *)
(* ------------------------------------------------------------------------- *)

val host : unit -> string

val time : unit -> string

val date : unit -> string

val readDirectory : {directory : string} -> {filename : string} list

val readTextFile : {filename : string} -> string

val writeTextFile : {contents : string, filename : string} -> unit

(* ------------------------------------------------------------------------- *)
(* Profiling and error reporting.                                            *)
(* ------------------------------------------------------------------------- *)

val try : ('a -> 'b) -> 'a -> 'b

val chat : string -> unit  (* stdout *)

val chide : string -> unit  (* stderr *)

val warn : string -> unit

val die : string -> 'exit

val timed : ('a -> 'b) -> 'a -> real * 'b

val timedMany : ('a -> 'b) -> 'a -> real * 'b

val executionTime : unit -> real  (* Wall clock execution time *)

end
(* ========================================================================= *)
(* ML UTILITY FUNCTIONS                                                      *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibUseful :> mlibUseful =
struct

(* ------------------------------------------------------------------------- *)
(* Exceptions.                                                               *)
(* ------------------------------------------------------------------------- *)

fun assert b e = if b then () else raise e

exception Error of string;

exception Bug of string;

fun errorToStringOption err =
    case err of
      Error message => SOME ("Error: " ^ message)
    | _ => NONE;

(*mlton
val () = MLton.Exn.addExnMessager errorToStringOption;
*)

fun errorToString err =
    case errorToStringOption err of
      SOME s => "\n" ^ s ^ "\n"
    | NONE => raise Bug "errorToString: not an Error exception";

fun bugToStringOption err =
    case err of
      Bug message => SOME ("Bug: " ^ message)
    | _ => NONE;

(*mlton
val () = MLton.Exn.addExnMessager bugToStringOption;
*)

fun bugToString err =
    case bugToStringOption err of
      SOME s => "\n" ^ s ^ "\n"
    | NONE => raise Bug "bugToString: not a Bug exception";

fun total f x = SOME (f x) handle Error _ => NONE;

fun can f = Option.isSome o total f;

(* ------------------------------------------------------------------------- *)
(* Tracing.                                                                  *)
(* ------------------------------------------------------------------------- *)

local
  val traceOut = TextIO.stdOut;

  fun tracePrintFn mesg =
      let
        val () = TextIO.output (traceOut,mesg)

        val () = TextIO.flushOut traceOut
      in
        ()
      end;
in
  val tracePrint = ref tracePrintFn;
end;

fun trace mesg = !tracePrint mesg;

(* ------------------------------------------------------------------------- *)
(* Combinators.                                                              *)
(* ------------------------------------------------------------------------- *)

fun C f x y = f y x;

fun I x = x;

fun K x y = x;

fun S f g x = f x (g x);

fun W f x = f x x;

fun funpow 0 _ x = x
  | funpow n f x = funpow (n - 1) f (f x);

fun exp m =
    let
      fun f _ 0 z = z
        | f x y z = f (m (x,x)) (y div 2) (if y mod 2 = 0 then z else m (z,x))
    in
      f
    end;

(* ------------------------------------------------------------------------- *)
(* Pairs.                                                                    *)
(* ------------------------------------------------------------------------- *)

fun fst (x,_) = x;

fun snd (_,y) = y;

fun pair x y = (x,y);

fun swap (x,y) = (y,x);

fun curry f x y = f (x,y);

fun uncurry f (x,y) = f x y;

val op## = fn (f,g) => fn (x,y) => (f x, g y);

(* ------------------------------------------------------------------------- *)
(* State transformers.                                                       *)
(* ------------------------------------------------------------------------- *)

val unit : 'a -> 's -> 'a * 's = pair;

fun bind f (g : 'a -> 's -> 'b * 's) = uncurry g o f;

fun mmap f (m : 's -> 'a * 's) = bind m (unit o f);

fun mjoin (f : 's -> ('s -> 'a * 's) * 's) = bind f I;

fun mwhile c b = let fun f a = if c a then bind (b a) f else unit a in f end;

(* ------------------------------------------------------------------------- *)
(* Equality.                                                                 *)
(* ------------------------------------------------------------------------- *)

val equal = fn x => fn y => x = y;

val notEqual = fn x => fn y => x <> y;

fun listEqual xEq =
    let
      fun xsEq [] [] = true
        | xsEq (x1 :: xs1) (x2 :: xs2) = xEq x1 x2 andalso xsEq xs1 xs2
        | xsEq _ _ = false
    in
      xsEq
    end;

(* ------------------------------------------------------------------------- *)
(* Comparisons.                                                              *)
(* ------------------------------------------------------------------------- *)

fun mapCompare f cmp (a,b) = cmp (f a, f b);

fun revCompare cmp x_y =
    case cmp x_y of LESS => GREATER | EQUAL => EQUAL | GREATER => LESS;

fun prodCompare xCmp yCmp ((x1,y1),(x2,y2)) =
    case xCmp (x1,x2) of
      LESS => LESS
    | EQUAL => yCmp (y1,y2)
    | GREATER => GREATER;

fun lexCompare cmp =
    let
      fun lex ([],[]) = EQUAL
        | lex ([], _ :: _) = LESS
        | lex (_ :: _, []) = GREATER
        | lex (x :: xs, y :: ys) =
          case cmp (x,y) of
            LESS => LESS
          | EQUAL => lex (xs,ys)
          | GREATER => GREATER
    in
      lex
    end;

fun optionCompare _ (NONE,NONE) = EQUAL
  | optionCompare _ (NONE,_) = LESS
  | optionCompare _ (_,NONE) = GREATER
  | optionCompare cmp (SOME x, SOME y) = cmp (x,y);

fun boolCompare (false,true) = LESS
  | boolCompare (true,false) = GREATER
  | boolCompare _ = EQUAL;

(* ------------------------------------------------------------------------- *)
(* Lists.                                                                    *)
(* ------------------------------------------------------------------------- *)

fun cons x y = x :: y;

fun hdTl l = (hd l, tl l);

fun append xs ys = xs @ ys;

fun singleton a = [a];

fun first f [] = NONE
  | first f (x :: xs) = (case f x of NONE => first f xs | s => s);

fun maps (_ : 'a -> 's -> 'b * 's) [] = unit []
  | maps f (x :: xs) =
    bind (f x) (fn y => bind (maps f xs) (fn ys => unit (y :: ys)));

fun mapsPartial (_ : 'a -> 's -> 'b option * 's) [] = unit []
  | mapsPartial f (x :: xs) =
    bind
      (f x)
      (fn yo =>
          bind
            (mapsPartial f xs)
            (fn ys => unit (case yo of NONE => ys | SOME y => y :: ys)));

fun zipWith f =
    let
      fun z l [] [] = l
        | z l (x :: xs) (y :: ys) = z (f x y :: l) xs ys
        | z _ _ _ = raise Error "zipWith: lists different lengths";
    in
      fn xs => fn ys => List.rev (z [] xs ys)
    end;

fun zip xs ys = zipWith pair xs ys;

local
  fun inc ((x,y),(xs,ys)) = (x :: xs, y :: ys);
in
  fun unzip ab = List.foldl inc ([],[]) (List.rev ab);
end;

fun cartwith f =
    let
      fun aux _ res _ [] = res
        | aux xsCopy res [] (y :: yt) = aux xsCopy res xsCopy yt
        | aux xsCopy res (x :: xt) (ys as y :: _) =
          aux xsCopy (f x y :: res) xt ys
    in
      fn xs => fn ys =>
         let val xs' = List.rev xs in aux xs' [] xs' (List.rev ys) end
    end;

fun cart xs ys = cartwith pair xs ys;

fun takeWhile p =
    let
      fun f acc [] = List.rev acc
        | f acc (x :: xs) = if p x then f (x :: acc) xs else List.rev acc
    in
      f []
    end;

fun dropWhile p =
    let
      fun f [] = []
        | f (l as x :: xs) = if p x then f xs else l
    in
      f
    end;

fun divideWhile p =
    let
      fun f acc [] = (List.rev acc, [])
        | f acc (l as x :: xs) = if p x then f (x :: acc) xs else (List.rev acc, l)
    in
      f []
    end;

fun groups f =
    let
      fun group acc row x l =
          case l of
            [] =>
            let
              val acc = if List.null row then acc else List.rev row :: acc
            in
              List.rev acc
            end
          | h :: t =>
            let
              val (eor,x) = f (h,x)
            in
              if eor then group (List.rev row :: acc) [h] x t
              else group acc (h :: row) x t
            end
    in
      group [] []
    end;

fun groupsBy eq =
    let
      fun f (x_y as (x,_)) = (not (eq x_y), x)
    in
      fn [] => []
       | h :: t =>
         case groups f h t of
           [] => [[h]]
         | hs :: ts => (h :: hs) :: ts
    end;

local
  fun fstEq ((x,_),(y,_)) = x = y;

  fun collapse l = (fst (hd l), List.map snd l);
in
  fun groupsByFst l = List.map collapse (groupsBy fstEq l);
end;

fun groupsOf n =
    let
      fun f (_,i) = if i = 1 then (true,n) else (false, i - 1)
    in
      groups f (n + 1)
    end;

fun index p =
  let
    fun idx _ [] = NONE
      | idx n (x :: xs) = if p x then SOME n else idx (n + 1) xs
  in
    idx 0
  end;

fun enumerate l = fst (maps (fn x => fn m => ((m, x), m + 1)) l 0);

local
  fun revDiv acc l 0 = (acc,l)
    | revDiv _ [] _ = raise Subscript
    | revDiv acc (h :: t) n = revDiv (h :: acc) t (n - 1);
in
  fun revDivide l = revDiv [] l;
end;

fun divide l n = let val (a,b) = revDivide l n in (List.rev a, b) end;

fun updateNth (n,x) l =
    let
      val (a,b) = revDivide l n
    in
      case b of [] => raise Subscript | _ :: t => List.revAppend (a, x :: t)
    end;

fun deleteNth n l =
    let
      val (a,b) = revDivide l n
    in
      case b of [] => raise Subscript | _ :: t => List.revAppend (a,t)
    end;

(* ------------------------------------------------------------------------- *)
(* Sets implemented with lists.                                              *)
(* ------------------------------------------------------------------------- *)

fun mem x = List.exists (equal x);

fun insert x s = if mem x s then s else x :: s;

fun delete x s = List.filter (not o equal x) s;

local
  fun inc (v,x) = if mem v x then x else v :: x;
in
  fun setify s = List.rev (List.foldl inc [] s);
end;

fun union s t =
    let
      fun inc (v,x) = if mem v t then x else v :: x
    in
      List.foldl inc t (List.rev s)
    end;

fun intersect s t =
    let
      fun inc (v,x) = if mem v t then v :: x else x
    in
      List.foldl inc [] (List.rev s)
    end;

fun difference s t =
    let
      fun inc (v,x) = if mem v t then x else v :: x
    in
      List.foldl inc [] (List.rev s)
    end;

fun subset s t = List.all (fn x => mem x t) s;

fun distinct [] = true
  | distinct (x :: rest) = not (mem x rest) andalso distinct rest;

(* ------------------------------------------------------------------------- *)
(* Sorting and searching.                                                    *)
(* ------------------------------------------------------------------------- *)

(* Finding the minimum and maximum element of a list, wrt some order. *)

fun minimum cmp =
    let
      fun min (l,m,r) _ [] = (m, List.revAppend (l,r))
        | min (best as (_,m,_)) l (x :: r) =
          min (case cmp (x,m) of LESS => (l,x,r) | _ => best) (x :: l) r
    in
      fn [] => raise Empty
       | h :: t => min ([],h,t) [h] t
    end;

fun maximum cmp = minimum (revCompare cmp);

(* Merge (for the following merge-sort, but generally useful too). *)

fun merge cmp =
    let
      fun mrg acc [] ys = List.revAppend (acc,ys)
        | mrg acc xs [] = List.revAppend (acc,xs)
        | mrg acc (xs as x :: xt) (ys as y :: yt) =
          (case cmp (x,y) of
             GREATER => mrg (y :: acc) xs yt
           | _ => mrg (x :: acc) xt ys)
    in
      mrg []
    end;

(* Merge sort (stable). *)

fun sort cmp =
    let
      fun findRuns acc r rs [] = List.rev (List.rev (r :: rs) :: acc)
        | findRuns acc r rs (x :: xs) =
          case cmp (r,x) of
            GREATER => findRuns (List.rev (r :: rs) :: acc) x [] xs
          | _ => findRuns acc x (r :: rs) xs

      fun mergeAdj acc [] = List.rev acc
        | mergeAdj acc (xs as [_]) = List.revAppend (acc,xs)
        | mergeAdj acc (x :: y :: xs) = mergeAdj (merge cmp x y :: acc) xs

      fun mergePairs [xs] = xs
        | mergePairs l = mergePairs (mergeAdj [] l)
    in
      fn [] => []
       | l as [_] => l
       | h :: t => mergePairs (findRuns [] h [] t)
    end;

fun sortMap _ _ [] = []
  | sortMap _ _ (l as [_]) = l
  | sortMap f cmp xs =
    let
      fun ncmp ((m,_),(n,_)) = cmp (m,n)
      val nxs = List.map (fn x => (f x, x)) xs
      val nys = sort ncmp nxs
    in
      List.map snd nys
    end;

(* ------------------------------------------------------------------------- *)
(* Integers.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun interval m 0 = []
  | interval m len = m :: interval (m + 1) (len - 1);

fun divides _ 0 = true
  | divides 0 _ = false
  | divides a b = b mod (Int.abs a) = 0;

local
  fun hcf 0 n = n
    | hcf 1 _ = 1
    | hcf m n = hcf (n mod m) m;
in
  fun gcd m n =
      let
        val m = Int.abs m
        and n = Int.abs n
      in
        if m < n then hcf m n else hcf n m
      end;
end;

local
  fun calcPrimes mode ps i n =
      if n = 0 then ps
      else if List.exists (fn p => divides p i) ps then
        let
          val i = i + 1
          and n = if mode then n else n - 1
        in
          calcPrimes mode ps i n
        end
      else
        let
          val ps = ps @ [i]
          and i = i + 1
          and n = n - 1
        in
          calcPrimes mode ps i n
        end;
in
  fun primes n =
      if n <= 0 then []
      else calcPrimes true [2] 3 (n - 1);

  fun primesUpTo n =
      if n < 2 then []
      else calcPrimes false [2] 3 (n - 2);
end;

(* ------------------------------------------------------------------------- *)
(* Strings.                                                                  *)
(* ------------------------------------------------------------------------- *)

local
  fun len l = (length l, l)

  val upper = len (String.explode "ABCDEFGHIJKLMNOPQRSTUVWXYZ");

  val lower = len (String.explode "abcdefghijklmnopqrstuvwxyz");

  fun rotate (n,l) c k =
      List.nth (l, (k + Option.valOf (index (equal c) l)) mod n);
in
  fun rot k c =
      if Char.isLower c then rotate lower c k
      else if Char.isUpper c then rotate upper c k
      else c;
end;

fun charToInt #"0" = SOME 0
  | charToInt #"1" = SOME 1
  | charToInt #"2" = SOME 2
  | charToInt #"3" = SOME 3
  | charToInt #"4" = SOME 4
  | charToInt #"5" = SOME 5
  | charToInt #"6" = SOME 6
  | charToInt #"7" = SOME 7
  | charToInt #"8" = SOME 8
  | charToInt #"9" = SOME 9
  | charToInt _ = NONE;

fun charFromInt 0 = SOME #"0"
  | charFromInt 1 = SOME #"1"
  | charFromInt 2 = SOME #"2"
  | charFromInt 3 = SOME #"3"
  | charFromInt 4 = SOME #"4"
  | charFromInt 5 = SOME #"5"
  | charFromInt 6 = SOME #"6"
  | charFromInt 7 = SOME #"7"
  | charFromInt 8 = SOME #"8"
  | charFromInt 9 = SOME #"9"
  | charFromInt _ = NONE;

fun nChars x =
    let
      fun dup 0 l = l | dup n l = dup (n - 1) (x :: l)
    in
      fn n => String.implode (dup n [])
    end;

fun chomp s =
    let
      val n = size s
    in
      if n = 0 orelse String.sub (s, n - 1) <> #"\n" then s
      else String.substring (s, 0, n - 1)
    end;

local
  fun chop l =
      case l of
        [] => []
      | h :: t => if Char.isSpace h then chop t else l;
in
  val trim = String.implode o chop o List.rev o chop o List.rev o String.explode;
end;

val join = String.concatWith;

local
  fun match [] l = SOME l
    | match _ [] = NONE
    | match (x :: xs) (y :: ys) = if x = y then match xs ys else NONE;

  fun stringify acc [] = acc
    | stringify acc (h :: t) = stringify (String.implode h :: acc) t;
in
  fun split sep =
      let
        val pat = String.explode sep

        fun div1 prev recent [] = stringify [] (List.rev recent :: prev)
          | div1 prev recent (l as h :: t) =
            case match pat l of
              NONE => div1 prev (h :: recent) t
            | SOME rest => div1 (List.rev recent :: prev) [] rest
      in
        fn s => div1 [] [] (String.explode s)
      end;
end;

fun capitalize s =
    if s = "" then s
    else str (Char.toUpper (String.sub (s,0))) ^ String.extract (s,1,NONE);

fun mkPrefix p s = p ^ s;

fun destPrefix p =
    let
      fun check s =
          if String.isPrefix p s then ()
          else raise Error "destPrefix"

      val sizeP = size p
    in
      fn s =>
         let
           val () = check s
         in
           String.extract (s,sizeP,NONE)
         end
    end;

fun isPrefix p = can (destPrefix p);

fun stripPrefix pred s =
    Substring.string (Substring.dropl pred (Substring.full s));

fun mkSuffix p s = s ^ p;

fun destSuffix p =
    let
      fun check s =
          if String.isSuffix p s then ()
          else raise Error "destSuffix"

      val sizeP = size p
    in
      fn s =>
         let
           val () = check s

           val sizeS = size s
         in
           String.substring (s, 0, sizeS - sizeP)
         end
    end;

fun isSuffix p = can (destSuffix p);

fun stripSuffix pred s =
    Substring.string (Substring.dropr pred (Substring.full s));

(* ------------------------------------------------------------------------- *)
(* Tables.                                                                   *)
(* ------------------------------------------------------------------------- *)

type columnAlignment = {leftAlign : bool, padChar : char}

fun alignColumn {leftAlign,padChar} column =
    let
      val (n,_) = maximum Int.compare (List.map size column)

      fun pad entry row =
          let
            val padding = nChars padChar (n - size entry)
          in
            if leftAlign then entry ^ padding ^ row
            else padding ^ entry ^ row
          end
    in
      zipWith pad column
    end;

local
  fun alignTab aligns rows =
      case aligns of
        [] => List.map (K "") rows
      | [{leftAlign = true, padChar = #" "}] => List.map hd rows
      | align :: aligns =>
        let
          val col = List.map hd rows
          and cols = alignTab aligns (List.map tl rows)
        in
          alignColumn align col cols
        end;
in
  fun alignTable aligns rows =
      if List.null rows then [] else alignTab aligns rows;
end;

(* ------------------------------------------------------------------------- *)
(* Reals.                                                                    *)
(* ------------------------------------------------------------------------- *)

val realToString = Real.toString;

fun percentToString x = Int.toString (Real.round (100.0 * x)) ^ "%";

fun pos r = Real.max (r,0.0);

local
  val invLn2 = 1.0 / Math.ln 2.0;
in
  fun log2 x = invLn2 * Math.ln x;
end;

(* ------------------------------------------------------------------------- *)
(* Sums.                                                                     *)
(* ------------------------------------------------------------------------- *)

datatype ('a,'b) sum = Left of 'a | Right of 'b

fun destLeft (Left l) = l
  | destLeft _ = raise Error "destLeft";

fun isLeft (Left _) = true
  | isLeft (Right _) = false;

fun destRight (Right r) = r
  | destRight _ = raise Error "destRight";

fun isRight (Left _) = false
  | isRight (Right _) = true;

(* ------------------------------------------------------------------------- *)
(* mlibUseful impure features.                                                   *)
(* ------------------------------------------------------------------------- *)

local
  val generator = ref 0

  fun newIntThunk () =
      let
        val n = !generator

        val () = generator := n + 1
      in
        n
      end;

  fun newIntsThunk k () =
      let
        val n = !generator

        val () = generator := n + k
      in
        interval n k
      end;
in
  fun newInt () = mlibPortable.critical newIntThunk ();

  fun newInts k =
      if k <= 0 then []
      else mlibPortable.critical (newIntsThunk k) ();
end;

fun withRef (r,new) f x =
  let
    val old = !r
    val () = r := new
    val y = f x handle e => (r := old; raise e)
    val () = r := old
  in
    y
  end;

fun cloneArray a =
    let
      fun index i = Array.sub (a,i)
    in
      Array.tabulate (Array.length a, index)
    end;

(* ------------------------------------------------------------------------- *)
(* Environment.                                                              *)
(* ------------------------------------------------------------------------- *)

fun host () = Option.getOpt (OS.Process.getEnv "HOSTNAME", "unknown");

fun time () = Date.fmt "%H:%M:%S" (Date.fromTimeLocal (Time.now ()));

fun date () = Date.fmt "%d/%m/%Y" (Date.fromTimeLocal (Time.now ()));

fun readDirectory {directory = dir} =
    let
      val dirStrm = OS.FileSys.openDir dir

      fun readAll acc =
          case OS.FileSys.readDir dirStrm of
            NONE => acc
          | SOME file =>
            let
              val filename = OS.Path.joinDirFile {dir = dir, file = file}

              val acc = {filename = filename} :: acc
            in
              readAll acc
            end

      val filenames = readAll []

      val () = OS.FileSys.closeDir dirStrm
    in
      List.rev filenames
    end;

fun readTextFile {filename} =
  let
    open TextIO

    val h = openIn filename

    val contents = inputAll h

    val () = closeIn h
  in
    contents
  end;

fun writeTextFile {contents,filename} =
  let
    open TextIO
    val h = openOut filename
    val () = output (h,contents)
    val () = closeOut h
  in
    ()
  end;

(* ------------------------------------------------------------------------- *)
(* Profiling and error reporting.                                            *)
(* ------------------------------------------------------------------------- *)

fun chat s = TextIO.output (TextIO.stdOut, s ^ "\n");

fun chide s = TextIO.output (TextIO.stdErr, s ^ "\n");

local
  fun err x s = chide (x ^ ": " ^ s);
in
  fun try f x = f x
      handle e as Error _ => (err "try" (errorToString e); raise e)
           | e as Bug _ => (err "try" (bugToString e); raise e)
           | e => (err "try" "strange exception raised"; raise e);

  val warn = err "WARNING";

  fun die s = (err "\nFATAL ERROR" s; OS.Process.exit OS.Process.failure);
end;

fun timed f a =
  let
    val tmr = Timer.startCPUTimer ()
    val res = f a
    val {usr,sys,...} = Timer.checkCPUTimer tmr
  in
    (Time.toReal usr + Time.toReal sys, res)
  end;

local
  val MIN = 1.0;

  fun several n t f a =
    let
      val (t',res) = timed f a
      val t = t + t'
      val n = n + 1
    in
      if t > MIN then (t / Real.fromInt n, res) else several n t f a
    end;
in
  fun timedMany f a = several 0 0.0 f a
end;

val executionTime =
    let
      val startTime = Time.toReal (Time.now ())
    in
      fn () => Time.toReal (Time.now ()) - startTime
    end;

end
(* ========================================================================= *)
(* SUPPORT FOR LAZY EVALUATION                                               *)
(* Copyright (c) 2007 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibLazy =
sig

type 'a lazy

val quickly : 'a -> 'a lazy

val delay : (unit -> 'a) -> 'a lazy

val force : 'a lazy -> 'a

val memoize : (unit -> 'a) -> unit -> 'a

end
(* ========================================================================= *)
(* SUPPORT FOR LAZY EVALUATION                                               *)
(* Copyright (c) 2007 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibLazy :> mlibLazy =
struct

datatype 'a thunk =
    Value of 'a
  | Thunk of unit -> 'a;

datatype 'a lazy = mlibLazy of 'a thunk ref;

fun quickly v = mlibLazy (ref (Value v));

fun delay f = mlibLazy (ref (Thunk f));

fun force (mlibLazy s) =
    case !s of
      Value v => v
    | Thunk f =>
      let
        val v = f ()

        val () = s := Value v
      in
        v
      end;

fun memoize f =
    let
      val t = delay f
    in
      fn () => force t
    end;

end
(* ========================================================================= *)
(* A POSSIBLY-INFINITE STREAM DATATYPE FOR ML                                *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibStream =
sig

(* ------------------------------------------------------------------------- *)
(* The stream type.                                                          *)
(* ------------------------------------------------------------------------- *)

datatype 'a stream = Nil | Cons of 'a * (unit -> 'a stream)

(* If you're wondering how to create an infinite stream: *)
(* val stream4 = let fun s4 () = mlibStream.Cons (4,s4) in s4 () end; *)

(* ------------------------------------------------------------------------- *)
(* mlibStream constructors.                                                      *)
(* ------------------------------------------------------------------------- *)

val repeat : 'a -> 'a stream

val count : int -> int stream

val funpows : ('a -> 'a) -> 'a -> 'a stream

(* ------------------------------------------------------------------------- *)
(* mlibStream versions of standard list operations: these should all terminate.  *)
(* ------------------------------------------------------------------------- *)

val cons : 'a -> (unit -> 'a stream) -> 'a stream

val null : 'a stream -> bool

val hd : 'a stream -> 'a  (* raises Empty *)

val tl : 'a stream -> 'a stream  (* raises Empty *)

val hdTl : 'a stream -> 'a * 'a stream  (* raises Empty *)

val singleton : 'a -> 'a stream

val append : 'a stream -> (unit -> 'a stream) -> 'a stream

val map : ('a -> 'b) -> 'a stream -> 'b stream

val maps :
    ('a -> 's -> 'b * 's) -> ('s -> 'b stream) -> 's -> 'a stream -> 'b stream

val zipwith : ('a -> 'b -> 'c) -> 'a stream -> 'b stream -> 'c stream

val zip : 'a stream -> 'b stream -> ('a * 'b) stream

val take : int -> 'a stream -> 'a stream  (* raises Subscript *)

val drop : int -> 'a stream -> 'a stream  (* raises Subscript *)

(* ------------------------------------------------------------------------- *)
(* mlibStream versions of standard list operations: these might not terminate.   *)
(* ------------------------------------------------------------------------- *)

val length : 'a stream -> int

val exists : ('a -> bool) -> 'a stream -> bool

val all : ('a -> bool) -> 'a stream -> bool

val filter : ('a -> bool) -> 'a stream -> 'a stream

val foldl : ('a * 's -> 's) -> 's -> 'a stream -> 's

val concat : 'a stream stream -> 'a stream

val mapPartial : ('a -> 'b option) -> 'a stream -> 'b stream

val mapsPartial :
    ('a -> 's -> 'b option * 's) -> ('s -> 'b stream) -> 's ->
    'a stream -> 'b stream

val mapConcat : ('a -> 'b stream) -> 'a stream -> 'b stream

val mapsConcat :
    ('a -> 's -> 'b stream * 's) -> ('s -> 'b stream) -> 's ->
    'a stream -> 'b stream

(* ------------------------------------------------------------------------- *)
(* mlibStream operations.                                                        *)
(* ------------------------------------------------------------------------- *)

val memoize : 'a stream -> 'a stream

val listConcat : 'a list stream -> 'a stream

val concatList : 'a stream list -> 'a stream

val toList : 'a stream -> 'a list

val fromList : 'a list -> 'a stream

val toString : char stream -> string

val fromString : string -> char stream

val toTextFile : {filename : string} -> string stream -> unit

val fromTextFile : {filename : string} -> string stream  (* line by line *)

end
(* ========================================================================= *)
(* A POSSIBLY-INFINITE STREAM DATATYPE FOR ML                                *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibStream :> mlibStream =
struct

val K = mlibUseful.K;

val pair = mlibUseful.pair;

val funpow = mlibUseful.funpow;

(* ------------------------------------------------------------------------- *)
(* The stream type.                                                          *)
(* ------------------------------------------------------------------------- *)

datatype 'a stream =
    Nil
  | Cons of 'a * (unit -> 'a stream);

(* ------------------------------------------------------------------------- *)
(* mlibStream constructors.                                                      *)
(* ------------------------------------------------------------------------- *)

fun repeat x = let fun rep () = Cons (x,rep) in rep () end;

fun count n = Cons (n, fn () => count (n + 1));

fun funpows f x = Cons (x, fn () => funpows f (f x));

(* ------------------------------------------------------------------------- *)
(* mlibStream versions of standard list operations: these should all terminate.  *)
(* ------------------------------------------------------------------------- *)

fun cons h t = Cons (h,t);

fun null Nil = true
  | null (Cons _) = false;

fun hd Nil = raise Empty
  | hd (Cons (h,_)) = h;

fun tl Nil = raise Empty
  | tl (Cons (_,t)) = t ();

fun hdTl s = (hd s, tl s);

fun singleton s = Cons (s, K Nil);

fun append Nil s = s ()
  | append (Cons (h,t)) s = Cons (h, fn () => append (t ()) s);

fun map f =
    let
      fun m Nil = Nil
        | m (Cons (h,t)) = Cons (f h, m o t)
    in
      m
    end;

fun maps f g =
    let
      fun mm s Nil = g s
        | mm s (Cons (x,xs)) =
          let
            val (y,s') = f x s
          in
            Cons (y, mm s' o xs)
          end
    in
      mm
    end;

fun zipwith f =
    let
      fun z Nil _ = Nil
        | z _ Nil = Nil
        | z (Cons (x,xs)) (Cons (y,ys)) =
          Cons (f x y, fn () => z (xs ()) (ys ()))
    in
      z
    end;

fun zip s t = zipwith pair s t;

fun take 0 _ = Nil
  | take n Nil = raise Subscript
  | take 1 (Cons (x,_)) = Cons (x, K Nil)
  | take n (Cons (x,xs)) = Cons (x, fn () => take (n - 1) (xs ()));

fun drop n s = funpow n tl s handle Empty => raise Subscript;

(* ------------------------------------------------------------------------- *)
(* mlibStream versions of standard list operations: these might not terminate.   *)
(* ------------------------------------------------------------------------- *)

local
  fun len n Nil = n
    | len n (Cons (_,t)) = len (n + 1) (t ());
in
  fun length s = len 0 s;
end;

fun exists pred =
    let
      fun f Nil = false
        | f (Cons (h,t)) = pred h orelse f (t ())
    in
      f
    end;

fun all pred = not o exists (not o pred);

fun filter p Nil = Nil
  | filter p (Cons (x,xs)) =
    if p x then Cons (x, fn () => filter p (xs ())) else filter p (xs ());

fun foldl f =
    let
      fun fold b Nil = b
        | fold b (Cons (h,t)) = fold (f (h,b)) (t ())
    in
      fold
    end;

fun concat Nil = Nil
  | concat (Cons (Nil, ss)) = concat (ss ())
  | concat (Cons (Cons (x, xs), ss)) =
    Cons (x, fn () => concat (Cons (xs (), ss)));

fun mapPartial f =
    let
      fun mp Nil = Nil
        | mp (Cons (h,t)) =
          case f h of
            NONE => mp (t ())
          | SOME h' => Cons (h', fn () => mp (t ()))
    in
      mp
    end;

fun mapsPartial f g =
    let
      fun mp s Nil = g s
        | mp s (Cons (h,t)) =
          let
            val (h,s) = f h s
          in
            case h of
              NONE => mp s (t ())
            | SOME h => Cons (h, fn () => mp s (t ()))
          end
    in
      mp
    end;

fun mapConcat f =
    let
      fun mc Nil = Nil
        | mc (Cons (h,t)) = append (f h) (fn () => mc (t ()))
    in
      mc
    end;

fun mapsConcat f g =
    let
      fun mc s Nil = g s
        | mc s (Cons (h,t)) =
          let
            val (l,s) = f h s
          in
            append l (fn () => mc s (t ()))
          end
    in
      mc
    end;

(* ------------------------------------------------------------------------- *)
(* mlibStream operations.                                                        *)
(* ------------------------------------------------------------------------- *)

fun memoize Nil = Nil
  | memoize (Cons (h,t)) = Cons (h, mlibLazy.memoize (fn () => memoize (t ())));

fun concatList [] = Nil
  | concatList (h :: t) = append h (fn () => concatList t);

local
  fun toLst res Nil = List.rev res
    | toLst res (Cons (x, xs)) = toLst (x :: res) (xs ());
in
  fun toList s = toLst [] s;
end;

fun fromList [] = Nil
  | fromList (x :: xs) = Cons (x, fn () => fromList xs);

fun listConcat s = concat (map fromList s);

fun toString s = String.implode (toList s);

fun fromString s = fromList (String.explode s);

fun toTextFile {filename = f} s =
    let
      val (h,close) =
          if f = "-" then (TextIO.stdOut, K ())
          else (TextIO.openOut f, TextIO.closeOut)

      fun toFile Nil = ()
        | toFile (Cons (x,y)) = (TextIO.output (h,x); toFile (y ()))

      val () = toFile s
    in
      close h
    end;

fun fromTextFile {filename = f} =
    let
      val (h,close) =
          if f = "-" then (TextIO.stdIn, K ())
          else (TextIO.openIn f, TextIO.closeIn)

      fun strm () =
          case TextIO.inputLine h of
            NONE => (close h; Nil)
          | SOME s => Cons (s,strm)
    in
      memoize (strm ())
    end;

end
(* ========================================================================= *)
(* ORDERED TYPES                                                             *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibOrdered =
sig

type t

val compare : t * t -> order

(*
  PROVIDES

  !x : t. compare (x,x) = EQUAL

  !x y : t. compare (x,y) = LESS <=> compare (y,x) = GREATER

  !x y : t. compare (x,y) = EQUAL ==> compare (y,x) = EQUAL

  !x y z : t. compare (x,y) = EQUAL ==> compare (x,z) = compare (y,z)

  !x y z : t.
    compare (x,y) = LESS andalso compare (y,z) = LESS ==>
    compare (x,z) = LESS

  !x y z : t.
    compare (x,y) = GREATER andalso compare (y,z) = GREATER ==>
    compare (x,z) = GREATER
*)

end
(* ========================================================================= *)
(* ORDERED TYPES                                                             *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibIntOrdered =
struct type t = int val compare = Int.compare end;

structure mlibIntPairOrdered =
struct

type t = int * int;

fun compare ((i1,j1),(i2,j2)) =
    case Int.compare (i1,i2) of
      LESS => LESS
    | EQUAL => Int.compare (j1,j2)
    | GREATER => GREATER;

end;

structure mlibStringOrdered =
struct type t = string val compare = String.compare end;
(* ========================================================================= *)
(* FINITE MAPS WITH A FIXED KEY TYPE                                         *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibKeyMap =
sig

(* ------------------------------------------------------------------------- *)
(* A type of map keys.                                                       *)
(* ------------------------------------------------------------------------- *)

type key

val compareKey : key * key -> order

val equalKey : key -> key -> bool

(* ------------------------------------------------------------------------- *)
(* A type of finite maps.                                                    *)
(* ------------------------------------------------------------------------- *)

type 'a map

(* ------------------------------------------------------------------------- *)
(* Constructors.                                                             *)
(* ------------------------------------------------------------------------- *)

val new : unit -> 'a map

val singleton : key * 'a -> 'a map

(* ------------------------------------------------------------------------- *)
(* mlibMap size.                                                                 *)
(* ------------------------------------------------------------------------- *)

val null : 'a map -> bool

val size : 'a map -> int

(* ------------------------------------------------------------------------- *)
(* Querying.                                                                 *)
(* ------------------------------------------------------------------------- *)

val peekKey : 'a map -> key -> (key * 'a) option

val peek : 'a map -> key -> 'a option

val get : 'a map -> key -> 'a  (* raises Error *)

val pick : 'a map -> key * 'a  (* an arbitrary key/value pair *)

val nth : 'a map -> int -> key * 'a  (* in the range [0,size-1] *)

val random : 'a map -> key * 'a

(* ------------------------------------------------------------------------- *)
(* Adding.                                                                   *)
(* ------------------------------------------------------------------------- *)

val insert : 'a map -> key * 'a -> 'a map

val insertList : 'a map -> (key * 'a) list -> 'a map

(* ------------------------------------------------------------------------- *)
(* Removing.                                                                 *)
(* ------------------------------------------------------------------------- *)

val delete : 'a map -> key -> 'a map  (* must be present *)

val remove : 'a map -> key -> 'a map

val deletePick : 'a map -> (key * 'a) * 'a map

val deleteNth : 'a map -> int -> (key * 'a) * 'a map

val deleteRandom : 'a map -> (key * 'a) * 'a map

(* ------------------------------------------------------------------------- *)
(* Joining (all join operations prefer keys in the second map).              *)
(* ------------------------------------------------------------------------- *)

val merge :
    {first : key * 'a -> 'c option,
     second : key * 'b -> 'c option,
     both : (key * 'a) * (key * 'b) -> 'c option} ->
    'a map -> 'b map -> 'c map

val union :
    ((key * 'a) * (key * 'a) -> 'a option) ->
    'a map -> 'a map -> 'a map

val intersect :
    ((key * 'a) * (key * 'b) -> 'c option) ->
    'a map -> 'b map -> 'c map

(* ------------------------------------------------------------------------- *)
(* Set operations on the domain.                                             *)
(* ------------------------------------------------------------------------- *)

val inDomain : key -> 'a map -> bool

val unionDomain : 'a map -> 'a map -> 'a map

val unionListDomain : 'a map list -> 'a map

val intersectDomain : 'a map -> 'a map -> 'a map

val intersectListDomain : 'a map list -> 'a map

val differenceDomain : 'a map -> 'a map -> 'a map

val symmetricDifferenceDomain : 'a map -> 'a map -> 'a map

val equalDomain : 'a map -> 'a map -> bool

val subsetDomain : 'a map -> 'a map -> bool

val disjointDomain : 'a map -> 'a map -> bool

(* ------------------------------------------------------------------------- *)
(* Mapping and folding.                                                      *)
(* ------------------------------------------------------------------------- *)

val mapPartial : (key * 'a -> 'b option) -> 'a map -> 'b map

val map : (key * 'a -> 'b) -> 'a map -> 'b map

val app : (key * 'a -> unit) -> 'a map -> unit

val transform : ('a -> 'b) -> 'a map -> 'b map

val filter : (key * 'a -> bool) -> 'a map -> 'a map

val partition :
    (key * 'a -> bool) -> 'a map -> 'a map * 'a map

val foldl : (key * 'a * 's -> 's) -> 's -> 'a map -> 's

val foldr : (key * 'a * 's -> 's) -> 's -> 'a map -> 's

(* ------------------------------------------------------------------------- *)
(* Searching.                                                                *)
(* ------------------------------------------------------------------------- *)

val findl : (key * 'a -> bool) -> 'a map -> (key * 'a) option

val findr : (key * 'a -> bool) -> 'a map -> (key * 'a) option

val firstl : (key * 'a -> 'b option) -> 'a map -> 'b option

val firstr : (key * 'a -> 'b option) -> 'a map -> 'b option

val exists : (key * 'a -> bool) -> 'a map -> bool

val all : (key * 'a -> bool) -> 'a map -> bool

val count : (key * 'a -> bool) -> 'a map -> int

(* ------------------------------------------------------------------------- *)
(* Comparing.                                                                *)
(* ------------------------------------------------------------------------- *)

val compare : ('a * 'a -> order) -> 'a map * 'a map -> order

val equal : ('a -> 'a -> bool) -> 'a map -> 'a map -> bool

(* ------------------------------------------------------------------------- *)
(* Converting to and from lists.                                             *)
(* ------------------------------------------------------------------------- *)

val keys : 'a map -> key list

val values : 'a map -> 'a list

val toList : 'a map -> (key * 'a) list

val fromList : (key * 'a) list -> 'a map

(* ------------------------------------------------------------------------- *)
(* Pretty-printing.                                                          *)
(* ------------------------------------------------------------------------- *)

val toString : 'a map -> string

(* ------------------------------------------------------------------------- *)
(* Iterators over maps.                                                      *)
(* ------------------------------------------------------------------------- *)

type 'a iterator

val mkIterator : 'a map -> 'a iterator option

val mkRevIterator : 'a map -> 'a iterator option

val readIterator : 'a iterator -> key * 'a

val advanceIterator : 'a iterator -> 'a iterator option

end
(* ========================================================================= *)
(* FINITE MAPS WITH A FIXED KEY TYPE                                         *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

functor mlibKeyMap (Key : mlibOrdered) :> mlibKeyMap where type key = Key.t =
struct

(* ------------------------------------------------------------------------- *)
(* Importing from the input signature.                                       *)
(* ------------------------------------------------------------------------- *)

type key = Key.t;

val compareKey = Key.compare;

(* ------------------------------------------------------------------------- *)
(* Importing useful functionality.                                           *)
(* ------------------------------------------------------------------------- *)

exception Bug = mlibUseful.Bug;

exception Error = mlibUseful.Error;

val pointerEqual = mlibPortable.pointerEqual;

val K = mlibUseful.K;

val randomInt = mlibPortable.randomInt;

val randomWord = mlibPortable.randomWord;

(* ------------------------------------------------------------------------- *)
(* Converting a comparison function to an equality function.                 *)
(* ------------------------------------------------------------------------- *)

fun equalKey key1 key2 = compareKey (key1,key2) = EQUAL;

(* ------------------------------------------------------------------------- *)
(* Priorities.                                                               *)
(* ------------------------------------------------------------------------- *)

type priority = Word.word;

val randomPriority = randomWord;

val comparePriority = Word.compare;

(* ------------------------------------------------------------------------- *)
(* Priority search trees.                                                    *)
(* ------------------------------------------------------------------------- *)

datatype 'value tree =
    E
  | T of 'value node

and 'value node =
    Node of
      {size : int,
       priority : priority,
       left : 'value tree,
       key : key,
       value : 'value,
       right : 'value tree};

fun lowerPriorityNode node1 node2 =
    let
      val Node {priority = p1, ...} = node1
      and Node {priority = p2, ...} = node2
    in
      comparePriority (p1,p2) = LESS
    end;

(* ------------------------------------------------------------------------- *)
(* Tree debugging functions.                                                 *)
(* ------------------------------------------------------------------------- *)

(*BasicDebug
local
  fun checkSizes tree =
      case tree of
        E => 0
      | T (Node {size,left,right,...}) =>
        let
          val l = checkSizes left
          and r = checkSizes right

          val () = if l + 1 + r = size then () else raise Bug "wrong size"
        in
          size
        end;

  fun checkSorted x tree =
      case tree of
        E => x
      | T (Node {left,key,right,...}) =>
        let
          val x = checkSorted x left

          val () =
              case x of
                NONE => ()
              | SOME k =>
                case compareKey (k,key) of
                  LESS => ()
                | EQUAL => raise Bug "duplicate keys"
                | GREATER => raise Bug "unsorted"

          val x = SOME key
        in
          checkSorted x right
        end;

  fun checkPriorities tree =
      case tree of
        E => NONE
      | T node =>
        let
          val Node {left,right,...} = node

          val () =
              case checkPriorities left of
                NONE => ()
              | SOME lnode =>
                if not (lowerPriorityNode node lnode) then ()
                else raise Bug "left child has greater priority"

          val () =
              case checkPriorities right of
                NONE => ()
              | SOME rnode =>
                if not (lowerPriorityNode node rnode) then ()
                else raise Bug "right child has greater priority"
        in
          SOME node
        end;
in
  fun treeCheckInvariants tree =
      let
        val _ = checkSizes tree

        val _ = checkSorted NONE tree

        val _ = checkPriorities tree
      in
        tree
      end
      handle Error err => raise Bug err;
end;
*)

(* ------------------------------------------------------------------------- *)
(* Tree operations.                                                          *)
(* ------------------------------------------------------------------------- *)

fun treeNew () = E;

fun nodeSize (Node {size = x, ...}) = x;

fun treeSize tree =
    case tree of
      E => 0
    | T x => nodeSize x;

fun mkNode priority left key value right =
    let
      val size = treeSize left + 1 + treeSize right
    in
      Node
        {size = size,
         priority = priority,
         left = left,
         key = key,
         value = value,
         right = right}
    end;

fun mkTree priority left key value right =
    let
      val node = mkNode priority left key value right
    in
      T node
    end;

(* ------------------------------------------------------------------------- *)
(* Extracting the left and right spines of a tree.                           *)
(* ------------------------------------------------------------------------- *)

fun treeLeftSpine acc tree =
    case tree of
      E => acc
    | T node => nodeLeftSpine acc node

and nodeLeftSpine acc node =
    let
      val Node {left,...} = node
    in
      treeLeftSpine (node :: acc) left
    end;

fun treeRightSpine acc tree =
    case tree of
      E => acc
    | T node => nodeRightSpine acc node

and nodeRightSpine acc node =
    let
      val Node {right,...} = node
    in
      treeRightSpine (node :: acc) right
    end;

(* ------------------------------------------------------------------------- *)
(* Singleton trees.                                                          *)
(* ------------------------------------------------------------------------- *)

fun mkNodeSingleton priority key value =
    let
      val size = 1
      and left = E
      and right = E
    in
      Node
        {size = size,
         priority = priority,
         left = left,
         key = key,
         value = value,
         right = right}
    end;

fun nodeSingleton (key,value) =
    let
      val priority = randomPriority ()
    in
      mkNodeSingleton priority key value
    end;

fun treeSingleton key_value =
    let
      val node = nodeSingleton key_value
    in
      T node
    end;

(* ------------------------------------------------------------------------- *)
(* Appending two trees, where every element of the first tree is less than   *)
(* every element of the second tree.                                         *)
(* ------------------------------------------------------------------------- *)

fun treeAppend tree1 tree2 =
    case tree1 of
      E => tree2
    | T node1 =>
      case tree2 of
        E => tree1
      | T node2 =>
        if lowerPriorityNode node1 node2 then
          let
            val Node {priority,left,key,value,right,...} = node2

            val left = treeAppend tree1 left
          in
            mkTree priority left key value right
          end
        else
          let
            val Node {priority,left,key,value,right,...} = node1

            val right = treeAppend right tree2
          in
            mkTree priority left key value right
          end;

(* ------------------------------------------------------------------------- *)
(* Appending two trees and a node, where every element of the first tree is  *)
(* less than the node, which in turn is less than every element of the       *)
(* second tree.                                                              *)
(* ------------------------------------------------------------------------- *)

fun treeCombine left node right =
    let
      val left_node = treeAppend left (T node)
    in
      treeAppend left_node right
    end;

(* ------------------------------------------------------------------------- *)
(* Searching a tree for a value.                                             *)
(* ------------------------------------------------------------------------- *)

fun treePeek pkey tree =
    case tree of
      E => NONE
    | T node => nodePeek pkey node

and nodePeek pkey node =
    let
      val Node {left,key,value,right,...} = node
    in
      case compareKey (pkey,key) of
        LESS => treePeek pkey left
      | EQUAL => SOME value
      | GREATER => treePeek pkey right
    end;

(* ------------------------------------------------------------------------- *)
(* Tree paths.                                                               *)
(* ------------------------------------------------------------------------- *)

(* Generating a path by searching a tree for a key/value pair *)

fun treePeekPath pkey path tree =
    case tree of
      E => (path,NONE)
    | T node => nodePeekPath pkey path node

and nodePeekPath pkey path node =
    let
      val Node {left,key,right,...} = node
    in
      case compareKey (pkey,key) of
        LESS => treePeekPath pkey ((true,node) :: path) left
      | EQUAL => (path, SOME node)
      | GREATER => treePeekPath pkey ((false,node) :: path) right
    end;

(* A path splits a tree into left/right components *)

fun addSidePath ((wentLeft,node),(leftTree,rightTree)) =
    let
      val Node {priority,left,key,value,right,...} = node
    in
      if wentLeft then (leftTree, mkTree priority rightTree key value right)
      else (mkTree priority left key value leftTree, rightTree)
    end;

fun addSidesPath left_right = List.foldl addSidePath left_right;

fun mkSidesPath path = addSidesPath (E,E) path;

(* Updating the subtree at a path *)

local
  fun updateTree ((wentLeft,node),tree) =
      let
        val Node {priority,left,key,value,right,...} = node
      in
        if wentLeft then mkTree priority tree key value right
        else mkTree priority left key value tree
      end;
in
  fun updateTreePath tree = List.foldl updateTree tree;
end;

(* Inserting a new node at a path position *)

fun insertNodePath node =
    let
      fun insert left_right path =
          case path of
            [] =>
            let
              val (left,right) = left_right
            in
              treeCombine left node right
            end
          | (step as (_,snode)) :: rest =>
            if lowerPriorityNode snode node then
              let
                val left_right = addSidePath (step,left_right)
              in
                insert left_right rest
              end
            else
              let
                val (left,right) = left_right

                val tree = treeCombine left node right
              in
                updateTreePath tree path
              end
    in
      insert (E,E)
    end;

(* ------------------------------------------------------------------------- *)
(* Using a key to split a node into three components: the keys comparing     *)
(* less than the supplied key, an optional equal key, and the keys comparing *)
(* greater.                                                                  *)
(* ------------------------------------------------------------------------- *)

fun nodePartition pkey node =
    let
      val (path,pnode) = nodePeekPath pkey [] node
    in
      case pnode of
        NONE =>
        let
          val (left,right) = mkSidesPath path
        in
          (left,NONE,right)
        end
      | SOME node =>
        let
          val Node {left,key,value,right,...} = node

          val (left,right) = addSidesPath (left,right) path
        in
          (left, SOME (key,value), right)
        end
    end;

(* ------------------------------------------------------------------------- *)
(* Searching a tree for a key/value pair.                                    *)
(* ------------------------------------------------------------------------- *)

fun treePeekKey pkey tree =
    case tree of
      E => NONE
    | T node => nodePeekKey pkey node

and nodePeekKey pkey node =
    let
      val Node {left,key,value,right,...} = node
    in
      case compareKey (pkey,key) of
        LESS => treePeekKey pkey left
      | EQUAL => SOME (key,value)
      | GREATER => treePeekKey pkey right
    end;

(* ------------------------------------------------------------------------- *)
(* Inserting new key/values into the tree.                                   *)
(* ------------------------------------------------------------------------- *)

fun treeInsert key_value tree =
    let
      val (key,value) = key_value

      val (path,inode) = treePeekPath key [] tree
    in
      case inode of
        NONE =>
        let
          val node = nodeSingleton (key,value)
        in
          insertNodePath node path
        end
      | SOME node =>
        let
          val Node {size,priority,left,right,...} = node

          val node =
              Node
                {size = size,
                 priority = priority,
                 left = left,
                 key = key,
                 value = value,
                 right = right}
        in
          updateTreePath (T node) path
        end
    end;

(* ------------------------------------------------------------------------- *)
(* Deleting key/value pairs: it raises an exception if the supplied key is   *)
(* not present.                                                              *)
(* ------------------------------------------------------------------------- *)

fun treeDelete dkey tree =
    case tree of
      E => raise Bug "mlibKeyMap.delete: element not found"
    | T node => nodeDelete dkey node

and nodeDelete dkey node =
    let
      val Node {size,priority,left,key,value,right} = node
    in
      case compareKey (dkey,key) of
        LESS =>
        let
          val size = size - 1
          and left = treeDelete dkey left

          val node =
              Node
                {size = size,
                 priority = priority,
                 left = left,
                 key = key,
                 value = value,
                 right = right}
        in
          T node
        end
      | EQUAL => treeAppend left right
      | GREATER =>
        let
          val size = size - 1
          and right = treeDelete dkey right

          val node =
              Node
                {size = size,
                 priority = priority,
                 left = left,
                 key = key,
                 value = value,
                 right = right}
        in
          T node
        end
    end;

(* ------------------------------------------------------------------------- *)
(* Partial map is the basic operation for preserving tree structure.         *)
(* It applies its argument function to the elements *in order*.              *)
(* ------------------------------------------------------------------------- *)

fun treeMapPartial f tree =
    case tree of
      E => E
    | T node => nodeMapPartial f node

and nodeMapPartial f (Node {priority,left,key,value,right,...}) =
    let
      val left = treeMapPartial f left
      and vo = f (key,value)
      and right = treeMapPartial f right
    in
      case vo of
        NONE => treeAppend left right
      | SOME value => mkTree priority left key value right
    end;

(* ------------------------------------------------------------------------- *)
(* Mapping tree values.                                                      *)
(* ------------------------------------------------------------------------- *)

fun treeMap f tree =
    case tree of
      E => E
    | T node => T (nodeMap f node)

and nodeMap f node =
    let
      val Node {size,priority,left,key,value,right} = node

      val left = treeMap f left
      and value = f (key,value)
      and right = treeMap f right
    in
      Node
        {size = size,
         priority = priority,
         left = left,
         key = key,
         value = value,
         right = right}
    end;

(* ------------------------------------------------------------------------- *)
(* Merge is the basic operation for joining two trees. Note that the merged  *)
(* key is always the one from the second map.                                *)
(* ------------------------------------------------------------------------- *)

fun treeMerge f1 f2 fb tree1 tree2 =
    case tree1 of
      E => treeMapPartial f2 tree2
    | T node1 =>
      case tree2 of
        E => treeMapPartial f1 tree1
      | T node2 => nodeMerge f1 f2 fb node1 node2

and nodeMerge f1 f2 fb node1 node2 =
    let
      val Node {priority,left,key,value,right,...} = node2

      val (l,kvo,r) = nodePartition key node1

      val left = treeMerge f1 f2 fb l left
      and right = treeMerge f1 f2 fb r right

      val vo =
          case kvo of
            NONE => f2 (key,value)
          | SOME kv => fb (kv,(key,value))
    in
      case vo of
        NONE => treeAppend left right
      | SOME value =>
        let
          val node = mkNodeSingleton priority key value
        in
          treeCombine left node right
        end
    end;

(* ------------------------------------------------------------------------- *)
(* A union operation on trees.                                               *)
(* ------------------------------------------------------------------------- *)

fun treeUnion f f2 tree1 tree2 =
    case tree1 of
      E => tree2
    | T node1 =>
      case tree2 of
        E => tree1
      | T node2 => nodeUnion f f2 node1 node2

and nodeUnion f f2 node1 node2 =
    if pointerEqual (node1,node2) then nodeMapPartial f2 node1
    else
      let
        val Node {priority,left,key,value,right,...} = node2

        val (l,kvo,r) = nodePartition key node1

        val left = treeUnion f f2 l left
        and right = treeUnion f f2 r right

        val vo =
            case kvo of
              NONE => SOME value
            | SOME kv => f (kv,(key,value))
      in
        case vo of
          NONE => treeAppend left right
        | SOME value =>
          let
            val node = mkNodeSingleton priority key value
          in
            treeCombine left node right
          end
      end;

(* ------------------------------------------------------------------------- *)
(* An intersect operation on trees.                                          *)
(* ------------------------------------------------------------------------- *)

fun treeIntersect f t1 t2 =
    case t1 of
      E => E
    | T n1 =>
      case t2 of
        E => E
      | T n2 => nodeIntersect f n1 n2

and nodeIntersect f n1 n2 =
    let
      val Node {priority,left,key,value,right,...} = n2

      val (l,kvo,r) = nodePartition key n1

      val left = treeIntersect f l left
      and right = treeIntersect f r right

      val vo =
          case kvo of
            NONE => NONE
          | SOME kv => f (kv,(key,value))
    in
      case vo of
        NONE => treeAppend left right
      | SOME value => mkTree priority left key value right
    end;

(* ------------------------------------------------------------------------- *)
(* A union operation on trees which simply chooses the second value.         *)
(* ------------------------------------------------------------------------- *)

fun treeUnionDomain tree1 tree2 =
    case tree1 of
      E => tree2
    | T node1 =>
      case tree2 of
        E => tree1
      | T node2 =>
        if pointerEqual (node1,node2) then tree2
        else nodeUnionDomain node1 node2

and nodeUnionDomain node1 node2 =
    let
      val Node {priority,left,key,value,right,...} = node2

      val (l,_,r) = nodePartition key node1

      val left = treeUnionDomain l left
      and right = treeUnionDomain r right

      val node = mkNodeSingleton priority key value
    in
      treeCombine left node right
    end;

(* ------------------------------------------------------------------------- *)
(* An intersect operation on trees which simply chooses the second value.    *)
(* ------------------------------------------------------------------------- *)

fun treeIntersectDomain tree1 tree2 =
    case tree1 of
      E => E
    | T node1 =>
      case tree2 of
        E => E
      | T node2 =>
        if pointerEqual (node1,node2) then tree2
        else nodeIntersectDomain node1 node2

and nodeIntersectDomain node1 node2 =
    let
      val Node {priority,left,key,value,right,...} = node2

      val (l,kvo,r) = nodePartition key node1

      val left = treeIntersectDomain l left
      and right = treeIntersectDomain r right
    in
      if Option.isSome kvo then mkTree priority left key value right
      else treeAppend left right
    end;

(* ------------------------------------------------------------------------- *)
(* A difference operation on trees.                                          *)
(* ------------------------------------------------------------------------- *)

fun treeDifferenceDomain t1 t2 =
    case t1 of
      E => E
    | T n1 =>
      case t2 of
        E => t1
      | T n2 => nodeDifferenceDomain n1 n2

and nodeDifferenceDomain n1 n2 =
    if pointerEqual (n1,n2) then E
    else
      let
        val Node {priority,left,key,value,right,...} = n1

        val (l,kvo,r) = nodePartition key n2

        val left = treeDifferenceDomain left l
        and right = treeDifferenceDomain right r
      in
        if Option.isSome kvo then treeAppend left right
        else mkTree priority left key value right
      end;

(* ------------------------------------------------------------------------- *)
(* A subset operation on trees.                                              *)
(* ------------------------------------------------------------------------- *)

fun treeSubsetDomain tree1 tree2 =
    case tree1 of
      E => true
    | T node1 =>
      case tree2 of
        E => false
      | T node2 => nodeSubsetDomain node1 node2

and nodeSubsetDomain node1 node2 =
    pointerEqual (node1,node2) orelse
    let
      val Node {size,left,key,right,...} = node1
    in
      size <= nodeSize node2 andalso
      let
        val (l,kvo,r) = nodePartition key node2
      in
        Option.isSome kvo andalso
        treeSubsetDomain left l andalso
        treeSubsetDomain right r
      end
    end;

(* ------------------------------------------------------------------------- *)
(* Picking an arbitrary key/value pair from a tree.                          *)
(* ------------------------------------------------------------------------- *)

fun nodePick node =
    let
      val Node {key,value,...} = node
    in
      (key,value)
    end;

fun treePick tree =
    case tree of
      E => raise Bug "mlibKeyMap.treePick"
    | T node => nodePick node;

(* ------------------------------------------------------------------------- *)
(* Removing an arbitrary key/value pair from a tree.                         *)
(* ------------------------------------------------------------------------- *)

fun nodeDeletePick node =
    let
      val Node {left,key,value,right,...} = node
    in
      ((key,value), treeAppend left right)
    end;

fun treeDeletePick tree =
    case tree of
      E => raise Bug "mlibKeyMap.treeDeletePick"
    | T node => nodeDeletePick node;

(* ------------------------------------------------------------------------- *)
(* Finding the nth smallest key/value (counting from 0).                     *)
(* ------------------------------------------------------------------------- *)

fun treeNth n tree =
    case tree of
      E => raise Bug "mlibKeyMap.treeNth"
    | T node => nodeNth n node

and nodeNth n node =
    let
      val Node {left,key,value,right,...} = node

      val k = treeSize left
    in
      if n = k then (key,value)
      else if n < k then treeNth n left
      else treeNth (n - (k + 1)) right
    end;

(* ------------------------------------------------------------------------- *)
(* Removing the nth smallest key/value (counting from 0).                    *)
(* ------------------------------------------------------------------------- *)

fun treeDeleteNth n tree =
    case tree of
      E => raise Bug "mlibKeyMap.treeDeleteNth"
    | T node => nodeDeleteNth n node

and nodeDeleteNth n node =
    let
      val Node {size,priority,left,key,value,right} = node

      val k = treeSize left
    in
      if n = k then ((key,value), treeAppend left right)
      else if n < k then
        let
          val (key_value,left) = treeDeleteNth n left

          val size = size - 1

          val node =
              Node
                {size = size,
                 priority = priority,
                 left = left,
                 key = key,
                 value = value,
                 right = right}
        in
          (key_value, T node)
        end
      else
        let
          val n = n - (k + 1)

          val (key_value,right) = treeDeleteNth n right

          val size = size - 1

          val node =
              Node
                {size = size,
                 priority = priority,
                 left = left,
                 key = key,
                 value = value,
                 right = right}
        in
          (key_value, T node)
        end
    end;

(* ------------------------------------------------------------------------- *)
(* Iterators.                                                                *)
(* ------------------------------------------------------------------------- *)

datatype 'value iterator =
    LeftToRightIterator of
      (key * 'value) * 'value tree * 'value node list
  | RightToLeftIterator of
      (key * 'value) * 'value tree * 'value node list;

fun fromSpineLeftToRightIterator nodes =
    case nodes of
      [] => NONE
    | Node {key,value,right,...} :: nodes =>
      SOME (LeftToRightIterator ((key,value),right,nodes));

fun fromSpineRightToLeftIterator nodes =
    case nodes of
      [] => NONE
    | Node {key,value,left,...} :: nodes =>
      SOME (RightToLeftIterator ((key,value),left,nodes));

fun addLeftToRightIterator nodes tree = fromSpineLeftToRightIterator (treeLeftSpine nodes tree);

fun addRightToLeftIterator nodes tree = fromSpineRightToLeftIterator (treeRightSpine nodes tree);

fun treeMkIterator tree = addLeftToRightIterator [] tree;

fun treeMkRevIterator tree = addRightToLeftIterator [] tree;

fun readIterator iter =
    case iter of
      LeftToRightIterator (key_value,_,_) => key_value
    | RightToLeftIterator (key_value,_,_) => key_value;

fun advanceIterator iter =
    case iter of
      LeftToRightIterator (_,tree,nodes) => addLeftToRightIterator nodes tree
    | RightToLeftIterator (_,tree,nodes) => addRightToLeftIterator nodes tree;

fun foldIterator f acc io =
    case io of
      NONE => acc
    | SOME iter =>
      let
        val (key,value) = readIterator iter
      in
        foldIterator f (f (key,value,acc)) (advanceIterator iter)
      end;

fun findIterator pred io =
    case io of
      NONE => NONE
    | SOME iter =>
      let
        val key_value = readIterator iter
      in
        if pred key_value then SOME key_value
        else findIterator pred (advanceIterator iter)
      end;

fun firstIterator f io =
    case io of
      NONE => NONE
    | SOME iter =>
      let
        val key_value = readIterator iter
      in
        case f key_value of
          NONE => firstIterator f (advanceIterator iter)
        | s => s
      end;

fun compareIterator compareValue io1 io2 =
    case (io1,io2) of
      (NONE,NONE) => EQUAL
    | (NONE, SOME _) => LESS
    | (SOME _, NONE) => GREATER
    | (SOME i1, SOME i2) =>
      let
        val (k1,v1) = readIterator i1
        and (k2,v2) = readIterator i2
      in
        case compareKey (k1,k2) of
          LESS => LESS
        | EQUAL =>
          (case compareValue (v1,v2) of
             LESS => LESS
           | EQUAL =>
             let
               val io1 = advanceIterator i1
               and io2 = advanceIterator i2
             in
               compareIterator compareValue io1 io2
             end
           | GREATER => GREATER)
        | GREATER => GREATER
      end;

fun equalIterator equalValue io1 io2 =
    case (io1,io2) of
      (NONE,NONE) => true
    | (NONE, SOME _) => false
    | (SOME _, NONE) => false
    | (SOME i1, SOME i2) =>
      let
        val (k1,v1) = readIterator i1
        and (k2,v2) = readIterator i2
      in
        equalKey k1 k2 andalso
        equalValue v1 v2 andalso
        let
          val io1 = advanceIterator i1
          and io2 = advanceIterator i2
        in
          equalIterator equalValue io1 io2
        end
      end;

(* ------------------------------------------------------------------------- *)
(* A type of finite maps.                                                    *)
(* ------------------------------------------------------------------------- *)

datatype 'value map =
    mlibMap of 'value tree;

(* ------------------------------------------------------------------------- *)
(* mlibMap debugging functions.                                                  *)
(* ------------------------------------------------------------------------- *)

(*BasicDebug
fun checkInvariants s m =
    let
      val mlibMap tree = m

      val _ = treeCheckInvariants tree
    in
      m
    end
    handle Bug bug => raise Bug (s ^ "\n" ^ "mlibKeyMap.checkInvariants: " ^ bug);
*)

(* ------------------------------------------------------------------------- *)
(* Constructors.                                                             *)
(* ------------------------------------------------------------------------- *)

fun new () =
    let
      val tree = treeNew ()
    in
      mlibMap tree
    end;

fun singleton key_value =
    let
      val tree = treeSingleton key_value
    in
      mlibMap tree
    end;

(* ------------------------------------------------------------------------- *)
(* mlibMap size.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun size (mlibMap tree) = treeSize tree;

fun null m = size m = 0;

(* ------------------------------------------------------------------------- *)
(* Querying.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun peekKey (mlibMap tree) key = treePeekKey key tree;

fun peek (mlibMap tree) key = treePeek key tree;

fun inDomain key m = Option.isSome (peek m key);

fun get m key =
    case peek m key of
      NONE => raise Error "mlibKeyMap.get: element not found"
    | SOME value => value;

fun pick (mlibMap tree) = treePick tree;

fun nth (mlibMap tree) n = treeNth n tree;

fun random m =
    let
      val n = size m
    in
      if n = 0 then raise Bug "mlibKeyMap.random: empty"
      else nth m (randomInt n)
    end;

(* ------------------------------------------------------------------------- *)
(* Adding.                                                                   *)
(* ------------------------------------------------------------------------- *)

fun insert (mlibMap tree) key_value =
    let
      val tree = treeInsert key_value tree
    in
      mlibMap tree
    end;

(*BasicDebug
val insert = fn m => fn kv =>
    checkInvariants "mlibKeyMap.insert: result"
      (insert (checkInvariants "mlibKeyMap.insert: input" m) kv);
*)

fun insertList m =
    let
      fun ins (key_value,acc) = insert acc key_value
    in
      List.foldl ins m
    end;

(* ------------------------------------------------------------------------- *)
(* Removing.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun delete (mlibMap tree) dkey =
    let
      val tree = treeDelete dkey tree
    in
      mlibMap tree
    end;

(*BasicDebug
val delete = fn m => fn k =>
    checkInvariants "mlibKeyMap.delete: result"
      (delete (checkInvariants "mlibKeyMap.delete: input" m) k);
*)

fun remove m key = if inDomain key m then delete m key else m;

fun deletePick (mlibMap tree) =
    let
      val (key_value,tree) = treeDeletePick tree
    in
      (key_value, mlibMap tree)
    end;

(*BasicDebug
val deletePick = fn m =>
    let
      val (kv,m) = deletePick (checkInvariants "mlibKeyMap.deletePick: input" m)
    in
      (kv, checkInvariants "mlibKeyMap.deletePick: result" m)
    end;
*)

fun deleteNth (mlibMap tree) n =
    let
      val (key_value,tree) = treeDeleteNth n tree
    in
      (key_value, mlibMap tree)
    end;

(*BasicDebug
val deleteNth = fn m => fn n =>
    let
      val (kv,m) = deleteNth (checkInvariants "mlibKeyMap.deleteNth: input" m) n
    in
      (kv, checkInvariants "mlibKeyMap.deleteNth: result" m)
    end;
*)

fun deleteRandom m =
    let
      val n = size m
    in
      if n = 0 then raise Bug "mlibKeyMap.deleteRandom: empty"
      else deleteNth m (randomInt n)
    end;

(* ------------------------------------------------------------------------- *)
(* Joining (all join operations prefer keys in the second map).              *)
(* ------------------------------------------------------------------------- *)

fun merge {first,second,both} (mlibMap tree1) (mlibMap tree2) =
    let
      val tree = treeMerge first second both tree1 tree2
    in
      mlibMap tree
    end;

(*BasicDebug
val merge = fn f => fn m1 => fn m2 =>
    checkInvariants "mlibKeyMap.merge: result"
      (merge f
         (checkInvariants "mlibKeyMap.merge: input 1" m1)
         (checkInvariants "mlibKeyMap.merge: input 2" m2));
*)

fun union f (mlibMap tree1) (mlibMap tree2) =
    let
      fun f2 kv = f (kv,kv)

      val tree = treeUnion f f2 tree1 tree2
    in
      mlibMap tree
    end;

(*BasicDebug
val union = fn f => fn m1 => fn m2 =>
    checkInvariants "mlibKeyMap.union: result"
      (union f
         (checkInvariants "mlibKeyMap.union: input 1" m1)
         (checkInvariants "mlibKeyMap.union: input 2" m2));
*)

fun intersect f (mlibMap tree1) (mlibMap tree2) =
    let
      val tree = treeIntersect f tree1 tree2
    in
      mlibMap tree
    end;

(*BasicDebug
val intersect = fn f => fn m1 => fn m2 =>
    checkInvariants "mlibKeyMap.intersect: result"
      (intersect f
         (checkInvariants "mlibKeyMap.intersect: input 1" m1)
         (checkInvariants "mlibKeyMap.intersect: input 2" m2));
*)

(* ------------------------------------------------------------------------- *)
(* Iterators over maps.                                                      *)
(* ------------------------------------------------------------------------- *)

fun mkIterator (mlibMap tree) = treeMkIterator tree;

fun mkRevIterator (mlibMap tree) = treeMkRevIterator tree;

(* ------------------------------------------------------------------------- *)
(* Mapping and folding.                                                      *)
(* ------------------------------------------------------------------------- *)

fun mapPartial f (mlibMap tree) =
    let
      val tree = treeMapPartial f tree
    in
      mlibMap tree
    end;

(*BasicDebug
val mapPartial = fn f => fn m =>
    checkInvariants "mlibKeyMap.mapPartial: result"
      (mapPartial f (checkInvariants "mlibKeyMap.mapPartial: input" m));
*)

fun map f (mlibMap tree) =
    let
      val tree = treeMap f tree
    in
      mlibMap tree
    end;

(*BasicDebug
val map = fn f => fn m =>
    checkInvariants "mlibKeyMap.map: result"
      (map f (checkInvariants "mlibKeyMap.map: input" m));
*)

fun transform f = map (fn (_,value) => f value);

fun filter pred =
    let
      fun f (key_value as (_,value)) =
          if pred key_value then SOME value else NONE
    in
      mapPartial f
    end;

fun partition p =
    let
      fun np x = not (p x)
    in
      fn m => (filter p m, filter np m)
    end;

fun foldl f b m = foldIterator f b (mkIterator m);

fun foldr f b m = foldIterator f b (mkRevIterator m);

fun app f m = foldl (fn (key,value,()) => f (key,value)) () m;

(* ------------------------------------------------------------------------- *)
(* Searching.                                                                *)
(* ------------------------------------------------------------------------- *)

fun findl p m = findIterator p (mkIterator m);

fun findr p m = findIterator p (mkRevIterator m);

fun firstl f m = firstIterator f (mkIterator m);

fun firstr f m = firstIterator f (mkRevIterator m);

fun exists p m = Option.isSome (findl p m);

fun all p =
    let
      fun np x = not (p x)
    in
      fn m => not (exists np m)
    end;

fun count pred =
    let
      fun f (k,v,acc) = if pred (k,v) then acc + 1 else acc
    in
      foldl f 0
    end;

(* ------------------------------------------------------------------------- *)
(* Comparing.                                                                *)
(* ------------------------------------------------------------------------- *)

fun compare compareValue (m1,m2) =
    if pointerEqual (m1,m2) then EQUAL
    else
      case Int.compare (size m1, size m2) of
        LESS => LESS
      | EQUAL =>
        let
          val mlibMap _ = m1

          val io1 = mkIterator m1
          and io2 = mkIterator m2
        in
          compareIterator compareValue io1 io2
        end
      | GREATER => GREATER;

fun equal equalValue m1 m2 =
    pointerEqual (m1,m2) orelse
    (size m1 = size m2 andalso
     let
       val mlibMap _ = m1

       val io1 = mkIterator m1
       and io2 = mkIterator m2
     in
       equalIterator equalValue io1 io2
     end);

(* ------------------------------------------------------------------------- *)
(* Set operations on the domain.                                             *)
(* ------------------------------------------------------------------------- *)

fun unionDomain (mlibMap tree1) (mlibMap tree2) =
    let
      val tree = treeUnionDomain tree1 tree2
    in
      mlibMap tree
    end;

(*BasicDebug
val unionDomain = fn m1 => fn m2 =>
    checkInvariants "mlibKeyMap.unionDomain: result"
      (unionDomain
         (checkInvariants "mlibKeyMap.unionDomain: input 1" m1)
         (checkInvariants "mlibKeyMap.unionDomain: input 2" m2));
*)

local
  fun uncurriedUnionDomain (m,acc) = unionDomain acc m;
in
  fun unionListDomain ms =
      case ms of
        [] => raise Bug "mlibKeyMap.unionListDomain: no sets"
      | m :: ms => List.foldl uncurriedUnionDomain m ms;
end;

fun intersectDomain (mlibMap tree1) (mlibMap tree2) =
    let
      val tree = treeIntersectDomain tree1 tree2
    in
      mlibMap tree
    end;

(*BasicDebug
val intersectDomain = fn m1 => fn m2 =>
    checkInvariants "mlibKeyMap.intersectDomain: result"
      (intersectDomain
         (checkInvariants "mlibKeyMap.intersectDomain: input 1" m1)
         (checkInvariants "mlibKeyMap.intersectDomain: input 2" m2));
*)

local
  fun uncurriedIntersectDomain (m,acc) = intersectDomain acc m;
in
  fun intersectListDomain ms =
      case ms of
        [] => raise Bug "mlibKeyMap.intersectListDomain: no sets"
      | m :: ms => List.foldl uncurriedIntersectDomain m ms;
end;

fun differenceDomain (mlibMap tree1) (mlibMap tree2) =
    let
      val tree = treeDifferenceDomain tree1 tree2
    in
      mlibMap tree
    end;

(*BasicDebug
val differenceDomain = fn m1 => fn m2 =>
    checkInvariants "mlibKeyMap.differenceDomain: result"
      (differenceDomain
         (checkInvariants "mlibKeyMap.differenceDomain: input 1" m1)
         (checkInvariants "mlibKeyMap.differenceDomain: input 2" m2));
*)

fun symmetricDifferenceDomain m1 m2 =
    unionDomain (differenceDomain m1 m2) (differenceDomain m2 m1);

fun equalDomain m1 m2 = equal (K (K true)) m1 m2;

fun subsetDomain (mlibMap tree1) (mlibMap tree2) =
    treeSubsetDomain tree1 tree2;

fun disjointDomain m1 m2 = null (intersectDomain m1 m2);

(* ------------------------------------------------------------------------- *)
(* Converting to and from lists.                                             *)
(* ------------------------------------------------------------------------- *)

fun keys m = foldr (fn (key,_,l) => key :: l) [] m;

fun values m = foldr (fn (_,value,l) => value :: l) [] m;

fun toList m = foldr (fn (key,value,l) => (key,value) :: l) [] m;

fun fromList l =
    let
      val m = new ()
    in
      insertList m l
    end;

(* ------------------------------------------------------------------------- *)
(* Pretty-printing.                                                          *)
(* ------------------------------------------------------------------------- *)

fun toString m = "<" ^ (if null m then "" else Int.toString (size m)) ^ ">";

end

structure mlibIntMap =
mlibKeyMap (mlibIntOrdered);

structure mlibIntPairMap =
mlibKeyMap (mlibIntPairOrdered);

structure mlibStringMap =
mlibKeyMap (mlibStringOrdered);
(* ========================================================================= *)
(* FINITE SETS WITH A FIXED ELEMENT TYPE                                     *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibElementSet =
sig

(* ------------------------------------------------------------------------- *)
(* A type of set elements.                                                   *)
(* ------------------------------------------------------------------------- *)

type element

val compareElement : element * element -> order

val equalElement : element -> element -> bool

(* ------------------------------------------------------------------------- *)
(* A type of finite sets.                                                    *)
(* ------------------------------------------------------------------------- *)

type set

(* ------------------------------------------------------------------------- *)
(* Constructors.                                                             *)
(* ------------------------------------------------------------------------- *)

val empty : set

val singleton : element -> set

(* ------------------------------------------------------------------------- *)
(* Set size.                                                                 *)
(* ------------------------------------------------------------------------- *)

val null : set -> bool

val size : set -> int

(* ------------------------------------------------------------------------- *)
(* Querying.                                                                 *)
(* ------------------------------------------------------------------------- *)

val peek : set -> element -> element option

val member : element -> set -> bool

val pick : set -> element  (* an arbitrary element *)

val nth : set -> int -> element  (* in the range [0,size-1] *)

val random : set -> element

(* ------------------------------------------------------------------------- *)
(* Adding.                                                                   *)
(* ------------------------------------------------------------------------- *)

val add : set -> element -> set

val addList : set -> element list -> set

(* ------------------------------------------------------------------------- *)
(* Removing.                                                                 *)
(* ------------------------------------------------------------------------- *)

val delete : set -> element -> set  (* must be present *)

val remove : set -> element -> set

val deletePick : set -> element * set

val deleteNth : set -> int -> element * set

val deleteRandom : set -> element * set

(* ------------------------------------------------------------------------- *)
(* Joining.                                                                  *)
(* ------------------------------------------------------------------------- *)

val union : set -> set -> set

val unionList : set list -> set

val intersect : set -> set -> set

val intersectList : set list -> set

val difference : set -> set -> set

val symmetricDifference : set -> set -> set

(* ------------------------------------------------------------------------- *)
(* Mapping and folding.                                                      *)
(* ------------------------------------------------------------------------- *)

val filter : (element -> bool) -> set -> set

val partition : (element -> bool) -> set -> set * set

val app : (element -> unit) -> set -> unit

val foldl : (element * 's -> 's) -> 's -> set -> 's

val foldr : (element * 's -> 's) -> 's -> set -> 's

(* ------------------------------------------------------------------------- *)
(* Searching.                                                                *)
(* ------------------------------------------------------------------------- *)

val findl : (element -> bool) -> set -> element option

val findr : (element -> bool) -> set -> element option

val firstl : (element -> 'a option) -> set -> 'a option

val firstr : (element -> 'a option) -> set -> 'a option

val exists : (element -> bool) -> set -> bool

val all : (element -> bool) -> set -> bool

val count : (element -> bool) -> set -> int

(* ------------------------------------------------------------------------- *)
(* Comparing.                                                                *)
(* ------------------------------------------------------------------------- *)

val compare : set * set -> order

val equal : set -> set -> bool

val subset : set -> set -> bool

val disjoint : set -> set -> bool

(* ------------------------------------------------------------------------- *)
(* Closing under an operation.                                               *)
(* ------------------------------------------------------------------------- *)

val closedAdd : (element -> set) -> set -> set -> set

val close : (element -> set) -> set -> set

(* ------------------------------------------------------------------------- *)
(* Converting to and from lists.                                             *)
(* ------------------------------------------------------------------------- *)

val transform : (element -> 'a) -> set -> 'a list

val toList : set -> element list

val fromList : element list -> set

(* ------------------------------------------------------------------------- *)
(* Converting to and from maps.                                              *)
(* ------------------------------------------------------------------------- *)

type 'a map

val mapPartial : (element -> 'a option) -> set -> 'a map

val map : (element -> 'a) -> set -> 'a map

val domain : 'a map -> set

(* ------------------------------------------------------------------------- *)
(* Pretty-printing.                                                          *)
(* ------------------------------------------------------------------------- *)

val toString : set -> string

(* ------------------------------------------------------------------------- *)
(* Iterators over sets                                                       *)
(* ------------------------------------------------------------------------- *)

type iterator

val mkIterator : set -> iterator option

val mkRevIterator : set -> iterator option

val readIterator : iterator -> element

val advanceIterator : iterator -> iterator option

end
(* ========================================================================= *)
(* FINITE SETS WITH A FIXED ELEMENT TYPE                                     *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

functor mlibElementSet (
  KM : mlibKeyMap
) :> mlibElementSet
where type element = KM.key
and type 'a map = 'a KM.map =
struct

(* ------------------------------------------------------------------------- *)
(* A type of set elements.                                                   *)
(* ------------------------------------------------------------------------- *)

type element = KM.key;

val compareElement = KM.compareKey;

val equalElement = KM.equalKey;

(* ------------------------------------------------------------------------- *)
(* A type of finite sets.                                                    *)
(* ------------------------------------------------------------------------- *)

type 'a map = 'a KM.map;

datatype set = Set of unit map;

(* ------------------------------------------------------------------------- *)
(* Converting to and from maps.                                              *)
(* ------------------------------------------------------------------------- *)

fun dest (Set m) = m;

fun mapPartial f =
    let
      fun mf (elt,()) = f elt
    in
      fn Set m => KM.mapPartial mf m
    end;

fun map f =
    let
      fun mf (elt,()) = f elt
    in
      fn Set m => KM.map mf m
    end;

fun domain m = Set (KM.transform (fn _ => ()) m);

(* ------------------------------------------------------------------------- *)
(* Constructors.                                                             *)
(* ------------------------------------------------------------------------- *)

val empty = Set (KM.new ());

fun singleton elt = Set (KM.singleton (elt,()));

(* ------------------------------------------------------------------------- *)
(* Set size.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun null (Set m) = KM.null m;

fun size (Set m) = KM.size m;

(* ------------------------------------------------------------------------- *)
(* Querying.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun peek (Set m) elt =
    case KM.peekKey m elt of
      SOME (elt,()) => SOME elt
    | NONE => NONE;

fun member elt (Set m) = KM.inDomain elt m;

fun pick (Set m) =
    let
      val (elt,_) = KM.pick m
    in
      elt
    end;

fun nth (Set m) n =
    let
      val (elt,_) = KM.nth m n
    in
      elt
    end;

fun random (Set m) =
    let
      val (elt,_) = KM.random m
    in
      elt
    end;

(* ------------------------------------------------------------------------- *)
(* Adding.                                                                   *)
(* ------------------------------------------------------------------------- *)

fun add (Set m) elt =
    let
      val m = KM.insert m (elt,())
    in
      Set m
    end;

local
  fun uncurriedAdd (elt,set) = add set elt;
in
  fun addList set = List.foldl uncurriedAdd set;
end;

(* ------------------------------------------------------------------------- *)
(* Removing.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun delete (Set m) elt =
    let
      val m = KM.delete m elt
    in
      Set m
    end;

fun remove (Set m) elt =
    let
      val m = KM.remove m elt
    in
      Set m
    end;

fun deletePick (Set m) =
    let
      val ((elt,()),m) = KM.deletePick m
    in
      (elt, Set m)
    end;

fun deleteNth (Set m) n =
    let
      val ((elt,()),m) = KM.deleteNth m n
    in
      (elt, Set m)
    end;

fun deleteRandom (Set m) =
    let
      val ((elt,()),m) = KM.deleteRandom m
    in
      (elt, Set m)
    end;

(* ------------------------------------------------------------------------- *)
(* Joining.                                                                  *)
(* ------------------------------------------------------------------------- *)

fun union (Set m1) (Set m2) = Set (KM.unionDomain m1 m2);

fun unionList sets =
    let
      val ms = List.map dest sets
    in
      Set (KM.unionListDomain ms)
    end;

fun intersect (Set m1) (Set m2) = Set (KM.intersectDomain m1 m2);

fun intersectList sets =
    let
      val ms = List.map dest sets
    in
      Set (KM.intersectListDomain ms)
    end;

fun difference (Set m1) (Set m2) =
    Set (KM.differenceDomain m1 m2);

fun symmetricDifference (Set m1) (Set m2) =
    Set (KM.symmetricDifferenceDomain m1 m2);

(* ------------------------------------------------------------------------- *)
(* Mapping and folding.                                                      *)
(* ------------------------------------------------------------------------- *)

fun filter pred =
    let
      fun mpred (elt,()) = pred elt
    in
      fn Set m => Set (KM.filter mpred m)
    end;

fun partition pred =
    let
      fun mpred (elt,()) = pred elt
    in
      fn Set m =>
         let
           val (m1,m2) = KM.partition mpred m
         in
           (Set m1, Set m2)
         end
    end;

fun app f =
    let
      fun mf (elt,()) = f elt
    in
      fn Set m => KM.app mf m
    end;

fun foldl f =
    let
      fun mf (elt,(),acc) = f (elt,acc)
    in
      fn acc => fn Set m => KM.foldl mf acc m
    end;

fun foldr f =
    let
      fun mf (elt,(),acc) = f (elt,acc)
    in
      fn acc => fn Set m => KM.foldr mf acc m
    end;

(* ------------------------------------------------------------------------- *)
(* Searching.                                                                *)
(* ------------------------------------------------------------------------- *)

fun findl p =
    let
      fun mp (elt,()) = p elt
    in
      fn Set m =>
         case KM.findl mp m of
           SOME (elt,()) => SOME elt
         | NONE => NONE
    end;

fun findr p =
    let
      fun mp (elt,()) = p elt
    in
      fn Set m =>
         case KM.findr mp m of
           SOME (elt,()) => SOME elt
         | NONE => NONE
    end;

fun firstl f =
    let
      fun mf (elt,()) = f elt
    in
      fn Set m => KM.firstl mf m
    end;

fun firstr f =
    let
      fun mf (elt,()) = f elt
    in
      fn Set m => KM.firstr mf m
    end;

fun exists p =
    let
      fun mp (elt,()) = p elt
    in
      fn Set m => KM.exists mp m
    end;

fun all p =
    let
      fun mp (elt,()) = p elt
    in
      fn Set m => KM.all mp m
    end;

fun count p =
    let
      fun mp (elt,()) = p elt
    in
      fn Set m => KM.count mp m
    end;

(* ------------------------------------------------------------------------- *)
(* Comparing.                                                                *)
(* ------------------------------------------------------------------------- *)

fun compareValue ((),()) = EQUAL;

fun equalValue () () = true;

fun compare (Set m1, Set m2) = KM.compare compareValue (m1,m2);

fun equal (Set m1) (Set m2) = KM.equal equalValue m1 m2;

fun subset (Set m1) (Set m2) = KM.subsetDomain m1 m2;

fun disjoint (Set m1) (Set m2) = KM.disjointDomain m1 m2;

(* ------------------------------------------------------------------------- *)
(* Closing under an operation.                                               *)
(* ------------------------------------------------------------------------- *)

fun closedAdd f =
    let
      fun adds acc set = foldl check acc set

      and check (elt,acc) =
          if member elt acc then acc
          else expand (add acc elt) elt

      and expand acc elt = adds acc (f elt)
    in
      adds
    end;

fun close f = closedAdd f empty;

(* ------------------------------------------------------------------------- *)
(* Converting to and from lists.                                             *)
(* ------------------------------------------------------------------------- *)

fun transform f =
    let
      fun inc (x,l) = f x :: l
    in
      foldr inc []
    end;

fun toList (Set m) = KM.keys m;

fun fromList elts = addList empty elts;

(* ------------------------------------------------------------------------- *)
(* Pretty-printing.                                                          *)
(* ------------------------------------------------------------------------- *)

fun toString set =
    "{" ^ (if null set then "" else Int.toString (size set)) ^ "}";

(* ------------------------------------------------------------------------- *)
(* Iterators over sets                                                       *)
(* ------------------------------------------------------------------------- *)

type iterator = unit KM.iterator;

fun mkIterator (Set m) = KM.mkIterator m;

fun mkRevIterator (Set m) = KM.mkRevIterator m;

fun readIterator iter =
    let
      val (elt,()) = KM.readIterator iter
    in
      elt
    end;

fun advanceIterator iter = KM.advanceIterator iter;

end

structure mlibIntSet =
mlibElementSet (mlibIntMap);

structure mlibIntPairSet =
mlibElementSet (mlibIntPairMap);

structure mlibStringSet =
mlibElementSet (mlibStringMap);
(* ========================================================================= *)
(* PRETTY-PRINTING                                                           *)
(* Copyright (c) 2008 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibPrint =
sig

(* ------------------------------------------------------------------------- *)
(* Escaping strings.                                                         *)
(* ------------------------------------------------------------------------- *)

val escapeString : {escape : char list} -> string -> string

(* ------------------------------------------------------------------------- *)
(* A type of pretty-printers.                                                *)
(* ------------------------------------------------------------------------- *)

type ppstream

type 'a pp = 'a -> ppstream

val skip : ppstream

val sequence : ppstream -> ppstream -> ppstream

val duplicate : int -> ppstream -> ppstream

val program : ppstream list -> ppstream

val stream : ppstream mlibStream.stream -> ppstream

val ppPpstream : ppstream pp

(* ------------------------------------------------------------------------- *)
(* Pretty-printing blocks.                                                   *)
(* ------------------------------------------------------------------------- *)

datatype style = Consistent | Inconsistent

datatype block =
    Block of
      {style : style,
       indent : int}

val styleBlock : block -> style

val indentBlock : block -> int

val block : block -> ppstream -> ppstream

val consistentBlock : int -> ppstream list -> ppstream

val inconsistentBlock : int -> ppstream list -> ppstream

(* ------------------------------------------------------------------------- *)
(* Words are unbreakable strings annotated with their effective size.        *)
(* ------------------------------------------------------------------------- *)

datatype word = Word of {word : string, size : int}

val mkWord : string -> word

val emptyWord : word

val charWord : char -> word

val ppWord : word pp

val space : ppstream

val spaces : int -> ppstream

(* ------------------------------------------------------------------------- *)
(* Possible line breaks.                                                     *)
(* ------------------------------------------------------------------------- *)

datatype break = Break of {size : int, extraIndent : int}

val mkBreak : int -> break

val ppBreak : break pp

val break : ppstream

val breaks : int -> ppstream

(* ------------------------------------------------------------------------- *)
(* Forced line breaks.                                                       *)
(* ------------------------------------------------------------------------- *)

val newline : ppstream

val newlines : int -> ppstream

(* ------------------------------------------------------------------------- *)
(* Pretty-printer combinators.                                               *)
(* ------------------------------------------------------------------------- *)

val ppMap : ('a -> 'b) -> 'b pp -> 'a pp

val ppBracket : string -> string -> 'a pp -> 'a pp

val ppOp : string -> ppstream

val ppOp2 : string -> 'a pp -> 'b pp -> ('a * 'b) pp

val ppOp3 : string -> string -> 'a pp -> 'b pp -> 'c pp -> ('a * 'b * 'c) pp

val ppOpList : string -> 'a pp -> 'a list pp

val ppOpStream : string -> 'a pp -> 'a mlibStream.stream pp

(* ------------------------------------------------------------------------- *)
(* Pretty-printers for common types.                                         *)
(* ------------------------------------------------------------------------- *)

val ppChar : char pp

val ppString : string pp

val ppEscapeString : {escape : char list} -> string pp

val ppUnit : unit pp

val ppBool : bool pp

val ppInt : int pp

val ppPrettyInt : int pp

val ppReal : real pp

val ppPercent : real pp

val ppOrder : order pp

val ppList : 'a pp -> 'a list pp

val ppStream : 'a pp -> 'a mlibStream.stream pp

val ppOption : 'a pp -> 'a option pp

val ppPair : 'a pp -> 'b pp -> ('a * 'b) pp

val ppTriple : 'a pp -> 'b pp -> 'c pp -> ('a * 'b * 'c) pp

val ppException : exn pp

(* ------------------------------------------------------------------------- *)
(* Pretty-printing infix operators.                                          *)
(* ------------------------------------------------------------------------- *)

type token = string

datatype assoc =
    LeftAssoc
  | NonAssoc
  | RightAssoc

datatype infixes =
    Infixes of
      {token : token,
       precedence : int,
       assoc : assoc} list

val tokensInfixes : infixes -> mlibStringSet.set

val layerInfixes : infixes -> {tokens : mlibStringSet.set, assoc : assoc} list

val ppInfixes :
    infixes ->
    ('a -> (token * 'a * 'a) option) -> ('a * token) pp ->
    ('a * bool) pp -> ('a * bool) pp

(* ------------------------------------------------------------------------- *)
(* Pretty-printer rendering.                                                 *)
(* ------------------------------------------------------------------------- *)

val render :
    {lineLength : int option} -> ppstream ->
    {indent : int, line : string} mlibStream.stream

val toStringWithLineLength :
    {lineLength : int option} -> 'a pp -> 'a -> string

val toStreamWithLineLength :
    {lineLength : int option} -> 'a pp -> 'a -> string mlibStream.stream

val toLine : 'a pp -> 'a -> string

(* ------------------------------------------------------------------------- *)
(* Pretty-printer rendering with a global line length.                       *)
(* ------------------------------------------------------------------------- *)

val lineLength : int ref

val toString : 'a pp -> 'a -> string

val toStream : 'a pp -> 'a -> string mlibStream.stream

val trace : 'a pp -> string -> 'a -> unit

end
(* ========================================================================= *)
(* PRETTY-PRINTING                                                           *)
(* Copyright (c) 2008 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibPrint :> mlibPrint =
struct

open mlibUseful;

(* ------------------------------------------------------------------------- *)
(* Constants.                                                                *)
(* ------------------------------------------------------------------------- *)

val initialLineLength = 75;

(* ------------------------------------------------------------------------- *)
(* Helper functions.                                                         *)
(* ------------------------------------------------------------------------- *)

fun revAppend xs s =
    case xs of
      [] => s ()
    | x :: xs => revAppend xs (K (mlibStream.Cons (x,s)));

fun revConcat strm =
    case strm of
      mlibStream.Nil => mlibStream.Nil
    | mlibStream.Cons (h,t) => revAppend h (fn () => revConcat (t ()));

local
  fun calcSpaces n = nChars #" " n;

  val cacheSize = 2 * initialLineLength;

  val cachedSpaces = Vector.tabulate (cacheSize,calcSpaces);
in
  fun nSpaces n =
      if n < cacheSize then Vector.sub (cachedSpaces,n)
      else calcSpaces n;
end;

(* ------------------------------------------------------------------------- *)
(* Escaping strings.                                                         *)
(* ------------------------------------------------------------------------- *)

fun escapeString {escape} =
    let
      val escapeMap = List.map (fn c => (c, "\\" ^ str c)) escape

      fun escapeChar c =
          case c of
            #"\\" => "\\\\"
          | #"\n" => "\\n"
          | #"\t" => "\\t"
          | _ =>
            case List.find (equal c o fst) escapeMap of
              SOME (_,s) => s
            | NONE => str c
    in
      String.translate escapeChar
    end;

(* ------------------------------------------------------------------------- *)
(* Pretty-printing blocks.                                                   *)
(* ------------------------------------------------------------------------- *)

datatype style = Consistent | Inconsistent;

datatype block =
    Block of
      {style : style,
       indent : int};

fun toStringStyle style =
    case style of
      Consistent => "Consistent"
    | Inconsistent => "Inconsistent";

fun isConsistentStyle style =
    case style of
      Consistent => true
    | Inconsistent => false;

fun isInconsistentStyle style =
    case style of
      Inconsistent => true
    | Consistent => false;

fun mkBlock style indent =
    Block
      {style = style,
       indent = indent};

val mkConsistentBlock = mkBlock Consistent;

val mkInconsistentBlock = mkBlock Inconsistent;

fun styleBlock (Block {style = x, ...}) = x;

fun indentBlock (Block {indent = x, ...}) = x;

fun isConsistentBlock block = isConsistentStyle (styleBlock block);

fun isInconsistentBlock block = isInconsistentStyle (styleBlock block);

(* ------------------------------------------------------------------------- *)
(* Words are unbreakable strings annotated with their effective size.        *)
(* ------------------------------------------------------------------------- *)

datatype word = Word of {word : string, size : int};

fun mkWord s = Word {word = s, size = String.size s};

val emptyWord = mkWord "";

fun charWord c = mkWord (str c);

fun spacesWord i = Word {word = nSpaces i, size = i};

fun sizeWord (Word {size = x, ...}) = x;

fun renderWord (Word {word = x, ...}) = x;

(* ------------------------------------------------------------------------- *)
(* Possible line breaks.                                                     *)
(* ------------------------------------------------------------------------- *)

datatype break = Break of {size : int, extraIndent : int};

fun mkBreak n = Break {size = n, extraIndent = 0};

fun sizeBreak (Break {size = x, ...}) = x;

fun extraIndentBreak (Break {extraIndent = x, ...}) = x;

fun renderBreak b = nSpaces (sizeBreak b);

fun updateSizeBreak size break =
    let
      val Break {size = _, extraIndent} = break
    in
      Break
        {size = size,
         extraIndent = extraIndent}
    end;

fun appendBreak break1 break2 =
    let
(*BasicDebug
      val () = warn "merging consecutive pretty-printing breaks"
*)
      val Break {size = size1, extraIndent = extraIndent1} = break1
      and Break {size = size2, extraIndent = extraIndent2} = break2

      val size = size1 + size2
      and extraIndent = Int.max (extraIndent1,extraIndent2)
    in
      Break
        {size = size,
         extraIndent = extraIndent}
    end;

(* ------------------------------------------------------------------------- *)
(* A type of pretty-printers.                                                *)
(* ------------------------------------------------------------------------- *)

datatype step =
    BeginBlock of block
  | EndBlock
  | AddWord of word
  | AddBreak of break
  | AddNewline;

type ppstream = step mlibStream.stream;

type 'a pp = 'a -> ppstream;

fun toStringStep step =
    case step of
      BeginBlock _ => "BeginBlock"
    | EndBlock => "EndBlock"
    | AddWord _ => "Word"
    | AddBreak _ => "Break"
    | AddNewline => "Newline";

val skip : ppstream = mlibStream.Nil;

fun sequence pp1 pp2 : ppstream = mlibStream.append pp1 (K pp2);

local
  fun dup pp n () = if n = 1 then pp else mlibStream.append pp (dup pp (n - 1));
in
  fun duplicate n pp : ppstream =
      let
(*BasicDebug
        val () = if 0 <= n then () else raise Bug "mlibPrint.duplicate"
*)
      in
        if n = 0 then skip else dup pp n ()
      end;
end;

val program : ppstream list -> ppstream = mlibStream.concatList;

val stream : ppstream mlibStream.stream -> ppstream = mlibStream.concat;

(* ------------------------------------------------------------------------- *)
(* Pretty-printing blocks.                                                   *)
(* ------------------------------------------------------------------------- *)

local
  fun beginBlock b = mlibStream.singleton (BeginBlock b);

  val endBlock = mlibStream.singleton EndBlock;
in
  fun block b pp = program [beginBlock b, pp, endBlock];
end;

fun consistentBlock i pps = block (mkConsistentBlock i) (program pps);

fun inconsistentBlock i pps = block (mkInconsistentBlock i) (program pps);

(* ------------------------------------------------------------------------- *)
(* Words are unbreakable strings annotated with their effective size.        *)
(* ------------------------------------------------------------------------- *)

fun ppWord w = mlibStream.singleton (AddWord w);

val space = ppWord (mkWord " ");

fun spaces i = ppWord (spacesWord i);

(* ------------------------------------------------------------------------- *)
(* Possible line breaks.                                                     *)
(* ------------------------------------------------------------------------- *)

fun ppBreak b = mlibStream.singleton (AddBreak b);

fun breaks i = ppBreak (mkBreak i);

val break = breaks 1;

(* ------------------------------------------------------------------------- *)
(* Forced line breaks.                                                       *)
(* ------------------------------------------------------------------------- *)

val newline = mlibStream.singleton AddNewline;

fun newlines i = duplicate i newline;

(* ------------------------------------------------------------------------- *)
(* Pretty-printer combinators.                                               *)
(* ------------------------------------------------------------------------- *)

fun ppMap f ppB a : ppstream = ppB (f a);

fun ppBracket' l r ppA a =
    let
      val n = sizeWord l
    in
      inconsistentBlock n
        [ppWord l,
         ppA a,
         ppWord r]
    end;

fun ppOp' w = sequence (ppWord w) break;

fun ppOp2' ab ppA ppB (a,b) =
    inconsistentBlock 0
      [ppA a,
       ppOp' ab,
       ppB b];

fun ppOp3' ab bc ppA ppB ppC (a,b,c) =
    inconsistentBlock 0
      [ppA a,
       ppOp' ab,
       ppB b,
       ppOp' bc,
       ppC c];

fun ppOpList' s ppA =
    let
      fun ppOpA a = sequence (ppOp' s) (ppA a)
    in
      fn [] => skip
       | h :: t => inconsistentBlock 0 (ppA h :: List.map ppOpA t)
    end;

fun ppOpStream' s ppA =
    let
      fun ppOpA a = sequence (ppOp' s) (ppA a)
    in
      fn mlibStream.Nil => skip
       | mlibStream.Cons (h,t) =>
         inconsistentBlock 0
           [ppA h,
            mlibStream.concat (mlibStream.map ppOpA (t ()))]
    end;

fun ppBracket l r = ppBracket' (mkWord l) (mkWord r);

fun ppOp s = ppOp' (mkWord s);

fun ppOp2 ab = ppOp2' (mkWord ab);

fun ppOp3 ab bc = ppOp3' (mkWord ab) (mkWord bc);

fun ppOpList s = ppOpList' (mkWord s);

fun ppOpStream s = ppOpStream' (mkWord s);

(* ------------------------------------------------------------------------- *)
(* Pretty-printers for common types.                                         *)
(* ------------------------------------------------------------------------- *)

fun ppChar c = ppWord (charWord c);

fun ppString s = ppWord (mkWord s);

fun ppEscapeString escape = ppMap (escapeString escape) ppString;

local
  val pp = ppString "()";
in
  fun ppUnit () = pp;
end;

local
  val ppTrue = ppString "true"
  and ppFalse = ppString "false";
in
  fun ppBool b = if b then ppTrue else ppFalse;
end;

val ppInt = ppMap Int.toString ppString;

local
  val ppNeg = ppString "~"
  and ppSep = ppString ","
  and ppZero = ppString "0"
  and ppZeroZero = ppString "00";

  fun ppIntBlock i =
      if i < 10 then sequence ppZeroZero (ppInt i)
      else if i < 100 then sequence ppZero (ppInt i)
      else ppInt i;

  fun ppIntBlocks i =
      if i < 1000 then ppInt i
      else sequence (ppIntBlocks (i div 1000))
             (sequence ppSep (ppIntBlock (i mod 1000)));
in
  fun ppPrettyInt i =
      if i < 0 then sequence ppNeg (ppIntBlocks (~i))
      else ppIntBlocks i;
end;

val ppReal = ppMap Real.toString ppString;

val ppPercent = ppMap percentToString ppString;

local
  val ppLess = ppString "Less"
  and ppEqual = ppString "Equal"
  and ppGreater = ppString "Greater";
in
  fun ppOrder ord =
      case ord of
        LESS => ppLess
      | EQUAL => ppEqual
      | GREATER => ppGreater;
end;

local
  val left = mkWord "["
  and right = mkWord "]"
  and sep = mkWord ",";
in
  fun ppList ppX xs = ppBracket' left right (ppOpList' sep ppX) xs;

  fun ppStream ppX xs = ppBracket' left right (ppOpStream' sep ppX) xs;
end;

local
  val ppNone = ppString "-";
in
  fun ppOption ppX xo =
      case xo of
        SOME x => ppX x
      | NONE => ppNone;
end;

local
  val left = mkWord "("
  and right = mkWord ")"
  and sep = mkWord ",";
in
  fun ppPair ppA ppB =
      ppBracket' left right (ppOp2' sep ppA ppB);

  fun ppTriple ppA ppB ppC =
      ppBracket' left right (ppOp3' sep sep ppA ppB ppC);
end;

fun ppException e = ppString (exnMessage e);

(* ------------------------------------------------------------------------- *)
(* A type of pretty-printers.                                                *)
(* ------------------------------------------------------------------------- *)

local
  val ppStepType = ppMap toStringStep ppString;

  val ppStyle = ppMap toStringStyle ppString;

  val ppBlockInfo =
      let
        val sep = mkWord " "
      in
        fn Block {style = s, indent = i} =>
           program [ppStyle s, ppWord sep, ppInt i]
      end;

  val ppWordInfo =
      let
        val left = mkWord "\""
        and right = mkWord "\""
        and escape = {escape = [#"\""]}

        val pp = ppBracket' left right (ppEscapeString escape)
      in
        fn Word {word = s, size = n} =>
           if size s = n then pp s else ppPair pp ppInt (s,n)
      end;

  val ppBreakInfo =
      let
        val sep = mkWord "+"
      in
        fn Break {size = n, extraIndent = k} =>
           if k = 0 then ppInt n else program [ppInt n, ppWord sep, ppInt k]
      end;

  fun ppStep step =
      inconsistentBlock 2
        (ppStepType step ::
         (case step of
            BeginBlock b =>
              [break,
               ppBlockInfo b]
          | EndBlock => []
          | AddWord w =>
              [break,
               ppWordInfo w]
          | AddBreak b =>
              [break,
               ppBreakInfo b]
          | AddNewline => []));
in
  val ppPpstream = ppStream ppStep;
end;

(* ------------------------------------------------------------------------- *)
(* Pretty-printing infix operators.                                          *)
(* ------------------------------------------------------------------------- *)

type token = string;

datatype assoc =
    LeftAssoc
  | NonAssoc
  | RightAssoc;

datatype infixes =
    Infixes of
      {token : token,
       precedence : int,
       assoc : assoc} list;

local
  fun comparePrecedence (io1,io2) =
      let
        val {token = _, precedence = p1, assoc = _} = io1
        and {token = _, precedence = p2, assoc = _} = io2
      in
        Int.compare (p2,p1)
      end;

  fun equalAssoc a a' =
      case a of
        LeftAssoc => (case a' of LeftAssoc => true | _ => false)
      | NonAssoc => (case a' of NonAssoc => true | _ => false)
      | RightAssoc => (case a' of RightAssoc => true | _ => false);

  fun new t a acc = {tokens = mlibStringSet.singleton t, assoc = a} :: acc;

  fun add t a' acc =
      case acc of
        [] => raise Bug "mlibPrint.layerInfixes: null"
      | {tokens = ts, assoc = a} :: acc =>
        if equalAssoc a a' then {tokens = mlibStringSet.add ts t, assoc = a} :: acc
        else raise Bug "mlibPrint.layerInfixes: mixed assocs";

  fun layer ({token = t, precedence = p, assoc = a}, (acc,p')) =
      let
        val acc = if p = p' then add t a acc else new t a acc
      in
        (acc,p)
      end;
in
  fun layerInfixes (Infixes ios) =
      case sort comparePrecedence ios of
        [] => []
      | {token = t, precedence = p, assoc = a} :: ios =>
        let
          val acc = new t a []

          val (acc,_) = List.foldl layer (acc,p) ios
        in
          acc
        end;
end;

local
  fun add ({tokens = ts, assoc = _}, tokens) = mlibStringSet.union ts tokens;
in
  fun tokensLayeredInfixes l = List.foldl add mlibStringSet.empty l;
end;

fun tokensInfixes ios = tokensLayeredInfixes (layerInfixes ios);

fun destInfixOp dest tokens tm =
    case dest tm of
      NONE => NONE
    | s as SOME (t,a,b) => if mlibStringSet.member t tokens then s else NONE;

fun ppLayeredInfixes dest ppTok {tokens,assoc} ppLower ppSub =
    let
      fun isLayer t = mlibStringSet.member t tokens

      fun ppTerm aligned (tm,r) =
          case dest tm of
            NONE => ppSub (tm,r)
          | SOME (t,a,b) =>
            if aligned andalso isLayer t then ppLayer (tm,t,a,b,r)
            else ppLower (tm,t,a,b,r)

      and ppLeft tm_r =
          let
            val aligned = case assoc of LeftAssoc => true | _ => false
          in
            ppTerm aligned tm_r
          end

      and ppLayer (tm,t,a,b,r) =
          program
            [ppLeft (a,true),
             ppTok (tm,t),
             ppRight (b,r)]

      and ppRight tm_r =
          let
            val aligned = case assoc of RightAssoc => true | _ => false
          in
            ppTerm aligned tm_r
          end
    in
      fn tm_t_a_b_r as (_,t,_,_,_) =>
         if isLayer t then inconsistentBlock 0 [ppLayer tm_t_a_b_r]
         else ppLower tm_t_a_b_r
    end;

local
  val leftBrack = mkWord "("
  and rightBrack = mkWord ")";
in
  fun ppInfixes ops =
      let
        val layers = layerInfixes ops

        val toks = tokensLayeredInfixes layers
      in
        fn dest => fn ppTok => fn ppSub =>
           let
             fun destOp tm = destInfixOp dest toks tm

             fun ppInfix tm_t_a_b_r = ppLayers layers tm_t_a_b_r

             and ppLayers ls (tm,t,a,b,r) =
                 case ls of
                   [] =>
                   ppBracket' leftBrack rightBrack ppInfix (tm,t,a,b,false)
                 | l :: ls =>
                   let
                     val ppLower = ppLayers ls
                   in
                     ppLayeredInfixes destOp ppTok l ppLower ppSub (tm,t,a,b,r)
                   end
           in
             fn (tm,r) =>
                case destOp tm of
                  SOME (t,a,b) => ppInfix (tm,t,a,b,r)
                | NONE => ppSub (tm,r)
           end
      end;
end;

(* ------------------------------------------------------------------------- *)
(* A type of output lines.                                                   *)
(* ------------------------------------------------------------------------- *)

type line = {indent : int, line : string};

val emptyLine =
    let
      val indent = 0
      and line = ""
    in
      {indent = indent,
       line = line}
    end;

fun addEmptyLine lines = emptyLine :: lines;

fun addLine lines indent line = {indent = indent, line = line} :: lines;

(* ------------------------------------------------------------------------- *)
(* Pretty-printer rendering.                                                 *)
(* ------------------------------------------------------------------------- *)

datatype chunk =
    WordChunk of word
  | BreakChunk of break
  | FrameChunk of frame

and frame =
    Frame of
      {block : block,
       broken : bool,
       indent : int,
       size : int,
       chunks : chunk list};

datatype state =
    State of
      {lineIndent : int,
       lineSize : int,
       stack : frame list};

fun blockFrame (Frame {block = x, ...}) = x;

fun brokenFrame (Frame {broken = x, ...}) = x;

fun indentFrame (Frame {indent = x, ...}) = x;

fun sizeFrame (Frame {size = x, ...}) = x;

fun chunksFrame (Frame {chunks = x, ...}) = x;

fun styleFrame frame = styleBlock (blockFrame frame);

fun isConsistentFrame frame = isConsistentBlock (blockFrame frame);

fun breakingFrame frame = isConsistentFrame frame andalso brokenFrame frame;

fun sizeChunk chunk =
    case chunk of
      WordChunk w => sizeWord w
    | BreakChunk b => sizeBreak b
    | FrameChunk f => sizeFrame f;

local
  fun add (c,acc) = sizeChunk c + acc;
in
  fun sizeChunks cs = List.foldl add 0 cs;
end;

local
  fun flattenChunks acc chunks =
      case chunks of
        [] => acc
      | chunk :: chunks => flattenChunk acc chunk chunks

  and flattenChunk acc chunk chunks =
      case chunk of
        WordChunk w => flattenChunks (renderWord w :: acc) chunks
      | BreakChunk b => flattenChunks (renderBreak b :: acc) chunks
      | FrameChunk f => flattenFrame acc f chunks

  and flattenFrame acc frame chunks =
      flattenChunks acc (chunksFrame frame @ chunks);
in
  fun renderChunks chunks = String.concat (flattenChunks [] chunks);
end;

fun addChunksLine lines indent chunks =
    addLine lines indent (renderChunks chunks);

local
  fun add baseIndent ((extraIndent,chunks),lines) =
      addChunksLine lines (baseIndent + extraIndent) chunks;
in
  fun addIndentChunksLines lines baseIndent indent_chunks =
      List.foldl (add baseIndent) lines indent_chunks;
end;

fun isEmptyFrame frame =
    let
      val chunks = chunksFrame frame

      val empty = List.null chunks

(*BasicDebug
      val () =
          if not empty orelse sizeFrame frame = 0 then ()
          else raise Bug "mlibPrint.isEmptyFrame: bad size"
*)
    in
      empty
    end;

local
  fun breakInconsistent blockIndent =
      let
        fun break chunks =
            case chunks of
              [] => NONE
            | chunk :: chunks =>
              case chunk of
                BreakChunk b =>
                let
                  val pre = chunks
                  and indent = blockIndent + extraIndentBreak b
                  and post = []
                in
                  SOME (pre,indent,post)
                end
              | _ =>
                case break chunks of
                  SOME (pre,indent,post) =>
                  let
                    val post = chunk :: post
                  in
                    SOME (pre,indent,post)
                  end
                | NONE => NONE
      in
        break
      end;

  fun breakConsistent blockIndent =
      let
        fun break indent_chunks chunks =
            case breakInconsistent blockIndent chunks of
              NONE => (chunks,indent_chunks)
            | SOME (pre,indent,post) =>
              break ((indent,post) :: indent_chunks) pre
      in
        break []
      end;
in
  fun breakFrame frame =
      let
        val Frame
              {block,
               broken = _,
               indent = _,
               size = _,
               chunks} = frame

        val blockIndent = indentBlock block
      in
        case breakInconsistent blockIndent chunks of
          NONE => NONE
        | SOME (pre,indent,post) =>
          let
            val broken = true
            and size = sizeChunks post

            val frame =
                Frame
                  {block = block,
                   broken = broken,
                   indent = indent,
                   size = size,
                   chunks = post}
          in
            case styleBlock block of
              Inconsistent =>
              let
                val indent_chunks = []
              in
                SOME (pre,indent_chunks,frame)
              end
            | Consistent =>
              let
                val (pre,indent_chunks) = breakConsistent blockIndent pre
              in
                SOME (pre,indent_chunks,frame)
              end
          end
      end;
end;

fun removeChunksFrame frame =
    let
      val Frame
            {block,
             broken,
             indent,
             size = _,
             chunks} = frame
    in
      if broken andalso List.null chunks then NONE
      else
        let
          val frame =
              Frame
                {block = block,
                 broken = true,
                 indent = indent,
                 size = 0,
                 chunks = []}
        in
          SOME (chunks,frame)
        end
    end;

val removeChunksFrames =
    let
      fun remove frames =
          case frames of
            [] =>
            let
              val chunks = []
              and frames = NONE
              and indent = 0
            in
              (chunks,frames,indent)
            end
          | top :: rest =>
            let
              val (chunks,rest',indent) = remove rest

              val indent = indent + indentFrame top

              val (chunks,top') =
                  case removeChunksFrame top of
                    NONE => (chunks,NONE)
                  | SOME (topChunks,top) => (topChunks @ chunks, SOME top)

              val frames' =
                  case (top',rest') of
                    (NONE,NONE) => NONE
                  | (SOME top, NONE) => SOME (top :: rest)
                  | (NONE, SOME rest) => SOME (top :: rest)
                  | (SOME top, SOME rest) => SOME (top :: rest)
            in
              (chunks,frames',indent)
            end
    in
      fn frames =>
         let
           val (chunks,frames',indent) = remove frames

           val frames = Option.getOpt (frames',frames)
         in
           (chunks,frames,indent)
         end
    end;

local
  fun breakUp lines lineIndent frames =
      case frames of
        [] => NONE
      | frame :: frames =>
        case breakUp lines lineIndent frames of
          SOME (lines_indent,lineSize,frames) =>
          let
            val lineSize = lineSize + sizeFrame frame
            and frames = frame :: frames
          in
            SOME (lines_indent,lineSize,frames)
          end
        | NONE =>
          case breakFrame frame of
            NONE => NONE
          | SOME (frameChunks,indent_chunks,frame) =>
            let
              val (chunks,frames,indent) = removeChunksFrames frames

              val chunks = frameChunks @ chunks

              val lines = addChunksLine lines lineIndent chunks

              val lines = addIndentChunksLines lines indent indent_chunks

              val lineIndent = indent + indentFrame frame

              val lineSize = sizeFrame frame

              val frames = frame :: frames
            in
              SOME ((lines,lineIndent),lineSize,frames)
            end;

  fun breakInsideChunk chunk =
      case chunk of
        WordChunk _ => NONE
      | BreakChunk _ => raise Bug "mlibPrint.breakInsideChunk"
      | FrameChunk frame =>
        case breakFrame frame of
          SOME (pathChunks,indent_chunks,frame) =>
          let
            val pathIndent = 0
            and breakIndent = indentFrame frame
          in
            SOME (pathChunks,pathIndent,indent_chunks,breakIndent,frame)
          end
        | NONE => breakInsideFrame frame

  and breakInsideChunks chunks =
      case chunks of
        [] => NONE
      | chunk :: chunks =>
        case breakInsideChunk chunk of
          SOME (pathChunks,pathIndent,indent_chunks,breakIndent,frame) =>
          let
            val pathChunks = pathChunks @ chunks
            and chunks = [FrameChunk frame]
          in
            SOME (pathChunks,pathIndent,indent_chunks,breakIndent,chunks)
          end
        | NONE =>
          case breakInsideChunks chunks of
            SOME (pathChunks,pathIndent,indent_chunks,breakIndent,chunks) =>
            let
              val chunks = chunk :: chunks
            in
              SOME (pathChunks,pathIndent,indent_chunks,breakIndent,chunks)
            end
          | NONE => NONE

  and breakInsideFrame frame =
      let
        val Frame
              {block,
               broken = _,
               indent,
               size = _,
               chunks} = frame
      in
        case breakInsideChunks chunks of
          SOME (pathChunks,pathIndent,indent_chunks,breakIndent,chunks) =>
          let
            val pathIndent = pathIndent + indent
            and broken = true
            and size = sizeChunks chunks

            val frame =
                Frame
                  {block = block,
                   broken = broken,
                   indent = indent,
                   size = size,
                   chunks = chunks}
          in
            SOME (pathChunks,pathIndent,indent_chunks,breakIndent,frame)
          end
        | NONE => NONE
      end;

  fun breakInside lines lineIndent frames =
      case frames of
        [] => NONE
      | frame :: frames =>
        case breakInsideFrame frame of
          SOME (pathChunks,pathIndent,indent_chunks,breakIndent,frame) =>
          let
            val (chunks,frames,indent) = removeChunksFrames frames

            val chunks = pathChunks @ chunks
            and indent = indent + pathIndent

            val lines = addChunksLine lines lineIndent chunks

            val lines = addIndentChunksLines lines indent indent_chunks

            val lineIndent = indent + breakIndent

            val lineSize = sizeFrame frame

            val frames = frame :: frames
          in
            SOME ((lines,lineIndent),lineSize,frames)
          end
        | NONE =>
          case breakInside lines lineIndent frames of
            SOME (lines_indent,lineSize,frames) =>
            let
              val lineSize = lineSize + sizeFrame frame
              and frames = frame :: frames
            in
              SOME (lines_indent,lineSize,frames)
            end
          | NONE => NONE;
in
  fun breakFrames lines lineIndent frames =
      case breakUp lines lineIndent frames of
        SOME ((lines,lineIndent),lineSize,frames) =>
        SOME (lines,lineIndent,lineSize,frames)
      | NONE =>
        case breakInside lines lineIndent frames of
          SOME ((lines,lineIndent),lineSize,frames) =>
          SOME (lines,lineIndent,lineSize,frames)
        | NONE => NONE;
end;

(*BasicDebug
fun checkChunk chunk =
    case chunk of
      WordChunk t => (false, sizeWord t)
    | BreakChunk b => (false, sizeBreak b)
    | FrameChunk b => checkFrame b

and checkChunks chunks =
    case chunks of
      [] => (false,0)
    | chunk :: chunks =>
      let
        val (bk,sz) = checkChunk chunk

        val (bk',sz') = checkChunks chunks
      in
        (bk orelse bk', sz + sz')
      end

and checkFrame frame =
    let
      val Frame
            {block = _,
             broken,
             indent = _,
             size,
             chunks} = frame

      val (bk,sz) = checkChunks chunks

      val () =
          if size = sz then ()
          else raise Bug "mlibPrint.checkFrame: wrong size"

      val () =
          if broken orelse not bk then ()
          else raise Bug "mlibPrint.checkFrame: deep broken frame"
    in
      (broken,size)
    end;

fun checkFrames frames =
    case frames of
      [] => (true,0)
    | frame :: frames =>
      let
        val (bk,sz) = checkFrame frame

        val (bk',sz') = checkFrames frames

        val () =
            if bk' orelse not bk then ()
            else raise Bug "mlibPrint.checkFrame: broken stack frame"
      in
        (bk, sz + sz')
      end;
*)

(*BasicDebug
fun checkState state =
    (let
       val State {lineIndent,lineSize,stack} = state

       val () =
           if not (List.null stack) then ()
           else raise Error "no stack"

       val (_,sz) = checkFrames stack

       val () =
           if lineSize = sz then ()
           else raise Error "wrong lineSize"
     in
       ()
     end
     handle Error err => raise Bug err)
    handle Bug bug => raise Bug ("mlibPrint.checkState: " ^ bug);
*)

fun isEmptyState state =
    let
      val State {lineSize,stack,...} = state
    in
      lineSize = 0 andalso List.all isEmptyFrame stack
    end;

fun isFinalState state =
    let
      val State {stack,...} = state
    in
      case stack of
        [] => raise Bug "mlibPrint.isFinalState: empty stack"
      | [frame] => isEmptyFrame frame
      | _ :: _ :: _ => false
    end;

local
  val initialBlock =
      let
        val indent = 0
        and style = Inconsistent
      in
        Block
          {indent = indent,
           style = style}
      end;

  val initialFrame =
      let
        val block = initialBlock
        and broken = false
        and indent = 0
        and size = 0
        and chunks = []
      in
        Frame
          {block = block,
           broken = broken,
           indent = indent,
           size = size,
           chunks = chunks}
      end;
in
  val initialState =
      let
        val lineIndent = 0
        and lineSize = 0
        and stack = [initialFrame]
      in
        State
          {lineIndent = lineIndent,
           lineSize = lineSize,
           stack = stack}
      end;
end;

fun normalizeState lineLength lines state =
    let
      val State {lineIndent,lineSize,stack} = state

      val within =
          case lineLength of
            NONE => true
          | SOME len => lineIndent + lineSize <= len
    in
      if within then (lines,state)
      else
        case breakFrames lines lineIndent stack of
          NONE => (lines,state)
        | SOME (lines,lineIndent,lineSize,stack) =>
          let
(*BasicDebug
            val () = checkState state
*)
            val state =
                State
                  {lineIndent = lineIndent,
                   lineSize = lineSize,
                   stack = stack}
          in
            normalizeState lineLength lines state
          end
    end
(*BasicDebug
    handle Bug bug => raise Bug ("mlibPrint.normalizeState:\n" ^ bug)
*)

local
  fun executeBeginBlock block lines state =
      let
        val State {lineIndent,lineSize,stack} = state

        val broken = false
        and indent = indentBlock block
        and size = 0
        and chunks = []

        val frame =
            Frame
              {block = block,
               broken = broken,
               indent = indent,
               size = size,
               chunks = chunks}

        val stack = frame :: stack

        val state =
            State
              {lineIndent = lineIndent,
               lineSize = lineSize,
               stack = stack}
      in
        (lines,state)
      end;

  fun executeEndBlock lines state =
      let
        val State {lineIndent,lineSize,stack} = state

        val (lineSize,stack) =
            case stack of
              [] => raise Bug "mlibPrint.executeEndBlock: empty stack"
            | topFrame :: stack =>
              let
                val Frame
                      {block = topBlock,
                       broken = topBroken,
                       indent = topIndent,
                       size = topSize,
                       chunks = topChunks} = topFrame

                val (lineSize,topSize,topChunks,topFrame) =
                    case topChunks of
                      BreakChunk break :: chunks =>
                      let
(*BasicDebug
                        val () =
                            let
                              val mesg =
                                  "ignoring a break at the end of a " ^
                                  "pretty-printing block"
                            in
                              warn mesg
                            end
*)
                        val n = sizeBreak break

                        val lineSize = lineSize - n
                        and topSize = topSize - n
                        and topChunks = chunks

                        val topFrame =
                            Frame
                              {block = topBlock,
                               broken = topBroken,
                               indent = topIndent,
                               size = topSize,
                               chunks = topChunks}
                      in
                        (lineSize,topSize,topChunks,topFrame)
                      end
                    | _ => (lineSize,topSize,topChunks,topFrame)
              in
                if List.null topChunks then (lineSize,stack)
                else
                  case stack of
                    [] => raise Error "mlibPrint.execute: too many end blocks"
                  | frame :: stack =>
                    let
                      val Frame
                            {block,
                             broken,
                             indent,
                             size,
                             chunks} = frame

                      val size = size + topSize

                      val chunk = FrameChunk topFrame

                      val chunks = chunk :: chunks

                      val frame =
                          Frame
                            {block = block,
                             broken = broken,
                             indent = indent,
                             size = size,
                             chunks = chunks}

                      val stack = frame :: stack
                    in
                      (lineSize,stack)
                    end
              end

        val state =
            State
              {lineIndent = lineIndent,
               lineSize = lineSize,
               stack = stack}
      in
        (lines,state)
      end;

  fun executeAddWord lineLength word lines state =
      let
        val State {lineIndent,lineSize,stack} = state

        val n = sizeWord word

        val lineSize = lineSize + n

        val stack =
            case stack of
              [] => raise Bug "mlibPrint.executeAddWord: empty stack"
            | frame :: stack =>
              let
                val Frame
                      {block,
                       broken,
                       indent,
                       size,
                       chunks} = frame

                val size = size + n

                val chunk = WordChunk word

                val chunks = chunk :: chunks

                val frame =
                    Frame
                      {block = block,
                       broken = broken,
                       indent = indent,
                       size = size,
                       chunks = chunks}

                val stack = frame :: stack
              in
                stack
              end

        val state =
            State
              {lineIndent = lineIndent,
               lineSize = lineSize,
               stack = stack}
      in
        normalizeState lineLength lines state
      end;

  fun executeAddBreak lineLength break lines state =
      let
        val State {lineIndent,lineSize,stack} = state

        val (topFrame,restFrames) =
            case stack of
              [] => raise Bug "mlibPrint.executeAddBreak: empty stack"
            | topFrame :: restFrames => (topFrame,restFrames)

        val Frame
              {block = topBlock,
               broken = topBroken,
               indent = topIndent,
               size = topSize,
               chunks = topChunks} = topFrame
      in
        case topChunks of
          [] => (lines,state)
        | topChunk :: restTopChunks =>
          let
            val (topChunks,n) =
                case topChunk of
                  BreakChunk break' =>
                  let
                    val break = appendBreak break' break

                    val chunk = BreakChunk break

                    val topChunks = chunk :: restTopChunks
                    and n = sizeBreak break - sizeBreak break'
                  in
                    (topChunks,n)
                  end
                | _ =>
                  let
                    val chunk = BreakChunk break

                    val topChunks = chunk :: topChunks
                    and n = sizeBreak break
                  in
                    (topChunks,n)
                  end

            val lineSize = lineSize + n

            val topSize = topSize + n

            val topFrame =
                Frame
                  {block = topBlock,
                   broken = topBroken,
                   indent = topIndent,
                   size = topSize,
                   chunks = topChunks}

            val stack = topFrame :: restFrames

            val state =
                State
                  {lineIndent = lineIndent,
                   lineSize = lineSize,
                   stack = stack}

            val lineLength =
                if breakingFrame topFrame then SOME ~1 else lineLength
          in
            normalizeState lineLength lines state
          end
      end;

  fun executeBigBreak lines state =
      let
        val lineLength = SOME ~1
        and break = mkBreak 0
      in
        executeAddBreak lineLength break lines state
      end;

  fun executeAddNewline lineLength lines state =
      if isEmptyState state then (addEmptyLine lines, state)
      else executeBigBreak lines state;

  fun executeEof lineLength lines state =
      if isFinalState state then (lines,state)
      else
        let
          val (lines,state) = executeBigBreak lines state

(*BasicDebug
          val () =
              if isFinalState state then ()
              else raise Bug "mlibPrint.executeEof: not a final state"
*)
        in
          (lines,state)
        end;
in
  fun render {lineLength} =
      let
        fun execute step state =
            let
              val lines = []
            in
              case step of
                BeginBlock block => executeBeginBlock block lines state
              | EndBlock => executeEndBlock lines state
              | AddWord word => executeAddWord lineLength word lines state
              | AddBreak break => executeAddBreak lineLength break lines state
              | AddNewline => executeAddNewline lineLength lines state
            end

(*BasicDebug
        val execute = fn step => fn state =>
            let
              val (lines,state) = execute step state

              val () = checkState state
            in
              (lines,state)
            end
            handle Bug bug =>
              raise Bug ("mlibPrint.execute: after " ^ toStringStep step ^
                         " command:\n" ^ bug)
*)

        fun final state =
            let
              val lines = []

              val (lines,state) = executeEof lineLength lines state

(*BasicDebug
              val () = checkState state
*)
            in
              if List.null lines then mlibStream.Nil else mlibStream.singleton lines
            end
(*BasicDebug
            handle Bug bug => raise Bug ("mlibPrint.final: " ^ bug)
*)
      in
        fn pps =>
           let
             val lines = mlibStream.maps execute final initialState pps
           in
             revConcat lines
           end
      end;
end;

local
  fun inc {indent,line} acc = line :: nSpaces indent :: acc;

  fun incn (indent_line,acc) = inc indent_line ("\n" :: acc);
in
  fun toStringWithLineLength len ppA a =
      case render len (ppA a) of
        mlibStream.Nil => ""
      | mlibStream.Cons (h,t) =>
        let
          val lines = mlibStream.foldl incn (inc h []) (t ())
        in
          String.concat (List.rev lines)
        end;
end;

local
  fun renderLine {indent,line} = nSpaces indent ^ line ^ "\n";
in
  fun toStreamWithLineLength len ppA a =
      mlibStream.map renderLine (render len (ppA a));
end;

fun toLine ppA a = toStringWithLineLength {lineLength = NONE} ppA a;

(* ------------------------------------------------------------------------- *)
(* Pretty-printer rendering with a global line length.                       *)
(* ------------------------------------------------------------------------- *)

val lineLength = ref initialLineLength;

fun toString ppA a =
    let
      val len = {lineLength = SOME (!lineLength)}
    in
      toStringWithLineLength len ppA a
    end;

fun toStream ppA a =
    let
      val len = {lineLength = SOME (!lineLength)}
    in
      toStreamWithLineLength len ppA a
    end;

local
  val sep = mkWord " =";
in
  fun trace ppX nameX x =
      mlibUseful.trace (toString (ppOp2' sep ppString ppX) (nameX,x) ^ "\n");
end;

end
(* ========================================================================= *)
(* NAMES                                                                     *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibName =
sig

(* ------------------------------------------------------------------------- *)
(* A type of names.                                                          *)
(* ------------------------------------------------------------------------- *)

type name

(* ------------------------------------------------------------------------- *)
(* A total ordering.                                                         *)
(* ------------------------------------------------------------------------- *)

val compare : name * name -> order

val equal : name -> name -> bool

(* ------------------------------------------------------------------------- *)
(* Fresh names.                                                              *)
(* ------------------------------------------------------------------------- *)

val newName : unit -> name

val newNames : int -> name list

val variantPrime : {avoid : name -> bool} -> name -> name

val variantNum : {avoid : name -> bool} -> name -> name

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty printing.                                              *)
(* ------------------------------------------------------------------------- *)

val pp : name mlibPrint.pp

val toString : name -> string

val fromString : string -> name

end
(* ========================================================================= *)
(* NAMES                                                                     *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibName :> mlibName =
struct

open mlibUseful;

(* ------------------------------------------------------------------------- *)
(* A type of names.                                                          *)
(* ------------------------------------------------------------------------- *)

type name = string;

(* ------------------------------------------------------------------------- *)
(* A total ordering.                                                         *)
(* ------------------------------------------------------------------------- *)

val compare = String.compare;

fun equal n1 n2 = n1 = n2;

(* ------------------------------------------------------------------------- *)
(* Fresh variables.                                                          *)
(* ------------------------------------------------------------------------- *)

local
  val prefix  = "_";

  fun numName i = mkPrefix prefix (Int.toString i);
in
  fun newName () = numName (newInt ());

  fun newNames n = List.map numName (newInts n);
end;

fun variantPrime {avoid} =
    let
      fun variant n = if avoid n then variant (n ^ "'") else n
    in
      variant
    end;

local
  fun isDigitOrPrime c = c = #"'" orelse Char.isDigit c;
in
  fun variantNum {avoid} n =
      if not (avoid n) then n
      else
        let
          val n = stripSuffix isDigitOrPrime n

          fun variant i =
              let
                val n_i = n ^ Int.toString i
              in
                if avoid n_i then variant (i + 1) else n_i
              end
        in
          variant 0
        end;
end;

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty printing.                                              *)
(* ------------------------------------------------------------------------- *)

val pp = mlibPrint.ppString;

fun toString s : string = s;

fun fromString s : name = s;

end

structure mlibNameOrdered =
struct type t = mlibName.name val compare = mlibName.compare end

structure mlibNameMap = mlibKeyMap (mlibNameOrdered);

structure mlibNameSet =
struct

local
  structure S = mlibElementSet (mlibNameMap);
in
  open S;
end;

val pp =
    mlibPrint.ppMap
      toList
      (mlibPrint.ppBracket "{" "}" (mlibPrint.ppOpList "," mlibName.pp));

end
(* ========================================================================= *)
(* NAME/ARITY PAIRS                                                          *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibNameArity =
sig

(* ------------------------------------------------------------------------- *)
(* A type of name/arity pairs.                                               *)
(* ------------------------------------------------------------------------- *)

type nameArity = mlibName.name * int

val name : nameArity -> mlibName.name

val arity : nameArity -> int

(* ------------------------------------------------------------------------- *)
(* Testing for different arities.                                            *)
(* ------------------------------------------------------------------------- *)

val nary : int -> nameArity -> bool

val nullary : nameArity -> bool

val unary : nameArity -> bool

val binary : nameArity -> bool

val ternary : nameArity -> bool

(* ------------------------------------------------------------------------- *)
(* A total ordering.                                                         *)
(* ------------------------------------------------------------------------- *)

val compare : nameArity * nameArity -> order

val equal : nameArity -> nameArity -> bool

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty printing.                                              *)
(* ------------------------------------------------------------------------- *)

val pp : nameArity mlibPrint.pp

end
(* ========================================================================= *)
(* NAME/ARITY PAIRS                                                          *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibNameArity :> mlibNameArity =
struct

(* ------------------------------------------------------------------------- *)
(* A type of name/arity pairs.                                               *)
(* ------------------------------------------------------------------------- *)

type nameArity = mlibName.name * int;

fun name ((n,_) : nameArity) = n;

fun arity ((_,i) : nameArity) = i;

(* ------------------------------------------------------------------------- *)
(* Testing for different arities.                                            *)
(* ------------------------------------------------------------------------- *)

fun nary i n_i = arity n_i = i;

val nullary = nary 0
and unary = nary 1
and binary = nary 2
and ternary = nary 3;

(* ------------------------------------------------------------------------- *)
(* A total ordering.                                                         *)
(* ------------------------------------------------------------------------- *)

fun compare ((n1,i1),(n2,i2)) =
    case mlibName.compare (n1,n2) of
      LESS => LESS
    | EQUAL => Int.compare (i1,i2)
    | GREATER => GREATER;

fun equal (n1,i1) (n2,i2) = i1 = i2 andalso mlibName.equal n1 n2;

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty printing.                                              *)
(* ------------------------------------------------------------------------- *)

fun pp (n,i) =
    mlibPrint.inconsistentBlock 0
      [mlibName.pp n,
       mlibPrint.ppString "/",
       mlibPrint.ppInt i];

end

structure mlibNameArityOrdered =
struct type t = mlibNameArity.nameArity val compare = mlibNameArity.compare end

structure mlibNameArityMap =
struct

local
  structure S = mlibKeyMap (mlibNameArityOrdered);
in
  open S;
end;

fun compose m1 m2 =
    let
      fun pk ((_,a),n) = peek m2 (n,a)
    in
      mapPartial pk m1
    end;

end

structure mlibNameAritySet =
struct

local
  structure S = mlibElementSet (mlibNameArityMap);
in
  open S;
end;

val allNullary = all mlibNameArity.nullary;

val pp =
    mlibPrint.ppMap
      toList
      (mlibPrint.ppBracket "{" "}" (mlibPrint.ppOpList "," mlibNameArity.pp));


end
(* ========================================================================= *)
(* PARSING                                                                   *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibParse =
sig

(* ------------------------------------------------------------------------- *)
(* A "cannot parse" exception.                                               *)
(* ------------------------------------------------------------------------- *)

exception NoParse

(* ------------------------------------------------------------------------- *)
(* Recursive descent parsing combinators.                                    *)
(* ------------------------------------------------------------------------- *)

(*
  Recommended fixities:

  infixr 9 >>++
  infixr 8 ++
  infixr 7 >>
  infixr 6 ||
*)

val error : 'a -> 'b * 'a

val ++ : ('a -> 'b * 'a) * ('a -> 'c * 'a) -> 'a -> ('b * 'c) * 'a

val >> : ('a -> 'b * 'a) * ('b -> 'c) -> 'a -> 'c * 'a

val >>++ : ('a -> 'b * 'a) * ('b -> 'a -> 'c * 'a) -> 'a -> 'c * 'a

val || : ('a -> 'b * 'a) * ('a -> 'b * 'a) -> 'a -> 'b * 'a

val first : ('a -> 'b * 'a) list -> 'a -> 'b * 'a

val mmany : ('s -> 'a -> 's * 'a) -> 's -> 'a -> 's * 'a

val many : ('a -> 'b * 'a) -> 'a -> 'b list * 'a

val atLeastOne : ('a -> 'b * 'a) -> 'a -> 'b list * 'a

val nothing : 'a -> unit * 'a

val optional : ('a -> 'b * 'a) -> 'a -> 'b option * 'a

(* ------------------------------------------------------------------------- *)
(* mlibStream-based parsers.                                                     *)
(* ------------------------------------------------------------------------- *)

type ('a,'b) parser = 'a mlibStream.stream -> 'b * 'a mlibStream.stream

val maybe : ('a -> 'b option) -> ('a,'b) parser

val finished : ('a,unit) parser

val some : ('a -> bool) -> ('a,'a) parser

val any : ('a,'a) parser

(* ------------------------------------------------------------------------- *)
(* Parsing whole streams.                                                    *)
(* ------------------------------------------------------------------------- *)

val fromStream : ('a,'b) parser -> 'a mlibStream.stream -> 'b

val fromList : ('a,'b) parser -> 'a list -> 'b

val everything : ('a, 'b list) parser -> 'a mlibStream.stream -> 'b mlibStream.stream

(* ------------------------------------------------------------------------- *)
(* Parsing lines of text.                                                    *)
(* ------------------------------------------------------------------------- *)

val initialize :
    {lines : string mlibStream.stream} ->
    {chars : char list mlibStream.stream,
     parseErrorLocation : unit -> string}

val exactChar : char -> (char,unit) parser

val exactCharList : char list -> (char,unit) parser

val exactString : string -> (char,unit) parser

val escapeString : {escape : char list} -> (char,string) parser

val manySpace : (char,unit) parser

val atLeastOneSpace : (char,unit) parser

val fromString : (char,'a) parser -> string -> 'a

(* ------------------------------------------------------------------------- *)
(* Infix operators.                                                          *)
(* ------------------------------------------------------------------------- *)

val parseInfixes :
    mlibPrint.infixes ->
    (mlibPrint.token * 'a * 'a -> 'a) -> ('b,mlibPrint.token) parser ->
    ('b,'a) parser -> ('b,'a) parser

(* ------------------------------------------------------------------------- *)
(* Quotations.                                                               *)
(* ------------------------------------------------------------------------- *)

type 'a quotation = 'a frag list

val parseQuotation : ('a -> string) -> (string -> 'b) -> 'a quotation -> 'b

end
(* ========================================================================= *)
(* PARSING                                                                   *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibParse :> mlibParse =
struct

open mlibUseful;

infixr 9 >>++
infixr 8 ++
infixr 7 >>
infixr 6 ||

(* ------------------------------------------------------------------------- *)
(* A "cannot parse" exception.                                               *)
(* ------------------------------------------------------------------------- *)

exception NoParse;

(* ------------------------------------------------------------------------- *)
(* Recursive descent parsing combinators.                                    *)
(* ------------------------------------------------------------------------- *)

val error : 'a -> 'b * 'a = fn _ => raise NoParse;

fun op ++ (parser1,parser2) input =
    let
      val (result1,input) = parser1 input
      val (result2,input) = parser2 input
    in
      ((result1,result2),input)
    end;

fun op >> (parser : 'a -> 'b * 'a, treatment) input =
    let
      val (result,input) = parser input
    in
      (treatment result, input)
    end;

fun op >>++ (parser,treatment) input =
    let
      val (result,input) = parser input
    in
      treatment result input
    end;

fun op || (parser1,parser2) input =
    parser1 input handle NoParse => parser2 input;

fun first [] _ = raise NoParse
  | first (parser :: parsers) input = (parser || first parsers) input;

fun mmany parser state input =
    let
      val (state,input) = parser state input
    in
      mmany parser state input
    end
    handle NoParse => (state,input);

fun many parser =
    let
      fun sparser l = parser >> (fn x => x :: l)
    in
      mmany sparser [] >> List.rev
    end;

fun atLeastOne p = (p ++ many p) >> op::;

fun nothing input = ((),input);

fun optional p = (p >> SOME) || (nothing >> K NONE);

(* ------------------------------------------------------------------------- *)
(* mlibStream-based parsers.                                                     *)
(* ------------------------------------------------------------------------- *)

type ('a,'b) parser = 'a mlibStream.stream -> 'b * 'a mlibStream.stream

fun maybe p mlibStream.Nil = raise NoParse
  | maybe p (mlibStream.Cons (h,t)) =
    case p h of SOME r => (r, t ()) | NONE => raise NoParse;

fun finished mlibStream.Nil = ((), mlibStream.Nil)
  | finished (mlibStream.Cons _) = raise NoParse;

fun some p = maybe (fn x => if p x then SOME x else NONE);

fun any input = some (K true) input;

(* ------------------------------------------------------------------------- *)
(* Parsing whole streams.                                                    *)
(* ------------------------------------------------------------------------- *)

fun fromStream parser input =
    let
      val (res,_) = (parser ++ finished >> fst) input
    in
      res
    end;

fun fromList parser l = fromStream parser (mlibStream.fromList l);

fun everything parser =
    let
      fun parserOption input =
          SOME (parser input)
          handle e as NoParse => if mlibStream.null input then NONE else raise e

      fun parserList input =
          case parserOption input of
            NONE => mlibStream.Nil
          | SOME (result,input) =>
            mlibStream.append (mlibStream.fromList result) (fn () => parserList input)
    in
      parserList
    end;

(* ------------------------------------------------------------------------- *)
(* Parsing lines of text.                                                    *)
(* ------------------------------------------------------------------------- *)

fun initialize {lines} =
    let
      val lastLine = ref (~1,"","","")

      val chars =
          let
            fun saveLast line =
                let
                  val ref (n,_,l2,l3) = lastLine
                  val () = lastLine := (n + 1, l2, l3, line)
                in
                  String.explode line
                end
          in
            mlibStream.memoize (mlibStream.map saveLast lines)
          end

      fun parseErrorLocation () =
          let
            val ref (n,l1,l2,l3) = lastLine
          in
            (if n <= 0 then "at start"
             else "around line " ^ Int.toString n) ^
            chomp (":\n" ^ l1 ^ l2 ^ l3)
          end
    in
      {chars = chars,
       parseErrorLocation = parseErrorLocation}
    end;

fun exactChar (c : char) = some (equal c) >> K ();

fun exactCharList cs =
    case cs of
      [] => nothing
    | c :: cs => (exactChar c ++ exactCharList cs) >> snd;

fun exactString s = exactCharList (String.explode s);

fun escapeString {escape} =
    let
      fun isEscape c = mem c escape

      fun isNormal c =
          case c of
            #"\\" => false
          | #"\n" => false
          | #"\t" => false
          | _ => not (isEscape c)

      val escapeParser =
          (exactChar #"\\" >> K #"\\") ||
          (exactChar #"n" >> K #"\n") ||
          (exactChar #"t" >> K #"\t") ||
          some isEscape

      val charParser =
          ((exactChar #"\\" ++ escapeParser) >> snd) ||
          some isNormal
    in
      many charParser >> String.implode
    end;

local
  val isSpace = Char.isSpace;

  val space = some isSpace;
in
  val manySpace = many space >> K ();

  val atLeastOneSpace = atLeastOne space >> K ();
end;

fun fromString parser s = fromList parser (String.explode s);

(* ------------------------------------------------------------------------- *)
(* Infix operators.                                                          *)
(* ------------------------------------------------------------------------- *)

fun parseLayeredInfixes {tokens,assoc} mk tokParser subParser =
    let
      fun layerTokParser inp =
          let
            val tok_rest as (tok,_) = tokParser inp
          in
            if mlibStringSet.member tok tokens then tok_rest
            else raise NoParse
          end

      fun layerMk (x,txs) =
          case assoc of
            mlibPrint.LeftAssoc =>
            let
              fun inc ((t,y),z) = mk (t,z,y)
            in
              List.foldl inc x txs
            end
          | mlibPrint.NonAssoc =>
            (case txs of
               [] => x
             | [(t,y)] => mk (t,x,y)
             | _ => raise NoParse)
          | mlibPrint.RightAssoc =>
            (case List.rev txs of
               [] => x
             | tx :: txs =>
               let
                 fun inc ((t,y),(u,z)) = (t, mk (u,y,z))

                 val (t,y) = List.foldl inc tx txs
               in
                 mk (t,x,y)
               end)

      val layerParser = subParser ++ many (layerTokParser ++ subParser)
    in
      layerParser >> layerMk
    end;

fun parseInfixes ops =
    let
      val layeredOps = mlibPrint.layerInfixes ops

      val iparsers = List.map parseLayeredInfixes layeredOps
    in
      fn mk => fn tokParser => fn subParser =>
         List.foldr (fn (p,sp) => p mk tokParser sp) subParser iparsers
    end;

(* ------------------------------------------------------------------------- *)
(* Quotations.                                                               *)
(* ------------------------------------------------------------------------- *)

type 'a quotation = 'a frag list;

fun parseQuotation printer parser quote =
  let
    fun expand (QUOTE q, s) = s ^ q
      | expand (ANTIQUOTE a, s) = s ^ printer a

    val string = List.foldl expand "" quote
  in
    parser string
  end;

end
(* ========================================================================= *)
(* FIRST ORDER LOGIC TERMS                                                   *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibTerm =
sig

(* ------------------------------------------------------------------------- *)
(* A type of first order logic terms.                                        *)
(* ------------------------------------------------------------------------- *)

type var = mlibName.name

type functionName = mlibName.name

type function = functionName * int

type const = functionName

datatype term =
    Var of var
  | Fn of functionName * term list

(* ------------------------------------------------------------------------- *)
(* Constructors and destructors.                                             *)
(* ------------------------------------------------------------------------- *)

(* Variables *)

val destVar : term -> var

val isVar : term -> bool

val equalVar : var -> term -> bool

(* Functions *)

val destFn : term -> functionName * term list

val isFn : term -> bool

val fnName : term -> functionName

val fnArguments : term -> term list

val fnArity : term -> int

val fnFunction : term -> function

val functions : term -> mlibNameAritySet.set

val functionNames : term -> mlibNameSet.set

(* Constants *)

val mkConst : const -> term

val destConst : term -> const

val isConst : term -> bool

(* Binary functions *)

val mkBinop : functionName -> term * term -> term

val destBinop : functionName -> term -> term * term

val isBinop : functionName -> term -> bool

(* ------------------------------------------------------------------------- *)
(* The size of a term in symbols.                                            *)
(* ------------------------------------------------------------------------- *)

val symbols : term -> int

(* ------------------------------------------------------------------------- *)
(* A total comparison function for terms.                                    *)
(* ------------------------------------------------------------------------- *)

val compare : term * term -> order

val equal : term -> term -> bool

(* ------------------------------------------------------------------------- *)
(* Subterms.                                                                 *)
(* ------------------------------------------------------------------------- *)

type path = int list

val subterm : term -> path -> term

val subterms : term -> (path * term) list

val replace : term -> path * term -> term

val find : (term -> bool) -> term -> path option

val ppPath : path mlibPrint.pp

val pathToString : path -> string

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

val freeIn : var -> term -> bool

val freeVars : term -> mlibNameSet.set

val freeVarsList : term list -> mlibNameSet.set

(* ------------------------------------------------------------------------- *)
(* Fresh variables.                                                          *)
(* ------------------------------------------------------------------------- *)

val newVar : unit -> term

val newVars : int -> term list

val variantPrime : mlibNameSet.set -> var -> var

val variantNum : mlibNameSet.set -> var -> var

(* ------------------------------------------------------------------------- *)
(* Special support for terms with type annotations.                          *)
(* ------------------------------------------------------------------------- *)

val hasTypeFunctionName : functionName

val hasTypeFunction : function

val isTypedVar : term -> bool

val typedSymbols : term -> int

val nonVarTypedSubterms : term -> (path * term) list

(* ------------------------------------------------------------------------- *)
(* Special support for terms with an explicit function application operator. *)
(* ------------------------------------------------------------------------- *)

val appName : mlibName.name

val mkApp : term * term -> term

val destApp : term -> term * term

val isApp : term -> bool

val listMkApp : term * term list -> term

val stripApp : term -> term * term list

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty printing.                                              *)
(* ------------------------------------------------------------------------- *)

(* Infix symbols *)

val infixes : mlibPrint.infixes ref

(* The negation symbol *)

val negation : string ref

(* Binder symbols *)

val binders : string list ref

(* Bracket symbols *)

val brackets : (string * string) list ref

(* Pretty printing *)

val pp : term mlibPrint.pp

val toString : term -> string

(* Parsing *)

val fromString : string -> term

val parse : term mlibParse.quotation -> term

end
(* ========================================================================= *)
(* FIRST ORDER LOGIC TERMS                                                   *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibTerm :> mlibTerm =
struct

open mlibUseful;

(* ------------------------------------------------------------------------- *)
(* A type of first order logic terms.                                        *)
(* ------------------------------------------------------------------------- *)

type var = mlibName.name;

type functionName = mlibName.name;

type function = functionName * int;

type const = functionName;

datatype term =
    Var of mlibName.name
  | Fn of mlibName.name * term list;

(* ------------------------------------------------------------------------- *)
(* Constructors and destructors.                                             *)
(* ------------------------------------------------------------------------- *)

(* Variables *)

fun destVar (Var v) = v
  | destVar (Fn _) = raise Error "destVar";

val isVar = can destVar;

fun equalVar v (Var v') = mlibName.equal v v'
  | equalVar _ _ = false;

(* Functions *)

fun destFn (Fn f) = f
  | destFn (Var _) = raise Error "destFn";

val isFn = can destFn;

fun fnName tm = fst (destFn tm);

fun fnArguments tm = snd (destFn tm);

fun fnArity tm = length (fnArguments tm);

fun fnFunction tm = (fnName tm, fnArity tm);

local
  fun func fs [] = fs
    | func fs (Var _ :: tms) = func fs tms
    | func fs (Fn (n,l) :: tms) =
      func (mlibNameAritySet.add fs (n, length l)) (l @ tms);
in
  fun functions tm = func mlibNameAritySet.empty [tm];
end;

local
  fun func fs [] = fs
    | func fs (Var _ :: tms) = func fs tms
    | func fs (Fn (n,l) :: tms) = func (mlibNameSet.add fs n) (l @ tms);
in
  fun functionNames tm = func mlibNameSet.empty [tm];
end;

(* Constants *)

fun mkConst c = (Fn (c, []));

fun destConst (Fn (c, [])) = c
  | destConst _ = raise Error "destConst";

val isConst = can destConst;

(* Binary functions *)

fun mkBinop f (a,b) = Fn (f,[a,b]);

fun destBinop f (Fn (x,[a,b])) =
    if mlibName.equal x f then (a,b) else raise Error "mlibTerm.destBinop: wrong binop"
  | destBinop _ _ = raise Error "mlibTerm.destBinop: not a binop";

fun isBinop f = can (destBinop f);

(* ------------------------------------------------------------------------- *)
(* The size of a term in symbols.                                            *)
(* ------------------------------------------------------------------------- *)

val VAR_SYMBOLS = 1;

val FN_SYMBOLS = 1;

local
  fun sz n [] = n
    | sz n (Var _ :: tms) = sz (n + VAR_SYMBOLS) tms
    | sz n (Fn (func,args) :: tms) = sz (n + FN_SYMBOLS) (args @ tms);
in
  fun symbols tm = sz 0 [tm];
end;

(* ------------------------------------------------------------------------- *)
(* A total comparison function for terms.                                    *)
(* ------------------------------------------------------------------------- *)

local
  fun cmp [] [] = EQUAL
    | cmp (tm1 :: tms1) (tm2 :: tms2) =
      let
        val tm1_tm2 = (tm1,tm2)
      in
        if mlibPortable.pointerEqual tm1_tm2 then cmp tms1 tms2
        else
          case tm1_tm2 of
            (Var v1, Var v2) =>
            (case mlibName.compare (v1,v2) of
               LESS => LESS
             | EQUAL => cmp tms1 tms2
             | GREATER => GREATER)
          | (Var _, Fn _) => LESS
          | (Fn _, Var _) => GREATER
          | (Fn (f1,a1), Fn (f2,a2)) =>
            (case mlibName.compare (f1,f2) of
               LESS => LESS
             | EQUAL =>
               (case Int.compare (length a1, length a2) of
                  LESS => LESS
                | EQUAL => cmp (a1 @ tms1) (a2 @ tms2)
                | GREATER => GREATER)
             | GREATER => GREATER)
      end
    | cmp _ _ = raise Bug "mlibTerm.compare";
in
  fun compare (tm1,tm2) = cmp [tm1] [tm2];
end;

fun equal tm1 tm2 = compare (tm1,tm2) = EQUAL;

(* ------------------------------------------------------------------------- *)
(* Subterms.                                                                 *)
(* ------------------------------------------------------------------------- *)

type path = int list;

fun subterm tm [] = tm
  | subterm (Var _) (_ :: _) = raise Error "mlibTerm.subterm: Var"
  | subterm (Fn (_,tms)) (h :: t) =
    if h >= length tms then raise Error "mlibTerm.replace: Fn"
    else subterm (List.nth (tms,h)) t;

local
  fun subtms [] acc = acc
    | subtms ((path,tm) :: rest) acc =
      let
        fun f (n,arg) = (n :: path, arg)

        val acc = (List.rev path, tm) :: acc
      in
        case tm of
          Var _ => subtms rest acc
        | Fn (_,args) => subtms (List.map f (enumerate args) @ rest) acc
      end;
in
  fun subterms tm = subtms [([],tm)] [];
end;

fun replace tm ([],res) = if equal res tm then tm else res
  | replace tm (h :: t, res) =
    case tm of
      Var _ => raise Error "mlibTerm.replace: Var"
    | Fn (func,tms) =>
      if h >= length tms then raise Error "mlibTerm.replace: Fn"
      else
        let
          val arg = List.nth (tms,h)
          val arg' = replace arg (t,res)
        in
          if mlibPortable.pointerEqual (arg',arg) then tm
          else Fn (func, updateNth (h,arg') tms)
        end;

fun find pred =
    let
      fun search [] = NONE
        | search ((path,tm) :: rest) =
          if pred tm then SOME (List.rev path)
          else
            case tm of
              Var _ => search rest
            | Fn (_,a) =>
              let
                val subtms = List.map (fn (i,t) => (i :: path, t)) (enumerate a)
              in
                search (subtms @ rest)
              end
    in
      fn tm => search [([],tm)]
    end;

val ppPath = mlibPrint.ppList mlibPrint.ppInt;

val pathToString = mlibPrint.toString ppPath;

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

local
  fun free _ [] = false
    | free v (Var w :: tms) = mlibName.equal v w orelse free v tms
    | free v (Fn (_,args) :: tms) = free v (args @ tms);
in
  fun freeIn v tm = free v [tm];
end;

local
  fun free vs [] = vs
    | free vs (Var v :: tms) = free (mlibNameSet.add vs v) tms
    | free vs (Fn (_,args) :: tms) = free vs (args @ tms);
in
  val freeVarsList = free mlibNameSet.empty;

  fun freeVars tm = freeVarsList [tm];
end;

(* ------------------------------------------------------------------------- *)
(* Fresh variables.                                                          *)
(* ------------------------------------------------------------------------- *)

fun newVar () = Var (mlibName.newName ());

fun newVars n = List.map Var (mlibName.newNames n);

local
  fun avoid av n = mlibNameSet.member n av;
in
  fun variantPrime av = mlibName.variantPrime {avoid = avoid av};

  fun variantNum av = mlibName.variantNum {avoid = avoid av};
end;

(* ------------------------------------------------------------------------- *)
(* Special support for terms with type annotations.                          *)
(* ------------------------------------------------------------------------- *)

val hasTypeFunctionName = mlibName.fromString ":";

val hasTypeFunction = (hasTypeFunctionName,2);

fun destFnHasType ((f,a) : functionName * term list) =
    if not (mlibName.equal f hasTypeFunctionName) then
      raise Error "mlibTerm.destFnHasType"
    else
      case a of
        [tm,ty] => (tm,ty)
      | _ => raise Error "mlibTerm.destFnHasType";

val isFnHasType = can destFnHasType;

fun isTypedVar tm =
    case tm of
      Var _ => true
    | Fn func =>
      case total destFnHasType func of
        SOME (Var _, _) => true
      | _ => false;

local
  fun sz n [] = n
    | sz n (tm :: tms) =
      case tm of
        Var _ => sz (n + 1) tms
      | Fn func =>
        case total destFnHasType func of
          SOME (tm,_) => sz n (tm :: tms)
        | NONE =>
          let
            val (_,a) = func
          in
            sz (n + 1) (a @ tms)
          end;
in
  fun typedSymbols tm = sz 0 [tm];
end;

local
  fun subtms [] acc = acc
    | subtms ((path,tm) :: rest) acc =
      case tm of
        Var _ => subtms rest acc
      | Fn func =>
        case total destFnHasType func of
          SOME (t,_) =>
          (case t of
             Var _ => subtms rest acc
           | Fn _ =>
             let
               val acc = (List.rev path, tm) :: acc
               val rest = (0 :: path, t) :: rest
             in
               subtms rest acc
             end)
        | NONE =>
          let
            fun f (n,arg) = (n :: path, arg)

            val (_,args) = func

            val acc = (List.rev path, tm) :: acc

            val rest = List.map f (enumerate args) @ rest
          in
            subtms rest acc
          end;
in
  fun nonVarTypedSubterms tm = subtms [([],tm)] [];
end;

(* ------------------------------------------------------------------------- *)
(* Special support for terms with an explicit function application operator. *)
(* ------------------------------------------------------------------------- *)

val appName = mlibName.fromString ".";

fun mkFnApp (fTm,aTm) = (appName, [fTm,aTm]);

fun mkApp f_a = Fn (mkFnApp f_a);

fun destFnApp ((f,a) : mlibName.name * term list) =
    if not (mlibName.equal f appName) then raise Error "mlibTerm.destFnApp"
    else
      case a of
        [fTm,aTm] => (fTm,aTm)
      | _ => raise Error "mlibTerm.destFnApp";

val isFnApp = can destFnApp;

fun destApp tm =
    case tm of
      Var _ => raise Error "mlibTerm.destApp"
    | Fn func => destFnApp func;

val isApp = can destApp;

fun listMkApp (f,l) = List.foldl mkApp f l;

local
  fun strip tms tm =
      case total destApp tm of
        SOME (f,a) => strip (a :: tms) f
      | NONE => (tm,tms);
in
  fun stripApp tm = strip [] tm;
end;

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty printing.                                              *)
(* ------------------------------------------------------------------------- *)

(* Operators parsed and printed infix *)

val infixes =
    (ref o mlibPrint.Infixes)
      [(* ML symbols *)
       {token = "/", precedence = 7, assoc = mlibPrint.LeftAssoc},
       {token = "div", precedence = 7, assoc = mlibPrint.LeftAssoc},
       {token = "mod", precedence = 7, assoc = mlibPrint.LeftAssoc},
       {token = "*", precedence = 7, assoc = mlibPrint.LeftAssoc},
       {token = "+", precedence = 6, assoc = mlibPrint.LeftAssoc},
       {token = "-", precedence = 6, assoc = mlibPrint.LeftAssoc},
       {token = "^", precedence = 6, assoc = mlibPrint.LeftAssoc},
       {token = "@", precedence = 5, assoc = mlibPrint.RightAssoc},
       {token = "::", precedence = 5, assoc = mlibPrint.RightAssoc},
       {token = "=", precedence = 4, assoc = mlibPrint.NonAssoc},
       {token = "<>", precedence = 4, assoc = mlibPrint.NonAssoc},
       {token = "<=", precedence = 4, assoc = mlibPrint.NonAssoc},
       {token = "<", precedence = 4, assoc = mlibPrint.NonAssoc},
       {token = ">=", precedence = 4, assoc = mlibPrint.NonAssoc},
       {token = ">", precedence = 4, assoc = mlibPrint.NonAssoc},
       {token = "o", precedence = 3, assoc = mlibPrint.LeftAssoc},
       {token = "->", precedence = 2, assoc = mlibPrint.RightAssoc},
       {token = ":", precedence = 1, assoc = mlibPrint.NonAssoc},
       {token = ",", precedence = 0, assoc = mlibPrint.RightAssoc},

       (* Logical connectives *)
       {token = "/\\", precedence = ~1, assoc = mlibPrint.RightAssoc},
       {token = "\\/", precedence = ~2, assoc = mlibPrint.RightAssoc},
       {token = "==>", precedence = ~3, assoc = mlibPrint.RightAssoc},
       {token = "<=>", precedence = ~4, assoc = mlibPrint.RightAssoc},

       (* Other symbols *)
       {token = ".", precedence = 9, assoc = mlibPrint.LeftAssoc},
       {token = "**", precedence = 8, assoc = mlibPrint.LeftAssoc},
       {token = "++", precedence = 6, assoc = mlibPrint.LeftAssoc},
       {token = "--", precedence = 6, assoc = mlibPrint.LeftAssoc},
       {token = "==", precedence = 4, assoc = mlibPrint.NonAssoc}];

(* The negation symbol *)

val negation : string ref = ref "~";

(* Binder symbols *)

val binders : string list ref = ref ["\\","!","?","?!"];

(* Bracket symbols *)

val brackets : (string * string) list ref = ref [("[","]"),("{","}")];

(* Pretty printing *)

fun pp inputTerm =
    let
      val quants = !binders
      and iOps = !infixes
      and neg = !negation
      and bracks = !brackets

      val bMap =
          let
            fun f (b1,b2) = (b1 ^ b2, b1, b2)
          in
            List.map f bracks
          end

      val bTokens = op@ (unzip bracks)

      val iTokens = mlibPrint.tokensInfixes iOps

      fun destI tm =
          case tm of
            Fn (f,[a,b]) =>
            let
              val f = mlibName.toString f
            in
              if mlibStringSet.member f iTokens then SOME (f,a,b) else NONE
            end
          | _ => NONE

      fun isI tm = Option.isSome (destI tm)

      fun iToken (_,tok) =
          mlibPrint.program
            [(if tok = "," then mlibPrint.skip else mlibPrint.ppString " "),
             mlibPrint.ppString tok,
             mlibPrint.break];

      val iPrinter = mlibPrint.ppInfixes iOps destI iToken

      val specialTokens =
          mlibStringSet.addList iTokens (neg :: quants @ ["$","(",")"] @ bTokens)

      fun vName bv s = mlibStringSet.member s bv

      fun checkVarName bv n =
          let
            val s = mlibName.toString n
          in
            if vName bv s then s else "$" ^ s
          end

      fun varName bv = mlibPrint.ppMap (checkVarName bv) mlibPrint.ppString

      fun checkFunctionName bv n =
          let
            val s = mlibName.toString n
          in
            if mlibStringSet.member s specialTokens orelse vName bv s then
              "(" ^ s ^ ")"
            else
              s
          end

      fun functionName bv = mlibPrint.ppMap (checkFunctionName bv) mlibPrint.ppString

      fun stripNeg tm =
          case tm of
            Fn (f,[a]) =>
            if mlibName.toString f <> neg then (0,tm)
            else let val (n,tm) = stripNeg a in (n + 1, tm) end
          | _ => (0,tm)

      val destQuant =
          let
            fun dest q (Fn (q', [Var v, body])) =
                if mlibName.toString q' <> q then NONE
                else
                  (case dest q body of
                     NONE => SOME (q,v,[],body)
                   | SOME (_,v',vs,body) => SOME (q, v, v' :: vs, body))
              | dest _ _ = NONE
          in
            fn tm => mlibUseful.first (fn q => dest q tm) quants
          end

      fun isQuant tm = Option.isSome (destQuant tm)

      fun destBrack (Fn (b,[tm])) =
          let
            val s = mlibName.toString b
          in
            case List.find (fn (n,_,_) => n = s) bMap of
              NONE => NONE
            | SOME (_,b1,b2) => SOME (b1,tm,b2)
          end
        | destBrack _ = NONE

      fun isBrack tm = Option.isSome (destBrack tm)

      fun functionArgument bv tm =
          mlibPrint.sequence
            mlibPrint.break
            (if isBrack tm then customBracket bv tm
             else if isVar tm orelse isConst tm then basic bv tm
             else bracket bv tm)

      and basic bv (Var v) = varName bv v
        | basic bv (Fn (f,args)) =
          mlibPrint.inconsistentBlock 2
            (functionName bv f :: List.map (functionArgument bv) args)

      and customBracket bv tm =
          case destBrack tm of
            SOME (b1,tm,b2) => mlibPrint.ppBracket b1 b2 (term bv) tm
          | NONE => basic bv tm

      and innerQuant bv tm =
          case destQuant tm of
            NONE => term bv tm
          | SOME (q,v,vs,tm) =>
            let
              val bv = mlibStringSet.addList bv (List.map mlibName.toString (v :: vs))
            in
              mlibPrint.program
                [mlibPrint.ppString q,
                 varName bv v,
                 mlibPrint.program
                   (List.map (mlibPrint.sequence mlibPrint.break o varName bv) vs),
                 mlibPrint.ppString ".",
                 mlibPrint.break,
                 innerQuant bv tm]
            end

      and quantifier bv tm =
          if not (isQuant tm) then customBracket bv tm
          else mlibPrint.inconsistentBlock 2 [innerQuant bv tm]

      and molecule bv (tm,r) =
          let
            val (n,tm) = stripNeg tm
          in
            mlibPrint.inconsistentBlock n
              [mlibPrint.duplicate n (mlibPrint.ppString neg),
               if isI tm orelse (r andalso isQuant tm) then bracket bv tm
               else quantifier bv tm]
          end

      and term bv tm = iPrinter (molecule bv) (tm,false)

      and bracket bv tm = mlibPrint.ppBracket "(" ")" (term bv) tm
    in
      term mlibStringSet.empty
    end inputTerm;

val toString = mlibPrint.toString pp;

(* Parsing *)

local
  open mlibParse;

  infixr 9 >>++
  infixr 8 ++
  infixr 7 >>
  infixr 6 ||

  val isAlphaNum =
      let
        val alphaNumChars = String.explode "_'"
      in
        fn c => mem c alphaNumChars orelse Char.isAlphaNum c
      end;

  local
    val alphaNumToken = atLeastOne (some isAlphaNum) >> String.implode;

    val symbolToken =
        let
          fun isNeg c = str c = !negation

          val symbolChars = String.explode "<>=-*+/\\?@|!$%&#^:;~"

          fun isSymbol c = mem c symbolChars

          fun isNonNegSymbol c = not (isNeg c) andalso isSymbol c
        in
          some isNeg >> str ||
          (some isNonNegSymbol ++ many (some isSymbol)) >>
          (String.implode o op::)
        end;

    val punctToken =
        let
          val punctChars = String.explode "()[]{}.,"

          fun isPunct c = mem c punctChars
        in
          some isPunct >> str
        end;

    val lexToken = alphaNumToken || symbolToken || punctToken;

    val space = many (some Char.isSpace);
  in
    val lexer = (space ++ lexToken ++ space) >> (fn (_,(tok,_)) => tok);
  end;

  fun termParser inputStream =
      let
        val quants = !binders
        and iOps = !infixes
        and neg = !negation
        and bracks = ("(",")") :: !brackets

        val bracks = List.map (fn (b1,b2) => (b1 ^ b2, b1, b2)) bracks

        val bTokens = List.map #2 bracks @ List.map #3 bracks

        fun possibleVarName "" = false
          | possibleVarName s = isAlphaNum (String.sub (s,0))

        fun vName bv s = mlibStringSet.member s bv

        val iTokens = mlibPrint.tokensInfixes iOps

        fun iMk (f,a,b) = Fn (mlibName.fromString f, [a,b])

        val iParser = parseInfixes iOps iMk any

        val specialTokens =
            mlibStringSet.addList iTokens (neg :: quants @ ["$"] @ bTokens)

        fun varName bv =
            some (vName bv) ||
            (some (mlibUseful.equal "$") ++ some possibleVarName) >> snd

        fun fName bv s =
            not (mlibStringSet.member s specialTokens) andalso not (vName bv s)

        fun functionName bv =
            some (fName bv) ||
            (some (mlibUseful.equal "(") ++ any ++ some (mlibUseful.equal ")")) >>
            (fn (_,(s,_)) => s)

        fun basic bv tokens =
            let
              val var = varName bv >> (Var o mlibName.fromString)

              val const =
                  functionName bv >> (fn f => Fn (mlibName.fromString f, []))

              fun bracket (ab,a,b) =
                  (some (mlibUseful.equal a) ++ term bv ++ some (mlibUseful.equal b)) >>
                  (fn (_,(tm,_)) =>
                      if ab = "()" then tm else Fn (mlibName.fromString ab, [tm]))

              fun quantifier q =
                  let
                    fun bind (v,t) =
                        Fn (mlibName.fromString q, [Var (mlibName.fromString v), t])
                  in
                    (some (mlibUseful.equal q) ++
                     atLeastOne (some possibleVarName) ++
                     some (mlibUseful.equal ".")) >>++
                    (fn (_,(vs,_)) =>
                        term (mlibStringSet.addList bv vs) >>
                        (fn body => List.foldr bind body vs))
                  end
            in
              var ||
              const ||
              first (List.map bracket bracks) ||
              first (List.map quantifier quants)
            end tokens

        and molecule bv tokens =
            let
              val negations = many (some (mlibUseful.equal neg)) >> length

              val function =
                  (functionName bv ++ many (basic bv)) >>
                  (fn (f,args) => Fn (mlibName.fromString f, args)) ||
                  basic bv
            in
              (negations ++ function) >>
              (fn (n,tm) => funpow n (fn t => Fn (mlibName.fromString neg, [t])) tm)
            end tokens

        and term bv tokens = iParser (molecule bv) tokens
      in
        term mlibStringSet.empty
      end inputStream;
in
  fun fromString input =
      let
        val chars = mlibStream.fromList (String.explode input)

        val tokens = everything (lexer >> singleton) chars

        val terms = everything (termParser >> singleton) tokens
      in
        case mlibStream.toList terms of
          [tm] => tm
        | _ => raise Error "mlibTerm.fromString"
      end;
end;

local
  val antiquotedTermToString =
      mlibPrint.toString (mlibPrint.ppBracket "(" ")" pp);
in
  val parse = mlibParse.parseQuotation antiquotedTermToString fromString;
end;

end

structure mlibTermOrdered =
struct type t = mlibTerm.term val compare = mlibTerm.compare end

structure mlibTermMap = mlibKeyMap (mlibTermOrdered);

structure mlibTermSet = mlibElementSet (mlibTermMap);
(* ========================================================================= *)
(* FINITE MAPS IMPLEMENTED WITH RANDOMLY BALANCED TREES                      *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibMap =
sig

(* ------------------------------------------------------------------------- *)
(* A type of finite maps.                                                    *)
(* ------------------------------------------------------------------------- *)

type ('key,'a) map

(* ------------------------------------------------------------------------- *)
(* Constructors.                                                             *)
(* ------------------------------------------------------------------------- *)

val new : ('key * 'key -> order) -> ('key,'a) map

val singleton : ('key * 'key -> order) -> 'key * 'a -> ('key,'a) map

(* ------------------------------------------------------------------------- *)
(* mlibMap size.                                                                 *)
(* ------------------------------------------------------------------------- *)

val null : ('key,'a) map -> bool

val size : ('key,'a) map -> int

(* ------------------------------------------------------------------------- *)
(* Querying.                                                                 *)
(* ------------------------------------------------------------------------- *)

val peekKey : ('key,'a) map -> 'key -> ('key * 'a) option

val peek : ('key,'a) map -> 'key -> 'a option

val get : ('key,'a) map -> 'key -> 'a  (* raises Error *)

val pick : ('key,'a) map -> 'key * 'a  (* an arbitrary key/value pair *)

val nth : ('key,'a) map -> int -> 'key * 'a  (* in the range [0,size-1] *)

val random : ('key,'a) map -> 'key * 'a

(* ------------------------------------------------------------------------- *)
(* Adding.                                                                   *)
(* ------------------------------------------------------------------------- *)

val insert : ('key,'a) map -> 'key * 'a -> ('key,'a) map

val insertList : ('key,'a) map -> ('key * 'a) list -> ('key,'a) map

(* ------------------------------------------------------------------------- *)
(* Removing.                                                                 *)
(* ------------------------------------------------------------------------- *)

val delete : ('key,'a) map -> 'key -> ('key,'a) map  (* must be present *)

val remove : ('key,'a) map -> 'key -> ('key,'a) map

val deletePick : ('key,'a) map -> ('key * 'a) * ('key,'a) map

val deleteNth : ('key,'a) map -> int -> ('key * 'a) * ('key,'a) map

val deleteRandom : ('key,'a) map -> ('key * 'a) * ('key,'a) map

(* ------------------------------------------------------------------------- *)
(* Joining (all join operations prefer keys in the second map).              *)
(* ------------------------------------------------------------------------- *)

val merge :
    {first : 'key * 'a -> 'c option,
     second : 'key * 'b -> 'c option,
     both : ('key * 'a) * ('key * 'b) -> 'c option} ->
    ('key,'a) map -> ('key,'b) map -> ('key,'c) map

val union :
    (('key * 'a) * ('key * 'a) -> 'a option) ->
    ('key,'a) map -> ('key,'a) map -> ('key,'a) map

val intersect :
    (('key * 'a) * ('key * 'b) -> 'c option) ->
    ('key,'a) map -> ('key,'b) map -> ('key,'c) map

(* ------------------------------------------------------------------------- *)
(* Set operations on the domain.                                             *)
(* ------------------------------------------------------------------------- *)

val inDomain : 'key -> ('key,'a) map -> bool

val unionDomain : ('key,'a) map -> ('key,'a) map -> ('key,'a) map

val unionListDomain : ('key,'a) map list -> ('key,'a) map

val intersectDomain : ('key,'a) map -> ('key,'a) map -> ('key,'a) map

val intersectListDomain : ('key,'a) map list -> ('key,'a) map

val differenceDomain : ('key,'a) map -> ('key,'a) map -> ('key,'a) map

val symmetricDifferenceDomain : ('key,'a) map -> ('key,'a) map -> ('key,'a) map

val equalDomain : ('key,'a) map -> ('key,'a) map -> bool

val subsetDomain : ('key,'a) map -> ('key,'a) map -> bool

val disjointDomain : ('key,'a) map -> ('key,'a) map -> bool

(* ------------------------------------------------------------------------- *)
(* Mapping and folding.                                                      *)
(* ------------------------------------------------------------------------- *)

val mapPartial : ('key * 'a -> 'b option) -> ('key,'a) map -> ('key,'b) map

val map : ('key * 'a -> 'b) -> ('key,'a) map -> ('key,'b) map

val app : ('key * 'a -> unit) -> ('key,'a) map -> unit

val transform : ('a -> 'b) -> ('key,'a) map -> ('key,'b) map

val filter : ('key * 'a -> bool) -> ('key,'a) map -> ('key,'a) map

val partition :
    ('key * 'a -> bool) -> ('key,'a) map -> ('key,'a) map * ('key,'a) map

val foldl : ('key * 'a * 's -> 's) -> 's -> ('key,'a) map -> 's

val foldr : ('key * 'a * 's -> 's) -> 's -> ('key,'a) map -> 's

(* ------------------------------------------------------------------------- *)
(* Searching.                                                                *)
(* ------------------------------------------------------------------------- *)

val findl : ('key * 'a -> bool) -> ('key,'a) map -> ('key * 'a) option

val findr : ('key * 'a -> bool) -> ('key,'a) map -> ('key * 'a) option

val firstl : ('key * 'a -> 'b option) -> ('key,'a) map -> 'b option

val firstr : ('key * 'a -> 'b option) -> ('key,'a) map -> 'b option

val exists : ('key * 'a -> bool) -> ('key,'a) map -> bool

val all : ('key * 'a -> bool) -> ('key,'a) map -> bool

val count : ('key * 'a -> bool) -> ('key,'a) map -> int

(* ------------------------------------------------------------------------- *)
(* Comparing.                                                                *)
(* ------------------------------------------------------------------------- *)

val compare : ('a * 'a -> order) -> ('key,'a) map * ('key,'a) map -> order

val equal : ('a -> 'a -> bool) -> ('key,'a) map -> ('key,'a) map -> bool

(* ------------------------------------------------------------------------- *)
(* Converting to and from lists.                                             *)
(* ------------------------------------------------------------------------- *)

val keys : ('key,'a) map -> 'key list

val values : ('key,'a) map -> 'a list

val toList : ('key,'a) map -> ('key * 'a) list

val fromList : ('key * 'key -> order) -> ('key * 'a) list -> ('key,'a) map

(* ------------------------------------------------------------------------- *)
(* Pretty-printing.                                                          *)
(* ------------------------------------------------------------------------- *)

val toString : ('key,'a) map -> string

(* ------------------------------------------------------------------------- *)
(* Iterators over maps.                                                      *)
(* ------------------------------------------------------------------------- *)

type ('key,'a) iterator

val mkIterator : ('key,'a) map -> ('key,'a) iterator option

val mkRevIterator : ('key,'a) map -> ('key,'a) iterator option

val readIterator : ('key,'a) iterator -> 'key * 'a

val advanceIterator : ('key,'a) iterator -> ('key,'a) iterator option

end
(* ========================================================================= *)
(* FINITE MAPS IMPLEMENTED WITH RANDOMLY BALANCED TREES                      *)
(* Copyright (c) 2004 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibMap :> mlibMap =
struct

(* ------------------------------------------------------------------------- *)
(* Importing useful functionality.                                           *)
(* ------------------------------------------------------------------------- *)

exception Bug = mlibUseful.Bug;

exception Error = mlibUseful.Error;

val pointerEqual = mlibPortable.pointerEqual;

val K = mlibUseful.K;

val randomInt = mlibPortable.randomInt;

val randomWord = mlibPortable.randomWord;

(* ------------------------------------------------------------------------- *)
(* Converting a comparison function to an equality function.                 *)
(* ------------------------------------------------------------------------- *)

fun equalKey compareKey key1 key2 = compareKey (key1,key2) = EQUAL;

(* ------------------------------------------------------------------------- *)
(* Priorities.                                                               *)
(* ------------------------------------------------------------------------- *)

type priority = Word.word;

val randomPriority = randomWord;

val comparePriority = Word.compare;

(* ------------------------------------------------------------------------- *)
(* Priority search trees.                                                    *)
(* ------------------------------------------------------------------------- *)

datatype ('key,'value) tree =
    E
  | T of ('key,'value) node

and ('key,'value) node =
    Node of
      {size : int,
       priority : priority,
       left : ('key,'value) tree,
       key : 'key,
       value : 'value,
       right : ('key,'value) tree};

fun lowerPriorityNode node1 node2 =
    let
      val Node {priority = p1, ...} = node1
      and Node {priority = p2, ...} = node2
    in
      comparePriority (p1,p2) = LESS
    end;

(* ------------------------------------------------------------------------- *)
(* Tree debugging functions.                                                 *)
(* ------------------------------------------------------------------------- *)

(*BasicDebug
local
  fun checkSizes tree =
      case tree of
        E => 0
      | T (Node {size,left,right,...}) =>
        let
          val l = checkSizes left
          and r = checkSizes right

          val () = if l + 1 + r = size then () else raise Bug "wrong size"
        in
          size
        end;

  fun checkSorted compareKey x tree =
      case tree of
        E => x
      | T (Node {left,key,right,...}) =>
        let
          val x = checkSorted compareKey x left

          val () =
              case x of
                NONE => ()
              | SOME k =>
                case compareKey (k,key) of
                  LESS => ()
                | EQUAL => raise Bug "duplicate keys"
                | GREATER => raise Bug "unsorted"

          val x = SOME key
        in
          checkSorted compareKey x right
        end;

  fun checkPriorities compareKey tree =
      case tree of
        E => NONE
      | T node =>
        let
          val Node {left,right,...} = node

          val () =
              case checkPriorities compareKey left of
                NONE => ()
              | SOME lnode =>
                if not (lowerPriorityNode node lnode) then ()
                else raise Bug "left child has greater priority"

          val () =
              case checkPriorities compareKey right of
                NONE => ()
              | SOME rnode =>
                if not (lowerPriorityNode node rnode) then ()
                else raise Bug "right child has greater priority"
        in
          SOME node
        end;
in
  fun treeCheckInvariants compareKey tree =
      let
        val _ = checkSizes tree

        val _ = checkSorted compareKey NONE tree

        val _ = checkPriorities compareKey tree
      in
        tree
      end
      handle Error err => raise Bug err;
end;
*)

(* ------------------------------------------------------------------------- *)
(* Tree operations.                                                          *)
(* ------------------------------------------------------------------------- *)

fun treeNew () = E;

fun nodeSize (Node {size = x, ...}) = x;

fun treeSize tree =
    case tree of
      E => 0
    | T x => nodeSize x;

fun mkNode priority left key value right =
    let
      val size = treeSize left + 1 + treeSize right
    in
      Node
        {size = size,
         priority = priority,
         left = left,
         key = key,
         value = value,
         right = right}
    end;

fun mkTree priority left key value right =
    let
      val node = mkNode priority left key value right
    in
      T node
    end;

(* ------------------------------------------------------------------------- *)
(* Extracting the left and right spines of a tree.                           *)
(* ------------------------------------------------------------------------- *)

fun treeLeftSpine acc tree =
    case tree of
      E => acc
    | T node => nodeLeftSpine acc node

and nodeLeftSpine acc node =
    let
      val Node {left,...} = node
    in
      treeLeftSpine (node :: acc) left
    end;

fun treeRightSpine acc tree =
    case tree of
      E => acc
    | T node => nodeRightSpine acc node

and nodeRightSpine acc node =
    let
      val Node {right,...} = node
    in
      treeRightSpine (node :: acc) right
    end;

(* ------------------------------------------------------------------------- *)
(* Singleton trees.                                                          *)
(* ------------------------------------------------------------------------- *)

fun mkNodeSingleton priority key value =
    let
      val size = 1
      and left = E
      and right = E
    in
      Node
        {size = size,
         priority = priority,
         left = left,
         key = key,
         value = value,
         right = right}
    end;

fun nodeSingleton (key,value) =
    let
      val priority = randomPriority ()
    in
      mkNodeSingleton priority key value
    end;

fun treeSingleton key_value =
    let
      val node = nodeSingleton key_value
    in
      T node
    end;

(* ------------------------------------------------------------------------- *)
(* Appending two trees, where every element of the first tree is less than   *)
(* every element of the second tree.                                         *)
(* ------------------------------------------------------------------------- *)

fun treeAppend tree1 tree2 =
    case tree1 of
      E => tree2
    | T node1 =>
      case tree2 of
        E => tree1
      | T node2 =>
        if lowerPriorityNode node1 node2 then
          let
            val Node {priority,left,key,value,right,...} = node2

            val left = treeAppend tree1 left
          in
            mkTree priority left key value right
          end
        else
          let
            val Node {priority,left,key,value,right,...} = node1

            val right = treeAppend right tree2
          in
            mkTree priority left key value right
          end;

(* ------------------------------------------------------------------------- *)
(* Appending two trees and a node, where every element of the first tree is  *)
(* less than the node, which in turn is less than every element of the       *)
(* second tree.                                                              *)
(* ------------------------------------------------------------------------- *)

fun treeCombine left node right =
    let
      val left_node = treeAppend left (T node)
    in
      treeAppend left_node right
    end;

(* ------------------------------------------------------------------------- *)
(* Searching a tree for a value.                                             *)
(* ------------------------------------------------------------------------- *)

fun treePeek compareKey pkey tree =
    case tree of
      E => NONE
    | T node => nodePeek compareKey pkey node

and nodePeek compareKey pkey node =
    let
      val Node {left,key,value,right,...} = node
    in
      case compareKey (pkey,key) of
        LESS => treePeek compareKey pkey left
      | EQUAL => SOME value
      | GREATER => treePeek compareKey pkey right
    end;

(* ------------------------------------------------------------------------- *)
(* Tree paths.                                                               *)
(* ------------------------------------------------------------------------- *)

(* Generating a path by searching a tree for a key/value pair *)

fun treePeekPath compareKey pkey path tree =
    case tree of
      E => (path,NONE)
    | T node => nodePeekPath compareKey pkey path node

and nodePeekPath compareKey pkey path node =
    let
      val Node {left,key,right,...} = node
    in
      case compareKey (pkey,key) of
        LESS => treePeekPath compareKey pkey ((true,node) :: path) left
      | EQUAL => (path, SOME node)
      | GREATER => treePeekPath compareKey pkey ((false,node) :: path) right
    end;

(* A path splits a tree into left/right components *)

fun addSidePath ((wentLeft,node),(leftTree,rightTree)) =
    let
      val Node {priority,left,key,value,right,...} = node
    in
      if wentLeft then (leftTree, mkTree priority rightTree key value right)
      else (mkTree priority left key value leftTree, rightTree)
    end;

fun addSidesPath left_right = List.foldl addSidePath left_right;

fun mkSidesPath path = addSidesPath (E,E) path;

(* Updating the subtree at a path *)

local
  fun updateTree ((wentLeft,node),tree) =
      let
        val Node {priority,left,key,value,right,...} = node
      in
        if wentLeft then mkTree priority tree key value right
        else mkTree priority left key value tree
      end;
in
  fun updateTreePath tree = List.foldl updateTree tree;
end;

(* Inserting a new node at a path position *)

fun insertNodePath node =
    let
      fun insert left_right path =
          case path of
            [] =>
            let
              val (left,right) = left_right
            in
              treeCombine left node right
            end
          | (step as (_,snode)) :: rest =>
            if lowerPriorityNode snode node then
              let
                val left_right = addSidePath (step,left_right)
              in
                insert left_right rest
              end
            else
              let
                val (left,right) = left_right

                val tree = treeCombine left node right
              in
                updateTreePath tree path
              end
    in
      insert (E,E)
    end;

(* ------------------------------------------------------------------------- *)
(* Using a key to split a node into three components: the keys comparing     *)
(* less than the supplied key, an optional equal key, and the keys comparing *)
(* greater.                                                                  *)
(* ------------------------------------------------------------------------- *)

fun nodePartition compareKey pkey node =
    let
      val (path,pnode) = nodePeekPath compareKey pkey [] node
    in
      case pnode of
        NONE =>
        let
          val (left,right) = mkSidesPath path
        in
          (left,NONE,right)
        end
      | SOME node =>
        let
          val Node {left,key,value,right,...} = node

          val (left,right) = addSidesPath (left,right) path
        in
          (left, SOME (key,value), right)
        end
    end;

(* ------------------------------------------------------------------------- *)
(* Searching a tree for a key/value pair.                                    *)
(* ------------------------------------------------------------------------- *)

fun treePeekKey compareKey pkey tree =
    case tree of
      E => NONE
    | T node => nodePeekKey compareKey pkey node

and nodePeekKey compareKey pkey node =
    let
      val Node {left,key,value,right,...} = node
    in
      case compareKey (pkey,key) of
        LESS => treePeekKey compareKey pkey left
      | EQUAL => SOME (key,value)
      | GREATER => treePeekKey compareKey pkey right
    end;

(* ------------------------------------------------------------------------- *)
(* Inserting new key/values into the tree.                                   *)
(* ------------------------------------------------------------------------- *)

fun treeInsert compareKey key_value tree =
    let
      val (key,value) = key_value

      val (path,inode) = treePeekPath compareKey key [] tree
    in
      case inode of
        NONE =>
        let
          val node = nodeSingleton (key,value)
        in
          insertNodePath node path
        end
      | SOME node =>
        let
          val Node {size,priority,left,right,...} = node

          val node =
              Node
                {size = size,
                 priority = priority,
                 left = left,
                 key = key,
                 value = value,
                 right = right}
        in
          updateTreePath (T node) path
        end
    end;

(* ------------------------------------------------------------------------- *)
(* Deleting key/value pairs: it raises an exception if the supplied key is   *)
(* not present.                                                              *)
(* ------------------------------------------------------------------------- *)

fun treeDelete compareKey dkey tree =
    case tree of
      E => raise Bug "mlibMap.delete: element not found"
    | T node => nodeDelete compareKey dkey node

and nodeDelete compareKey dkey node =
    let
      val Node {size,priority,left,key,value,right} = node
    in
      case compareKey (dkey,key) of
        LESS =>
        let
          val size = size - 1
          and left = treeDelete compareKey dkey left

          val node =
              Node
                {size = size,
                 priority = priority,
                 left = left,
                 key = key,
                 value = value,
                 right = right}
        in
          T node
        end
      | EQUAL => treeAppend left right
      | GREATER =>
        let
          val size = size - 1
          and right = treeDelete compareKey dkey right

          val node =
              Node
                {size = size,
                 priority = priority,
                 left = left,
                 key = key,
                 value = value,
                 right = right}
        in
          T node
        end
    end;

(* ------------------------------------------------------------------------- *)
(* Partial map is the basic operation for preserving tree structure.         *)
(* It applies its argument function to the elements *in order*.              *)
(* ------------------------------------------------------------------------- *)

fun treeMapPartial f tree =
    case tree of
      E => E
    | T node => nodeMapPartial f node

and nodeMapPartial f (Node {priority,left,key,value,right,...}) =
    let
      val left = treeMapPartial f left
      and vo = f (key,value)
      and right = treeMapPartial f right
    in
      case vo of
        NONE => treeAppend left right
      | SOME value => mkTree priority left key value right
    end;

(* ------------------------------------------------------------------------- *)
(* Mapping tree values.                                                      *)
(* ------------------------------------------------------------------------- *)

fun treeMap f tree =
    case tree of
      E => E
    | T node => T (nodeMap f node)

and nodeMap f node =
    let
      val Node {size,priority,left,key,value,right} = node

      val left = treeMap f left
      and value = f (key,value)
      and right = treeMap f right
    in
      Node
        {size = size,
         priority = priority,
         left = left,
         key = key,
         value = value,
         right = right}
    end;

(* ------------------------------------------------------------------------- *)
(* Merge is the basic operation for joining two trees. Note that the merged  *)
(* key is always the one from the second map.                                *)
(* ------------------------------------------------------------------------- *)

fun treeMerge compareKey f1 f2 fb tree1 tree2 =
    case tree1 of
      E => treeMapPartial f2 tree2
    | T node1 =>
      case tree2 of
        E => treeMapPartial f1 tree1
      | T node2 => nodeMerge compareKey f1 f2 fb node1 node2

and nodeMerge compareKey f1 f2 fb node1 node2 =
    let
      val Node {priority,left,key,value,right,...} = node2

      val (l,kvo,r) = nodePartition compareKey key node1

      val left = treeMerge compareKey f1 f2 fb l left
      and right = treeMerge compareKey f1 f2 fb r right

      val vo =
          case kvo of
            NONE => f2 (key,value)
          | SOME kv => fb (kv,(key,value))
    in
      case vo of
        NONE => treeAppend left right
      | SOME value =>
        let
          val node = mkNodeSingleton priority key value
        in
          treeCombine left node right
        end
    end;

(* ------------------------------------------------------------------------- *)
(* A union operation on trees.                                               *)
(* ------------------------------------------------------------------------- *)

fun treeUnion compareKey f f2 tree1 tree2 =
    case tree1 of
      E => tree2
    | T node1 =>
      case tree2 of
        E => tree1
      | T node2 => nodeUnion compareKey f f2 node1 node2

and nodeUnion compareKey f f2 node1 node2 =
    if pointerEqual (node1,node2) then nodeMapPartial f2 node1
    else
      let
        val Node {priority,left,key,value,right,...} = node2

        val (l,kvo,r) = nodePartition compareKey key node1

        val left = treeUnion compareKey f f2 l left
        and right = treeUnion compareKey f f2 r right

        val vo =
            case kvo of
              NONE => SOME value
            | SOME kv => f (kv,(key,value))
      in
        case vo of
          NONE => treeAppend left right
        | SOME value =>
          let
            val node = mkNodeSingleton priority key value
          in
            treeCombine left node right
          end
      end;

(* ------------------------------------------------------------------------- *)
(* An intersect operation on trees.                                          *)
(* ------------------------------------------------------------------------- *)

fun treeIntersect compareKey f t1 t2 =
    case t1 of
      E => E
    | T n1 =>
      case t2 of
        E => E
      | T n2 => nodeIntersect compareKey f n1 n2

and nodeIntersect compareKey f n1 n2 =
    let
      val Node {priority,left,key,value,right,...} = n2

      val (l,kvo,r) = nodePartition compareKey key n1

      val left = treeIntersect compareKey f l left
      and right = treeIntersect compareKey f r right

      val vo =
          case kvo of
            NONE => NONE
          | SOME kv => f (kv,(key,value))
    in
      case vo of
        NONE => treeAppend left right
      | SOME value => mkTree priority left key value right
    end;

(* ------------------------------------------------------------------------- *)
(* A union operation on trees which simply chooses the second value.         *)
(* ------------------------------------------------------------------------- *)

fun treeUnionDomain compareKey tree1 tree2 =
    case tree1 of
      E => tree2
    | T node1 =>
      case tree2 of
        E => tree1
      | T node2 =>
        if pointerEqual (node1,node2) then tree2
        else nodeUnionDomain compareKey node1 node2

and nodeUnionDomain compareKey node1 node2 =
    let
      val Node {priority,left,key,value,right,...} = node2

      val (l,_,r) = nodePartition compareKey key node1

      val left = treeUnionDomain compareKey l left
      and right = treeUnionDomain compareKey r right

      val node = mkNodeSingleton priority key value
    in
      treeCombine left node right
    end;

(* ------------------------------------------------------------------------- *)
(* An intersect operation on trees which simply chooses the second value.    *)
(* ------------------------------------------------------------------------- *)

fun treeIntersectDomain compareKey tree1 tree2 =
    case tree1 of
      E => E
    | T node1 =>
      case tree2 of
        E => E
      | T node2 =>
        if pointerEqual (node1,node2) then tree2
        else nodeIntersectDomain compareKey node1 node2

and nodeIntersectDomain compareKey node1 node2 =
    let
      val Node {priority,left,key,value,right,...} = node2

      val (l,kvo,r) = nodePartition compareKey key node1

      val left = treeIntersectDomain compareKey l left
      and right = treeIntersectDomain compareKey r right
    in
      if Option.isSome kvo then mkTree priority left key value right
      else treeAppend left right
    end;

(* ------------------------------------------------------------------------- *)
(* A difference operation on trees.                                          *)
(* ------------------------------------------------------------------------- *)

fun treeDifferenceDomain compareKey t1 t2 =
    case t1 of
      E => E
    | T n1 =>
      case t2 of
        E => t1
      | T n2 => nodeDifferenceDomain compareKey n1 n2

and nodeDifferenceDomain compareKey n1 n2 =
    if pointerEqual (n1,n2) then E
    else
      let
        val Node {priority,left,key,value,right,...} = n1

        val (l,kvo,r) = nodePartition compareKey key n2

        val left = treeDifferenceDomain compareKey left l
        and right = treeDifferenceDomain compareKey right r
      in
        if Option.isSome kvo then treeAppend left right
        else mkTree priority left key value right
      end;

(* ------------------------------------------------------------------------- *)
(* A subset operation on trees.                                              *)
(* ------------------------------------------------------------------------- *)

fun treeSubsetDomain compareKey tree1 tree2 =
    case tree1 of
      E => true
    | T node1 =>
      case tree2 of
        E => false
      | T node2 => nodeSubsetDomain compareKey node1 node2

and nodeSubsetDomain compareKey node1 node2 =
    pointerEqual (node1,node2) orelse
    let
      val Node {size,left,key,right,...} = node1
    in
      size <= nodeSize node2 andalso
      let
        val (l,kvo,r) = nodePartition compareKey key node2
      in
        Option.isSome kvo andalso
        treeSubsetDomain compareKey left l andalso
        treeSubsetDomain compareKey right r
      end
    end;

(* ------------------------------------------------------------------------- *)
(* Picking an arbitrary key/value pair from a tree.                          *)
(* ------------------------------------------------------------------------- *)

fun nodePick node =
    let
      val Node {key,value,...} = node
    in
      (key,value)
    end;

fun treePick tree =
    case tree of
      E => raise Bug "mlibMap.treePick"
    | T node => nodePick node;

(* ------------------------------------------------------------------------- *)
(* Removing an arbitrary key/value pair from a tree.                         *)
(* ------------------------------------------------------------------------- *)

fun nodeDeletePick node =
    let
      val Node {left,key,value,right,...} = node
    in
      ((key,value), treeAppend left right)
    end;

fun treeDeletePick tree =
    case tree of
      E => raise Bug "mlibMap.treeDeletePick"
    | T node => nodeDeletePick node;

(* ------------------------------------------------------------------------- *)
(* Finding the nth smallest key/value (counting from 0).                     *)
(* ------------------------------------------------------------------------- *)

fun treeNth n tree =
    case tree of
      E => raise Bug "mlibMap.treeNth"
    | T node => nodeNth n node

and nodeNth n node =
    let
      val Node {left,key,value,right,...} = node

      val k = treeSize left
    in
      if n = k then (key,value)
      else if n < k then treeNth n left
      else treeNth (n - (k + 1)) right
    end;

(* ------------------------------------------------------------------------- *)
(* Removing the nth smallest key/value (counting from 0).                    *)
(* ------------------------------------------------------------------------- *)

fun treeDeleteNth n tree =
    case tree of
      E => raise Bug "mlibMap.treeDeleteNth"
    | T node => nodeDeleteNth n node

and nodeDeleteNth n node =
    let
      val Node {size,priority,left,key,value,right} = node

      val k = treeSize left
    in
      if n = k then ((key,value), treeAppend left right)
      else if n < k then
        let
          val (key_value,left) = treeDeleteNth n left

          val size = size - 1

          val node =
              Node
                {size = size,
                 priority = priority,
                 left = left,
                 key = key,
                 value = value,
                 right = right}
        in
          (key_value, T node)
        end
      else
        let
          val n = n - (k + 1)

          val (key_value,right) = treeDeleteNth n right

          val size = size - 1

          val node =
              Node
                {size = size,
                 priority = priority,
                 left = left,
                 key = key,
                 value = value,
                 right = right}
        in
          (key_value, T node)
        end
    end;

(* ------------------------------------------------------------------------- *)
(* Iterators.                                                                *)
(* ------------------------------------------------------------------------- *)

datatype ('key,'value) iterator =
    LeftToRightIterator of
      ('key * 'value) * ('key,'value) tree * ('key,'value) node list
  | RightToLeftIterator of
      ('key * 'value) * ('key,'value) tree * ('key,'value) node list;

fun fromSpineLeftToRightIterator nodes =
    case nodes of
      [] => NONE
    | Node {key,value,right,...} :: nodes =>
      SOME (LeftToRightIterator ((key,value),right,nodes));

fun fromSpineRightToLeftIterator nodes =
    case nodes of
      [] => NONE
    | Node {key,value,left,...} :: nodes =>
      SOME (RightToLeftIterator ((key,value),left,nodes));

fun addLeftToRightIterator nodes tree = fromSpineLeftToRightIterator (treeLeftSpine nodes tree);

fun addRightToLeftIterator nodes tree = fromSpineRightToLeftIterator (treeRightSpine nodes tree);

fun treeMkIterator tree = addLeftToRightIterator [] tree;

fun treeMkRevIterator tree = addRightToLeftIterator [] tree;

fun readIterator iter =
    case iter of
      LeftToRightIterator (key_value,_,_) => key_value
    | RightToLeftIterator (key_value,_,_) => key_value;

fun advanceIterator iter =
    case iter of
      LeftToRightIterator (_,tree,nodes) => addLeftToRightIterator nodes tree
    | RightToLeftIterator (_,tree,nodes) => addRightToLeftIterator nodes tree;

fun foldIterator f acc io =
    case io of
      NONE => acc
    | SOME iter =>
      let
        val (key,value) = readIterator iter
      in
        foldIterator f (f (key,value,acc)) (advanceIterator iter)
      end;

fun findIterator pred io =
    case io of
      NONE => NONE
    | SOME iter =>
      let
        val key_value = readIterator iter
      in
        if pred key_value then SOME key_value
        else findIterator pred (advanceIterator iter)
      end;

fun firstIterator f io =
    case io of
      NONE => NONE
    | SOME iter =>
      let
        val key_value = readIterator iter
      in
        case f key_value of
          NONE => firstIterator f (advanceIterator iter)
        | s => s
      end;

fun compareIterator compareKey compareValue io1 io2 =
    case (io1,io2) of
      (NONE,NONE) => EQUAL
    | (NONE, SOME _) => LESS
    | (SOME _, NONE) => GREATER
    | (SOME i1, SOME i2) =>
      let
        val (k1,v1) = readIterator i1
        and (k2,v2) = readIterator i2
      in
        case compareKey (k1,k2) of
          LESS => LESS
        | EQUAL =>
          (case compareValue (v1,v2) of
             LESS => LESS
           | EQUAL =>
             let
               val io1 = advanceIterator i1
               and io2 = advanceIterator i2
             in
               compareIterator compareKey compareValue io1 io2
             end
           | GREATER => GREATER)
        | GREATER => GREATER
      end;

fun equalIterator equalKey equalValue io1 io2 =
    case (io1,io2) of
      (NONE,NONE) => true
    | (NONE, SOME _) => false
    | (SOME _, NONE) => false
    | (SOME i1, SOME i2) =>
      let
        val (k1,v1) = readIterator i1
        and (k2,v2) = readIterator i2
      in
        equalKey k1 k2 andalso
        equalValue v1 v2 andalso
        let
          val io1 = advanceIterator i1
          and io2 = advanceIterator i2
        in
          equalIterator equalKey equalValue io1 io2
        end
      end;

(* ------------------------------------------------------------------------- *)
(* A type of finite maps.                                                    *)
(* ------------------------------------------------------------------------- *)

datatype ('key,'value) map =
    mlibMap of ('key * 'key -> order) * ('key,'value) tree;

(* ------------------------------------------------------------------------- *)
(* mlibMap debugging functions.                                                  *)
(* ------------------------------------------------------------------------- *)

(*BasicDebug
fun checkInvariants s m =
    let
      val mlibMap (compareKey,tree) = m

      val _ = treeCheckInvariants compareKey tree
    in
      m
    end
    handle Bug bug => raise Bug (s ^ "\n" ^ "mlibMap.checkInvariants: " ^ bug);
*)

(* ------------------------------------------------------------------------- *)
(* Constructors.                                                             *)
(* ------------------------------------------------------------------------- *)

fun new compareKey =
    let
      val tree = treeNew ()
    in
      mlibMap (compareKey,tree)
    end;

fun singleton compareKey key_value =
    let
      val tree = treeSingleton key_value
    in
      mlibMap (compareKey,tree)
    end;

(* ------------------------------------------------------------------------- *)
(* mlibMap size.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun size (mlibMap (_,tree)) = treeSize tree;

fun null m = size m = 0;

(* ------------------------------------------------------------------------- *)
(* Querying.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun peekKey (mlibMap (compareKey,tree)) key = treePeekKey compareKey key tree;

fun peek (mlibMap (compareKey,tree)) key = treePeek compareKey key tree;

fun inDomain key m = Option.isSome (peek m key);

fun get m key =
    case peek m key of
      NONE => raise Error "mlibMap.get: element not found"
    | SOME value => value;

fun pick (mlibMap (_,tree)) = treePick tree;

fun nth (mlibMap (_,tree)) n = treeNth n tree;

fun random m =
    let
      val n = size m
    in
      if n = 0 then raise Bug "mlibMap.random: empty"
      else nth m (randomInt n)
    end;

(* ------------------------------------------------------------------------- *)
(* Adding.                                                                   *)
(* ------------------------------------------------------------------------- *)

fun insert (mlibMap (compareKey,tree)) key_value =
    let
      val tree = treeInsert compareKey key_value tree
    in
      mlibMap (compareKey,tree)
    end;

(*BasicDebug
val insert = fn m => fn kv =>
    checkInvariants "mlibMap.insert: result"
      (insert (checkInvariants "mlibMap.insert: input" m) kv);
*)

fun insertList m =
    let
      fun ins (key_value,acc) = insert acc key_value
    in
      List.foldl ins m
    end;

(* ------------------------------------------------------------------------- *)
(* Removing.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun delete (mlibMap (compareKey,tree)) dkey =
    let
      val tree = treeDelete compareKey dkey tree
    in
      mlibMap (compareKey,tree)
    end;

(*BasicDebug
val delete = fn m => fn k =>
    checkInvariants "mlibMap.delete: result"
      (delete (checkInvariants "mlibMap.delete: input" m) k);
*)

fun remove m key = if inDomain key m then delete m key else m;

fun deletePick (mlibMap (compareKey,tree)) =
    let
      val (key_value,tree) = treeDeletePick tree
    in
      (key_value, mlibMap (compareKey,tree))
    end;

(*BasicDebug
val deletePick = fn m =>
    let
      val (kv,m) = deletePick (checkInvariants "mlibMap.deletePick: input" m)
    in
      (kv, checkInvariants "mlibMap.deletePick: result" m)
    end;
*)

fun deleteNth (mlibMap (compareKey,tree)) n =
    let
      val (key_value,tree) = treeDeleteNth n tree
    in
      (key_value, mlibMap (compareKey,tree))
    end;

(*BasicDebug
val deleteNth = fn m => fn n =>
    let
      val (kv,m) = deleteNth (checkInvariants "mlibMap.deleteNth: input" m) n
    in
      (kv, checkInvariants "mlibMap.deleteNth: result" m)
    end;
*)

fun deleteRandom m =
    let
      val n = size m
    in
      if n = 0 then raise Bug "mlibMap.deleteRandom: empty"
      else deleteNth m (randomInt n)
    end;

(* ------------------------------------------------------------------------- *)
(* Joining (all join operations prefer keys in the second map).              *)
(* ------------------------------------------------------------------------- *)

fun merge {first,second,both} (mlibMap (compareKey,tree1)) (mlibMap (_,tree2)) =
    let
      val tree = treeMerge compareKey first second both tree1 tree2
    in
      mlibMap (compareKey,tree)
    end;

(*BasicDebug
val merge = fn f => fn m1 => fn m2 =>
    checkInvariants "mlibMap.merge: result"
      (merge f
         (checkInvariants "mlibMap.merge: input 1" m1)
         (checkInvariants "mlibMap.merge: input 2" m2));
*)

fun union f (mlibMap (compareKey,tree1)) (mlibMap (_,tree2)) =
    let
      fun f2 kv = f (kv,kv)

      val tree = treeUnion compareKey f f2 tree1 tree2
    in
      mlibMap (compareKey,tree)
    end;

(*BasicDebug
val union = fn f => fn m1 => fn m2 =>
    checkInvariants "mlibMap.union: result"
      (union f
         (checkInvariants "mlibMap.union: input 1" m1)
         (checkInvariants "mlibMap.union: input 2" m2));
*)

fun intersect f (mlibMap (compareKey,tree1)) (mlibMap (_,tree2)) =
    let
      val tree = treeIntersect compareKey f tree1 tree2
    in
      mlibMap (compareKey,tree)
    end;

(*BasicDebug
val intersect = fn f => fn m1 => fn m2 =>
    checkInvariants "mlibMap.intersect: result"
      (intersect f
         (checkInvariants "mlibMap.intersect: input 1" m1)
         (checkInvariants "mlibMap.intersect: input 2" m2));
*)

(* ------------------------------------------------------------------------- *)
(* Iterators over maps.                                                      *)
(* ------------------------------------------------------------------------- *)

fun mkIterator (mlibMap (_,tree)) = treeMkIterator tree;

fun mkRevIterator (mlibMap (_,tree)) = treeMkRevIterator tree;

(* ------------------------------------------------------------------------- *)
(* Mapping and folding.                                                      *)
(* ------------------------------------------------------------------------- *)

fun mapPartial f (mlibMap (compareKey,tree)) =
    let
      val tree = treeMapPartial f tree
    in
      mlibMap (compareKey,tree)
    end;

(*BasicDebug
val mapPartial = fn f => fn m =>
    checkInvariants "mlibMap.mapPartial: result"
      (mapPartial f (checkInvariants "mlibMap.mapPartial: input" m));
*)

fun map f (mlibMap (compareKey,tree)) =
    let
      val tree = treeMap f tree
    in
      mlibMap (compareKey,tree)
    end;

(*BasicDebug
val map = fn f => fn m =>
    checkInvariants "mlibMap.map: result"
      (map f (checkInvariants "mlibMap.map: input" m));
*)

fun transform f = map (fn (_,value) => f value);

fun filter pred =
    let
      fun f (key_value as (_,value)) =
          if pred key_value then SOME value else NONE
    in
      mapPartial f
    end;

fun partition p =
    let
      fun np x = not (p x)
    in
      fn m => (filter p m, filter np m)
    end;

fun foldl f b m = foldIterator f b (mkIterator m);

fun foldr f b m = foldIterator f b (mkRevIterator m);

fun app f m = foldl (fn (key,value,()) => f (key,value)) () m;

(* ------------------------------------------------------------------------- *)
(* Searching.                                                                *)
(* ------------------------------------------------------------------------- *)

fun findl p m = findIterator p (mkIterator m);

fun findr p m = findIterator p (mkRevIterator m);

fun firstl f m = firstIterator f (mkIterator m);

fun firstr f m = firstIterator f (mkRevIterator m);

fun exists p m = Option.isSome (findl p m);

fun all p =
    let
      fun np x = not (p x)
    in
      fn m => not (exists np m)
    end;

fun count pred =
    let
      fun f (k,v,acc) = if pred (k,v) then acc + 1 else acc
    in
      foldl f 0
    end;

(* ------------------------------------------------------------------------- *)
(* Comparing.                                                                *)
(* ------------------------------------------------------------------------- *)

fun compare compareValue (m1,m2) =
    if pointerEqual (m1,m2) then EQUAL
    else
      case Int.compare (size m1, size m2) of
        LESS => LESS
      | EQUAL =>
        let
          val mlibMap (compareKey,_) = m1

          val io1 = mkIterator m1
          and io2 = mkIterator m2
        in
          compareIterator compareKey compareValue io1 io2
        end
      | GREATER => GREATER;

fun equal equalValue m1 m2 =
    pointerEqual (m1,m2) orelse
    (size m1 = size m2 andalso
     let
       val mlibMap (compareKey,_) = m1

       val io1 = mkIterator m1
       and io2 = mkIterator m2
     in
       equalIterator (equalKey compareKey) equalValue io1 io2
     end);

(* ------------------------------------------------------------------------- *)
(* Set operations on the domain.                                             *)
(* ------------------------------------------------------------------------- *)

fun unionDomain (mlibMap (compareKey,tree1)) (mlibMap (_,tree2)) =
    let
      val tree = treeUnionDomain compareKey tree1 tree2
    in
      mlibMap (compareKey,tree)
    end;

(*BasicDebug
val unionDomain = fn m1 => fn m2 =>
    checkInvariants "mlibMap.unionDomain: result"
      (unionDomain
         (checkInvariants "mlibMap.unionDomain: input 1" m1)
         (checkInvariants "mlibMap.unionDomain: input 2" m2));
*)

local
  fun uncurriedUnionDomain (m,acc) = unionDomain acc m;
in
  fun unionListDomain ms =
      case ms of
        [] => raise Bug "mlibMap.unionListDomain: no sets"
      | m :: ms => List.foldl uncurriedUnionDomain m ms;
end;

fun intersectDomain (mlibMap (compareKey,tree1)) (mlibMap (_,tree2)) =
    let
      val tree = treeIntersectDomain compareKey tree1 tree2
    in
      mlibMap (compareKey,tree)
    end;

(*BasicDebug
val intersectDomain = fn m1 => fn m2 =>
    checkInvariants "mlibMap.intersectDomain: result"
      (intersectDomain
         (checkInvariants "mlibMap.intersectDomain: input 1" m1)
         (checkInvariants "mlibMap.intersectDomain: input 2" m2));
*)

local
  fun uncurriedIntersectDomain (m,acc) = intersectDomain acc m;
in
  fun intersectListDomain ms =
      case ms of
        [] => raise Bug "mlibMap.intersectListDomain: no sets"
      | m :: ms => List.foldl uncurriedIntersectDomain m ms;
end;

fun differenceDomain (mlibMap (compareKey,tree1)) (mlibMap (_,tree2)) =
    let
      val tree = treeDifferenceDomain compareKey tree1 tree2
    in
      mlibMap (compareKey,tree)
    end;

(*BasicDebug
val differenceDomain = fn m1 => fn m2 =>
    checkInvariants "mlibMap.differenceDomain: result"
      (differenceDomain
         (checkInvariants "mlibMap.differenceDomain: input 1" m1)
         (checkInvariants "mlibMap.differenceDomain: input 2" m2));
*)

fun symmetricDifferenceDomain m1 m2 =
    unionDomain (differenceDomain m1 m2) (differenceDomain m2 m1);

fun equalDomain m1 m2 = equal (K (K true)) m1 m2;

fun subsetDomain (mlibMap (compareKey,tree1)) (mlibMap (_,tree2)) =
    treeSubsetDomain compareKey tree1 tree2;

fun disjointDomain m1 m2 = null (intersectDomain m1 m2);

(* ------------------------------------------------------------------------- *)
(* Converting to and from lists.                                             *)
(* ------------------------------------------------------------------------- *)

fun keys m = foldr (fn (key,_,l) => key :: l) [] m;

fun values m = foldr (fn (_,value,l) => value :: l) [] m;

fun toList m = foldr (fn (key,value,l) => (key,value) :: l) [] m;

fun fromList compareKey l =
    let
      val m = new compareKey
    in
      insertList m l
    end;

(* ------------------------------------------------------------------------- *)
(* Pretty-printing.                                                          *)
(* ------------------------------------------------------------------------- *)

fun toString m = "<" ^ (if null m then "" else Int.toString (size m)) ^ ">";

end
(* ========================================================================= *)
(* PRESERVING SHARING OF ML VALUES                                           *)
(* Copyright (c) 2005 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibSharing =
sig

(* ------------------------------------------------------------------------- *)
(* Option operations.                                                        *)
(* ------------------------------------------------------------------------- *)

val mapOption : ('a -> 'a) -> 'a option -> 'a option

val mapsOption : ('a -> 's -> 'a * 's) -> 'a option -> 's -> 'a option * 's

(* ------------------------------------------------------------------------- *)
(* List operations.                                                          *)
(* ------------------------------------------------------------------------- *)

val map : ('a -> 'a) -> 'a list -> 'a list

val revMap : ('a -> 'a) -> 'a list -> 'a list

val maps : ('a -> 's -> 'a * 's) -> 'a list -> 's -> 'a list * 's

val revMaps : ('a -> 's -> 'a * 's) -> 'a list -> 's -> 'a list * 's

val updateNth : int * 'a -> 'a list -> 'a list

val setify : ''a list -> ''a list

(* ------------------------------------------------------------------------- *)
(* Function caching.                                                         *)
(* ------------------------------------------------------------------------- *)

val cache : ('a * 'a -> order) -> ('a -> 'b) -> 'a -> 'b

(* ------------------------------------------------------------------------- *)
(* Hash consing.                                                             *)
(* ------------------------------------------------------------------------- *)

val hashCons : ('a * 'a -> order) -> 'a -> 'a

end
(* ========================================================================= *)
(* PRESERVING SHARING OF ML VALUES                                           *)
(* Copyright (c) 2005 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibSharing :> mlibSharing =
struct

infix ==

val op== = mlibPortable.pointerEqual;

(* ------------------------------------------------------------------------- *)
(* Option operations.                                                        *)
(* ------------------------------------------------------------------------- *)

fun mapOption f xo =
    case xo of
      SOME x =>
      let
        val y = f x
      in
        if x == y then xo else SOME y
      end
    | NONE => xo;

fun mapsOption f xo acc =
    case xo of
      SOME x =>
      let
        val (y,acc) = f x acc
      in
        if x == y then (xo,acc) else (SOME y, acc)
      end
    | NONE => (xo,acc);

(* ------------------------------------------------------------------------- *)
(* List operations.                                                          *)
(* ------------------------------------------------------------------------- *)

fun map f =
    let
      fun m ys ys_xs xs =
          case xs of
            [] => List.revAppend ys_xs
          | x :: xs =>
            let
              val y = f x
              val ys = y :: ys
              val ys_xs = if x == y then ys_xs else (ys,xs)
            in
              m ys ys_xs xs
            end
    in
      fn xs => m [] ([],xs) xs
    end;

fun maps f =
    let
      fun m acc ys ys_xs xs =
          case xs of
            [] => (List.revAppend ys_xs, acc)
          | x :: xs =>
            let
              val (y,acc) = f x acc
              val ys = y :: ys
              val ys_xs = if x == y then ys_xs else (ys,xs)
            in
              m acc ys ys_xs xs
            end
    in
      fn xs => fn acc => m acc [] ([],xs) xs
    end;

local
  fun revTails acc xs =
      case xs of
        [] => acc
      | x :: xs' => revTails ((x,xs) :: acc) xs';
in
  fun revMap f =
      let
        fun m ys same xxss =
            case xxss of
              [] => ys
            | (x,xs) :: xxss =>
              let
                val y = f x
                val same = same andalso x == y
                val ys = if same then xs else y :: ys
              in
                m ys same xxss
              end
      in
        fn xs => m [] true (revTails [] xs)
      end;

  fun revMaps f =
      let
        fun m acc ys same xxss =
            case xxss of
              [] => (ys,acc)
            | (x,xs) :: xxss =>
              let
                val (y,acc) = f x acc
                val same = same andalso x == y
                val ys = if same then xs else y :: ys
              in
                m acc ys same xxss
              end
      in
        fn xs => fn acc => m acc [] true (revTails [] xs)
      end;
end;

fun updateNth (n,x) l =
    let
      val (a,b) = mlibUseful.revDivide l n
    in
      case b of
        [] => raise Subscript
      | h :: t => if x == h then l else List.revAppend (a, x :: t)
    end;

fun setify l =
    let
      val l' = mlibUseful.setify l
    in
      if length l' = length l then l else l'
    end;

(* ------------------------------------------------------------------------- *)
(* Function caching.                                                         *)
(* ------------------------------------------------------------------------- *)

fun cache cmp f =
    let
      val cache = ref (mlibMap.new cmp)
    in
      fn a =>
         case mlibMap.peek (!cache) a of
           SOME b => b
         | NONE =>
           let
             val b = f a
             val () = cache := mlibMap.insert (!cache) (a,b)
           in
             b
           end
    end;

(* ------------------------------------------------------------------------- *)
(* Hash consing.                                                             *)
(* ------------------------------------------------------------------------- *)

fun hashCons cmp = cache cmp mlibUseful.I;

end
(* ========================================================================= *)
(* FIRST ORDER LOGIC SUBSTITUTIONS                                           *)
(* Copyright (c) 2002 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibSubst =
sig

(* ------------------------------------------------------------------------- *)
(* A type of first order logic substitutions.                                *)
(* ------------------------------------------------------------------------- *)

type subst

(* ------------------------------------------------------------------------- *)
(* Basic operations.                                                         *)
(* ------------------------------------------------------------------------- *)

val empty : subst

val null : subst -> bool

val size : subst -> int

val peek : subst -> mlibTerm.var -> mlibTerm.term option

val insert : subst -> mlibTerm.var * mlibTerm.term -> subst

val singleton : mlibTerm.var * mlibTerm.term -> subst

val toList : subst -> (mlibTerm.var * mlibTerm.term) list

val fromList : (mlibTerm.var * mlibTerm.term) list -> subst

val foldl : (mlibTerm.var * mlibTerm.term * 'a -> 'a) -> 'a -> subst -> 'a

val foldr : (mlibTerm.var * mlibTerm.term * 'a -> 'a) -> 'a -> subst -> 'a

val pp : subst mlibPrint.pp

val toString : subst -> string

(* ------------------------------------------------------------------------- *)
(* Normalizing removes identity substitutions.                               *)
(* ------------------------------------------------------------------------- *)

val normalize : subst -> subst

(* ------------------------------------------------------------------------- *)
(* Applying a substitution to a first order logic term.                      *)
(* ------------------------------------------------------------------------- *)

val subst : subst -> mlibTerm.term -> mlibTerm.term

(* ------------------------------------------------------------------------- *)
(* Restricting a substitution to a smaller set of variables.                 *)
(* ------------------------------------------------------------------------- *)

val restrict : subst -> mlibNameSet.set -> subst

val remove : subst -> mlibNameSet.set -> subst

(* ------------------------------------------------------------------------- *)
(* Composing two substitutions so that the following identity holds:         *)
(*                                                                           *)
(* subst (compose sub1 sub2) tm = subst sub2 (subst sub1 tm)                 *)
(* ------------------------------------------------------------------------- *)

val compose : subst -> subst -> subst

(* ------------------------------------------------------------------------- *)
(* Creating the union of two compatible substitutions.                       *)
(* ------------------------------------------------------------------------- *)

val union : subst -> subst -> subst  (* raises Error *)

(* ------------------------------------------------------------------------- *)
(* Substitutions can be inverted iff they are renaming substitutions.        *)
(* ------------------------------------------------------------------------- *)

val invert : subst -> subst  (* raises Error *)

val isRenaming : subst -> bool

(* ------------------------------------------------------------------------- *)
(* Creating a substitution to freshen variables.                             *)
(* ------------------------------------------------------------------------- *)

val freshVars : mlibNameSet.set -> subst

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

val redexes : subst -> mlibNameSet.set

val residueFreeVars : subst -> mlibNameSet.set

val freeVars : subst -> mlibNameSet.set

(* ------------------------------------------------------------------------- *)
(* Functions.                                                                *)
(* ------------------------------------------------------------------------- *)

val functions : subst -> mlibNameAritySet.set

(* ------------------------------------------------------------------------- *)
(* Matching for first order logic terms.                                     *)
(* ------------------------------------------------------------------------- *)

val match : subst -> mlibTerm.term -> mlibTerm.term -> subst  (* raises Error *)

(* ------------------------------------------------------------------------- *)
(* Unification for first order logic terms.                                  *)
(* ------------------------------------------------------------------------- *)

val unify : subst -> mlibTerm.term -> mlibTerm.term -> subst  (* raises Error *)

end
(* ========================================================================= *)
(* FIRST ORDER LOGIC SUBSTITUTIONS                                           *)
(* Copyright (c) 2002 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibSubst :> mlibSubst =
struct

open mlibUseful;

(* ------------------------------------------------------------------------- *)
(* A type of first order logic substitutions.                                *)
(* ------------------------------------------------------------------------- *)

datatype subst = mlibSubst of mlibTerm.term mlibNameMap.map;

(* ------------------------------------------------------------------------- *)
(* Basic operations.                                                         *)
(* ------------------------------------------------------------------------- *)

val empty = mlibSubst (mlibNameMap.new ());

fun null (mlibSubst m) = mlibNameMap.null m;

fun size (mlibSubst m) = mlibNameMap.size m;

fun peek (mlibSubst m) v = mlibNameMap.peek m v;

fun insert (mlibSubst m) v_tm = mlibSubst (mlibNameMap.insert m v_tm);

fun singleton v_tm = insert empty v_tm;

fun toList (mlibSubst m) = mlibNameMap.toList m;

fun fromList l = mlibSubst (mlibNameMap.fromList l);

fun foldl f b (mlibSubst m) = mlibNameMap.foldl f b m;

fun foldr f b (mlibSubst m) = mlibNameMap.foldr f b m;

fun pp sub =
    mlibPrint.ppBracket "<[" "]>"
      (mlibPrint.ppOpList "," (mlibPrint.ppOp2 " |->" mlibName.pp mlibTerm.pp))
      (toList sub);

val toString = mlibPrint.toString pp;

(* ------------------------------------------------------------------------- *)
(* Normalizing removes identity substitutions.                               *)
(* ------------------------------------------------------------------------- *)

local
  fun isNotId (v,tm) = not (mlibTerm.equalVar v tm);
in
  fun normalize (sub as mlibSubst m) =
      let
        val m' = mlibNameMap.filter isNotId m
      in
        if mlibNameMap.size m = mlibNameMap.size m' then sub else mlibSubst m'
      end;
end;

(* ------------------------------------------------------------------------- *)
(* Applying a substitution to a first order logic term.                      *)
(* ------------------------------------------------------------------------- *)

fun subst sub =
    let
      fun tmSub (tm as mlibTerm.Var v) =
          (case peek sub v of
             SOME tm' => if mlibPortable.pointerEqual (tm,tm') then tm else tm'
           | NONE => tm)
        | tmSub (tm as mlibTerm.Fn (f,args)) =
          let
            val args' = mlibSharing.map tmSub args
          in
            if mlibPortable.pointerEqual (args,args') then tm
            else mlibTerm.Fn (f,args')
          end
    in
      fn tm => if null sub then tm else tmSub tm
    end;

(* ------------------------------------------------------------------------- *)
(* Restricting a substitution to a given set of variables.                   *)
(* ------------------------------------------------------------------------- *)

fun restrict (sub as mlibSubst m) varSet =
    let
      fun isRestrictedVar (v,_) = mlibNameSet.member v varSet

      val m' = mlibNameMap.filter isRestrictedVar m
    in
      if mlibNameMap.size m = mlibNameMap.size m' then sub else mlibSubst m'
    end;

fun remove (sub as mlibSubst m) varSet =
    let
      fun isRestrictedVar (v,_) = not (mlibNameSet.member v varSet)

      val m' = mlibNameMap.filter isRestrictedVar m
    in
      if mlibNameMap.size m = mlibNameMap.size m' then sub else mlibSubst m'
    end;

(* ------------------------------------------------------------------------- *)
(* Composing two substitutions so that the following identity holds:         *)
(*                                                                           *)
(* subst (compose sub1 sub2) tm = subst sub2 (subst sub1 tm)                 *)
(* ------------------------------------------------------------------------- *)

fun compose (sub1 as mlibSubst m1) sub2 =
    let
      fun f (v,tm,s) = insert s (v, subst sub2 tm)
    in
      if null sub2 then sub1 else mlibNameMap.foldl f sub2 m1
    end;

(* ------------------------------------------------------------------------- *)
(* Creating the union of two compatible substitutions.                       *)
(* ------------------------------------------------------------------------- *)

local
  fun compatible ((_,tm1),(_,tm2)) =
      if mlibTerm.equal tm1 tm2 then SOME tm1
      else raise Error "mlibSubst.union: incompatible";
in
  fun union (s1 as mlibSubst m1) (s2 as mlibSubst m2) =
      if mlibNameMap.null m1 then s2
      else if mlibNameMap.null m2 then s1
      else mlibSubst (mlibNameMap.union compatible m1 m2);
end;

(* ------------------------------------------------------------------------- *)
(* Substitutions can be inverted iff they are renaming substitutions.        *)
(* ------------------------------------------------------------------------- *)

local
  fun inv (v, mlibTerm.Var w, s) =
      if mlibNameMap.inDomain w s then raise Error "mlibSubst.invert: non-injective"
      else mlibNameMap.insert s (w, mlibTerm.Var v)
    | inv (_, mlibTerm.Fn _, _) = raise Error "mlibSubst.invert: non-variable";
in
  fun invert (mlibSubst m) = mlibSubst (mlibNameMap.foldl inv (mlibNameMap.new ()) m);
end;

val isRenaming = can invert;

(* ------------------------------------------------------------------------- *)
(* Creating a substitution to freshen variables.                             *)
(* ------------------------------------------------------------------------- *)

val freshVars =
    let
      fun add (v,m) = insert m (v, mlibTerm.newVar ())
    in
      mlibNameSet.foldl add empty
    end;

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

val redexes =
    let
      fun add (v,_,s) = mlibNameSet.add s v
    in
      foldl add mlibNameSet.empty
    end;

val residueFreeVars =
    let
      fun add (_,t,s) = mlibNameSet.union s (mlibTerm.freeVars t)
    in
      foldl add mlibNameSet.empty
    end;

val freeVars =
    let
      fun add (v,t,s) = mlibNameSet.union (mlibNameSet.add s v) (mlibTerm.freeVars t)
    in
      foldl add mlibNameSet.empty
    end;

(* ------------------------------------------------------------------------- *)
(* Functions.                                                                *)
(* ------------------------------------------------------------------------- *)

val functions =
    let
      fun add (_,t,s) = mlibNameAritySet.union s (mlibTerm.functions t)
    in
      foldl add mlibNameAritySet.empty
    end;

(* ------------------------------------------------------------------------- *)
(* Matching for first order logic terms.                                     *)
(* ------------------------------------------------------------------------- *)

local
  fun matchList sub [] = sub
    | matchList sub ((mlibTerm.Var v, tm) :: rest) =
      let
        val sub =
            case peek sub v of
              NONE => insert sub (v,tm)
            | SOME tm' =>
              if mlibTerm.equal tm tm' then sub
              else raise Error "mlibSubst.match: incompatible matches"
      in
        matchList sub rest
      end
    | matchList sub ((mlibTerm.Fn (f1,args1), mlibTerm.Fn (f2,args2)) :: rest) =
      if mlibName.equal f1 f2 andalso length args1 = length args2 then
        matchList sub (zip args1 args2 @ rest)
      else raise Error "mlibSubst.match: different structure"
    | matchList _ _ = raise Error "mlibSubst.match: functions can't match vars";
in
  fun match sub tm1 tm2 = matchList sub [(tm1,tm2)];
end;

(* ------------------------------------------------------------------------- *)
(* Unification for first order logic terms.                                  *)
(* ------------------------------------------------------------------------- *)

local
  fun solve sub [] = sub
    | solve sub ((tm1_tm2 as (tm1,tm2)) :: rest) =
      if mlibPortable.pointerEqual tm1_tm2 then solve sub rest
      else solve' sub (subst sub tm1) (subst sub tm2) rest

  and solve' sub (mlibTerm.Var v) tm rest =
      if mlibTerm.equalVar v tm then solve sub rest
      else if mlibTerm.freeIn v tm then raise Error "mlibSubst.unify: occurs check"
      else
        (case peek sub v of
           NONE => solve (compose sub (singleton (v,tm))) rest
         | SOME tm' => solve' sub tm' tm rest)
    | solve' sub tm1 (tm2 as mlibTerm.Var _) rest = solve' sub tm2 tm1 rest
    | solve' sub (mlibTerm.Fn (f1,args1)) (mlibTerm.Fn (f2,args2)) rest =
      if mlibName.equal f1 f2 andalso length args1 = length args2 then
        solve sub (zip args1 args2 @ rest)
      else
        raise Error "mlibSubst.unify: different structure";
in
  fun unify sub tm1 tm2 = solve sub [(tm1,tm2)];
end;

end
(* ========================================================================= *)
(* FIRST ORDER LOGIC ATOMS                                                   *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibAtom =
sig

(* ------------------------------------------------------------------------- *)
(* A type for storing first order logic atoms.                               *)
(* ------------------------------------------------------------------------- *)

type relationName = mlibName.name

type relation = relationName * int

type atom = relationName * mlibTerm.term list

(* ------------------------------------------------------------------------- *)
(* Constructors and destructors.                                             *)
(* ------------------------------------------------------------------------- *)

val name : atom -> relationName

val arguments : atom -> mlibTerm.term list

val arity : atom -> int

val relation : atom -> relation

val functions : atom -> mlibNameAritySet.set

val functionNames : atom -> mlibNameSet.set

(* Binary relations *)

val mkBinop : relationName -> mlibTerm.term * mlibTerm.term -> atom

val destBinop : relationName -> atom -> mlibTerm.term * mlibTerm.term

val isBinop : relationName -> atom -> bool

(* ------------------------------------------------------------------------- *)
(* The size of an atom in symbols.                                           *)
(* ------------------------------------------------------------------------- *)

val symbols : atom -> int

(* ------------------------------------------------------------------------- *)
(* A total comparison function for atoms.                                    *)
(* ------------------------------------------------------------------------- *)

val compare : atom * atom -> order

val equal : atom -> atom -> bool

(* ------------------------------------------------------------------------- *)
(* Subterms.                                                                 *)
(* ------------------------------------------------------------------------- *)

val subterm : atom -> mlibTerm.path -> mlibTerm.term

val subterms : atom -> (mlibTerm.path * mlibTerm.term) list

val replace : atom -> mlibTerm.path * mlibTerm.term -> atom

val find : (mlibTerm.term -> bool) -> atom -> mlibTerm.path option

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

val freeIn : mlibTerm.var -> atom -> bool

val freeVars : atom -> mlibNameSet.set

(* ------------------------------------------------------------------------- *)
(* Substitutions.                                                            *)
(* ------------------------------------------------------------------------- *)

val subst : mlibSubst.subst -> atom -> atom

(* ------------------------------------------------------------------------- *)
(* Matching.                                                                 *)
(* ------------------------------------------------------------------------- *)

val match : mlibSubst.subst -> atom -> atom -> mlibSubst.subst  (* raises Error *)

(* ------------------------------------------------------------------------- *)
(* Unification.                                                              *)
(* ------------------------------------------------------------------------- *)

val unify : mlibSubst.subst -> atom -> atom -> mlibSubst.subst  (* raises Error *)

(* ------------------------------------------------------------------------- *)
(* The equality relation.                                                    *)
(* ------------------------------------------------------------------------- *)

val eqRelationName : relationName

val eqRelation : relation

val mkEq : mlibTerm.term * mlibTerm.term -> atom

val destEq : atom -> mlibTerm.term * mlibTerm.term

val isEq : atom -> bool

val mkRefl : mlibTerm.term -> atom

val destRefl : atom -> mlibTerm.term

val isRefl : atom -> bool

val sym : atom -> atom  (* raises Error if given a refl *)

val lhs : atom -> mlibTerm.term

val rhs : atom -> mlibTerm.term

(* ------------------------------------------------------------------------- *)
(* Special support for terms with type annotations.                          *)
(* ------------------------------------------------------------------------- *)

val typedSymbols : atom -> int

val nonVarTypedSubterms : atom -> (mlibTerm.path * mlibTerm.term) list

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty printing.                                              *)
(* ------------------------------------------------------------------------- *)

val pp : atom mlibPrint.pp

val toString : atom -> string

val fromString : string -> atom

val parse : mlibTerm.term mlibParse.quotation -> atom

end
(* ========================================================================= *)
(* FIRST ORDER LOGIC ATOMS                                                   *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibAtom :> mlibAtom =
struct

open mlibUseful;

(* ------------------------------------------------------------------------- *)
(* A type for storing first order logic atoms.                               *)
(* ------------------------------------------------------------------------- *)

type relationName = mlibName.name;

type relation = relationName * int;

type atom = relationName * mlibTerm.term list;

(* ------------------------------------------------------------------------- *)
(* Constructors and destructors.                                             *)
(* ------------------------------------------------------------------------- *)

fun name ((rel,_) : atom) = rel;

fun arguments ((_,args) : atom) = args;

fun arity atm = length (arguments atm);

fun relation atm = (name atm, arity atm);

val functions =
    let
      fun f (tm,acc) = mlibNameAritySet.union (mlibTerm.functions tm) acc
    in
      fn atm => List.foldl f mlibNameAritySet.empty (arguments atm)
    end;

val functionNames =
    let
      fun f (tm,acc) = mlibNameSet.union (mlibTerm.functionNames tm) acc
    in
      fn atm => List.foldl f mlibNameSet.empty (arguments atm)
    end;

(* Binary relations *)

fun mkBinop p (a,b) : atom = (p,[a,b]);

fun destBinop p (x,[a,b]) =
    if mlibName.equal x p then (a,b) else raise Error "mlibAtom.destBinop: wrong binop"
  | destBinop _ _ = raise Error "mlibAtom.destBinop: not a binop";

fun isBinop p = can (destBinop p);

(* ------------------------------------------------------------------------- *)
(* The size of an atom in symbols.                                           *)
(* ------------------------------------------------------------------------- *)

fun symbols atm =
    List.foldl (fn (tm,z) => mlibTerm.symbols tm + z) 1 (arguments atm);

(* ------------------------------------------------------------------------- *)
(* A total comparison function for atoms.                                    *)
(* ------------------------------------------------------------------------- *)

fun compare ((p1,tms1),(p2,tms2)) =
    case mlibName.compare (p1,p2) of
      LESS => LESS
    | EQUAL => lexCompare mlibTerm.compare (tms1,tms2)
    | GREATER => GREATER;

fun equal atm1 atm2 = compare (atm1,atm2) = EQUAL;

(* ------------------------------------------------------------------------- *)
(* Subterms.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun subterm _ [] = raise Bug "mlibAtom.subterm: empty path"
  | subterm ((_,tms) : atom) (h :: t) =
    if h >= length tms then raise Error "mlibAtom.subterm: bad path"
    else mlibTerm.subterm (List.nth (tms,h)) t;

fun subterms ((_,tms) : atom) =
    let
      fun f ((n,tm),l) = List.map (fn (p,s) => (n :: p, s)) (mlibTerm.subterms tm) @ l
    in
      List.foldl f [] (enumerate tms)
    end;

fun replace _ ([],_) = raise Bug "mlibAtom.replace: empty path"
  | replace (atm as (rel,tms)) (h :: t, res) : atom =
    if h >= length tms then raise Error "mlibAtom.replace: bad path"
    else
      let
        val tm = List.nth (tms,h)
        val tm' = mlibTerm.replace tm (t,res)
      in
        if mlibPortable.pointerEqual (tm,tm') then atm
        else (rel, updateNth (h,tm') tms)
      end;

fun find pred =
    let
      fun f (i,tm) =
          case mlibTerm.find pred tm of
            SOME path => SOME (i :: path)
          | NONE => NONE
    in
      fn (_,tms) : atom => first f (enumerate tms)
    end;

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

fun freeIn v atm = List.exists (mlibTerm.freeIn v) (arguments atm);

val freeVars =
    let
      fun f (tm,acc) = mlibNameSet.union (mlibTerm.freeVars tm) acc
    in
      fn atm => List.foldl f mlibNameSet.empty (arguments atm)
    end;

(* ------------------------------------------------------------------------- *)
(* Substitutions.                                                            *)
(* ------------------------------------------------------------------------- *)

fun subst sub (atm as (p,tms)) : atom =
    let
      val tms' = mlibSharing.map (mlibSubst.subst sub) tms
    in
      if mlibPortable.pointerEqual (tms',tms) then atm else (p,tms')
    end;

(* ------------------------------------------------------------------------- *)
(* Matching.                                                                 *)
(* ------------------------------------------------------------------------- *)

local
  fun matchArg ((tm1,tm2),sub) = mlibSubst.match sub tm1 tm2;
in
  fun match sub (p1,tms1) (p2,tms2) =
      let
        val _ = (mlibName.equal p1 p2 andalso length tms1 = length tms2) orelse
                raise Error "mlibAtom.match"
      in
        List.foldl matchArg sub (zip tms1 tms2)
      end;
end;

(* ------------------------------------------------------------------------- *)
(* Unification.                                                              *)
(* ------------------------------------------------------------------------- *)

local
  fun unifyArg ((tm1,tm2),sub) = mlibSubst.unify sub tm1 tm2;
in
  fun unify sub (p1,tms1) (p2,tms2) =
      let
        val _ = (mlibName.equal p1 p2 andalso length tms1 = length tms2) orelse
                raise Error "mlibAtom.unify"
      in
        List.foldl unifyArg sub (zip tms1 tms2)
      end;
end;

(* ------------------------------------------------------------------------- *)
(* The equality relation.                                                    *)
(* ------------------------------------------------------------------------- *)

val eqRelationName = mlibName.fromString "=";

val eqRelationArity = 2;

val eqRelation = (eqRelationName,eqRelationArity);

val mkEq = mkBinop eqRelationName;

fun destEq x = destBinop eqRelationName x;

fun isEq x = isBinop eqRelationName x;

fun mkRefl tm = mkEq (tm,tm);

fun destRefl atm =
    let
      val (l,r) = destEq atm
      val _ = mlibTerm.equal l r orelse raise Error "mlibAtom.destRefl"
    in
      l
    end;

fun isRefl x = can destRefl x;

fun sym atm =
    let
      val (l,r) = destEq atm
      val _ = not (mlibTerm.equal l r) orelse raise Error "mlibAtom.sym: refl"
    in
      mkEq (r,l)
    end;

fun lhs atm = fst (destEq atm);

fun rhs atm = snd (destEq atm);

(* ------------------------------------------------------------------------- *)
(* Special support for terms with type annotations.                          *)
(* ------------------------------------------------------------------------- *)

fun typedSymbols ((_,tms) : atom) =
    List.foldl (fn (tm,z) => mlibTerm.typedSymbols tm + z) 1 tms;

fun nonVarTypedSubterms (_,tms) =
    let
      fun addArg ((n,arg),acc) =
          let
            fun addTm ((path,tm),acc) = (n :: path, tm) :: acc
          in
            List.foldl addTm acc (mlibTerm.nonVarTypedSubterms arg)
          end
    in
      List.foldl addArg [] (enumerate tms)
    end;

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty printing.                                              *)
(* ------------------------------------------------------------------------- *)

val pp = mlibPrint.ppMap mlibTerm.Fn mlibTerm.pp;

val toString = mlibPrint.toString pp;

fun fromString s = mlibTerm.destFn (mlibTerm.fromString s);

val parse = mlibParse.parseQuotation mlibTerm.toString fromString;

end

structure mlibAtomOrdered =
struct type t = mlibAtom.atom val compare = mlibAtom.compare end

structure mlibAtomMap = mlibKeyMap (mlibAtomOrdered);

structure mlibAtomSet = mlibElementSet (mlibAtomMap);
(* ========================================================================= *)
(* FIRST ORDER LOGIC FORMULAS                                                *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibFormula =
sig

(* ------------------------------------------------------------------------- *)
(* A type of first order logic formulas.                                     *)
(* ------------------------------------------------------------------------- *)

datatype formula =
    True
  | False
  | mlibAtom of mlibAtom.atom
  | Not of formula
  | And of formula * formula
  | Or of formula * formula
  | Imp of formula * formula
  | Iff of formula * formula
  | Forall of mlibTerm.var * formula
  | Exists of mlibTerm.var * formula

(* ------------------------------------------------------------------------- *)
(* Constructors and destructors.                                             *)
(* ------------------------------------------------------------------------- *)

(* Booleans *)

val mkBoolean : bool -> formula

val destBoolean : formula -> bool

val isBoolean : formula -> bool

val isTrue : formula -> bool

val isFalse : formula -> bool

(* Functions *)

val functions : formula -> mlibNameAritySet.set

val functionNames : formula -> mlibNameSet.set

(* Relations *)

val relations : formula -> mlibNameAritySet.set

val relationNames : formula -> mlibNameSet.set

(* Atoms *)

val destAtom : formula -> mlibAtom.atom

val isAtom : formula -> bool

(* Negations *)

val destNeg : formula -> formula

val isNeg : formula -> bool

val stripNeg : formula -> int * formula

(* Conjunctions *)

val listMkConj : formula list -> formula

val stripConj : formula -> formula list

val flattenConj : formula -> formula list

(* Disjunctions *)

val listMkDisj : formula list -> formula

val stripDisj : formula -> formula list

val flattenDisj : formula -> formula list

(* Equivalences *)

val listMkEquiv : formula list -> formula

val stripEquiv : formula -> formula list

val flattenEquiv : formula -> formula list

(* Universal quantification *)

val destForall : formula -> mlibTerm.var * formula

val isForall : formula -> bool

val listMkForall : mlibTerm.var list * formula -> formula

val setMkForall : mlibNameSet.set * formula -> formula

val stripForall : formula -> mlibTerm.var list * formula

(* Existential quantification *)

val destExists : formula -> mlibTerm.var * formula

val isExists : formula -> bool

val listMkExists : mlibTerm.var list * formula -> formula

val setMkExists : mlibNameSet.set * formula -> formula

val stripExists : formula -> mlibTerm.var list * formula

(* ------------------------------------------------------------------------- *)
(* The size of a formula in symbols.                                         *)
(* ------------------------------------------------------------------------- *)

val symbols : formula -> int

(* ------------------------------------------------------------------------- *)
(* A total comparison function for formulas.                                 *)
(* ------------------------------------------------------------------------- *)

val compare : formula * formula -> order

val equal : formula -> formula -> bool

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

val freeIn : mlibTerm.var -> formula -> bool

val freeVars : formula -> mlibNameSet.set

val freeVarsList : formula list -> mlibNameSet.set

val specialize : formula -> formula

val generalize : formula -> formula

(* ------------------------------------------------------------------------- *)
(* Substitutions.                                                            *)
(* ------------------------------------------------------------------------- *)

val subst : mlibSubst.subst -> formula -> formula

(* ------------------------------------------------------------------------- *)
(* The equality relation.                                                    *)
(* ------------------------------------------------------------------------- *)

val mkEq : mlibTerm.term * mlibTerm.term -> formula

val destEq : formula -> mlibTerm.term * mlibTerm.term

val isEq : formula -> bool

val mkNeq : mlibTerm.term * mlibTerm.term -> formula

val destNeq : formula -> mlibTerm.term * mlibTerm.term

val isNeq : formula -> bool

val mkRefl : mlibTerm.term -> formula

val destRefl : formula -> mlibTerm.term

val isRefl : formula -> bool

val sym : formula -> formula  (* raises Error if given a refl *)

val lhs : formula -> mlibTerm.term

val rhs : formula -> mlibTerm.term

(* ------------------------------------------------------------------------- *)
(* Splitting goals.                                                          *)
(* ------------------------------------------------------------------------- *)

val splitGoal : formula -> formula list

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty-printing.                                              *)
(* ------------------------------------------------------------------------- *)

type quotation = formula mlibParse.quotation

val pp : formula mlibPrint.pp

val toString : formula -> string

val fromString : string -> formula

val parse : quotation -> formula

end
(* ========================================================================= *)
(* FIRST ORDER LOGIC FORMULAS                                                *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibFormula :> mlibFormula =
struct

open mlibUseful;

(* ------------------------------------------------------------------------- *)
(* A type of first order logic formulas.                                     *)
(* ------------------------------------------------------------------------- *)

datatype formula =
    True
  | False
  | mlibAtom of mlibAtom.atom
  | Not of formula
  | And of formula * formula
  | Or of formula * formula
  | Imp of formula * formula
  | Iff of formula * formula
  | Forall of mlibTerm.var * formula
  | Exists of mlibTerm.var * formula;

(* ------------------------------------------------------------------------- *)
(* Constructors and destructors.                                             *)
(* ------------------------------------------------------------------------- *)

(* Booleans *)

fun mkBoolean true = True
  | mkBoolean false = False;

fun destBoolean True = true
  | destBoolean False = false
  | destBoolean _ = raise Error "destBoolean";

val isBoolean = can destBoolean;

fun isTrue fm =
    case fm of
      True => true
    | _ => false;

fun isFalse fm =
    case fm of
      False => true
    | _ => false;

(* Functions *)

local
  fun funcs fs [] = fs
    | funcs fs (True :: fms) = funcs fs fms
    | funcs fs (False :: fms) = funcs fs fms
    | funcs fs (mlibAtom atm :: fms) =
      funcs (mlibNameAritySet.union (mlibAtom.functions atm) fs) fms
    | funcs fs (Not p :: fms) = funcs fs (p :: fms)
    | funcs fs (And (p,q) :: fms) = funcs fs (p :: q :: fms)
    | funcs fs (Or (p,q) :: fms) = funcs fs (p :: q :: fms)
    | funcs fs (Imp (p,q) :: fms) = funcs fs (p :: q :: fms)
    | funcs fs (Iff (p,q) :: fms) = funcs fs (p :: q :: fms)
    | funcs fs (Forall (_,p) :: fms) = funcs fs (p :: fms)
    | funcs fs (Exists (_,p) :: fms) = funcs fs (p :: fms);
in
  fun functions fm = funcs mlibNameAritySet.empty [fm];
end;

local
  fun funcs fs [] = fs
    | funcs fs (True :: fms) = funcs fs fms
    | funcs fs (False :: fms) = funcs fs fms
    | funcs fs (mlibAtom atm :: fms) =
      funcs (mlibNameSet.union (mlibAtom.functionNames atm) fs) fms
    | funcs fs (Not p :: fms) = funcs fs (p :: fms)
    | funcs fs (And (p,q) :: fms) = funcs fs (p :: q :: fms)
    | funcs fs (Or (p,q) :: fms) = funcs fs (p :: q :: fms)
    | funcs fs (Imp (p,q) :: fms) = funcs fs (p :: q :: fms)
    | funcs fs (Iff (p,q) :: fms) = funcs fs (p :: q :: fms)
    | funcs fs (Forall (_,p) :: fms) = funcs fs (p :: fms)
    | funcs fs (Exists (_,p) :: fms) = funcs fs (p :: fms);
in
  fun functionNames fm = funcs mlibNameSet.empty [fm];
end;

(* Relations *)

local
  fun rels fs [] = fs
    | rels fs (True :: fms) = rels fs fms
    | rels fs (False :: fms) = rels fs fms
    | rels fs (mlibAtom atm :: fms) =
      rels (mlibNameAritySet.add fs (mlibAtom.relation atm)) fms
    | rels fs (Not p :: fms) = rels fs (p :: fms)
    | rels fs (And (p,q) :: fms) = rels fs (p :: q :: fms)
    | rels fs (Or (p,q) :: fms) = rels fs (p :: q :: fms)
    | rels fs (Imp (p,q) :: fms) = rels fs (p :: q :: fms)
    | rels fs (Iff (p,q) :: fms) = rels fs (p :: q :: fms)
    | rels fs (Forall (_,p) :: fms) = rels fs (p :: fms)
    | rels fs (Exists (_,p) :: fms) = rels fs (p :: fms);
in
  fun relations fm = rels mlibNameAritySet.empty [fm];
end;

local
  fun rels fs [] = fs
    | rels fs (True :: fms) = rels fs fms
    | rels fs (False :: fms) = rels fs fms
    | rels fs (mlibAtom atm :: fms) = rels (mlibNameSet.add fs (mlibAtom.name atm)) fms
    | rels fs (Not p :: fms) = rels fs (p :: fms)
    | rels fs (And (p,q) :: fms) = rels fs (p :: q :: fms)
    | rels fs (Or (p,q) :: fms) = rels fs (p :: q :: fms)
    | rels fs (Imp (p,q) :: fms) = rels fs (p :: q :: fms)
    | rels fs (Iff (p,q) :: fms) = rels fs (p :: q :: fms)
    | rels fs (Forall (_,p) :: fms) = rels fs (p :: fms)
    | rels fs (Exists (_,p) :: fms) = rels fs (p :: fms);
in
  fun relationNames fm = rels mlibNameSet.empty [fm];
end;

(* Atoms *)

fun destAtom (mlibAtom atm) = atm
  | destAtom _ = raise Error "mlibFormula.destAtom";

val isAtom = can destAtom;

(* Negations *)

fun destNeg (Not p) = p
  | destNeg _ = raise Error "mlibFormula.destNeg";

val isNeg = can destNeg;

val stripNeg =
    let
      fun strip n (Not fm) = strip (n + 1) fm
        | strip n fm = (n,fm)
    in
      strip 0
    end;

(* Conjunctions *)

fun listMkConj fms =
    case List.rev fms of [] => True | fm :: fms => List.foldl And fm fms;

local
  fun strip cs (And (p,q)) = strip (p :: cs) q
    | strip cs fm = List.rev (fm :: cs);
in
  fun stripConj True = []
    | stripConj fm = strip [] fm;
end;

val flattenConj =
    let
      fun flat acc [] = acc
        | flat acc (And (p,q) :: fms) = flat acc (q :: p :: fms)
        | flat acc (True :: fms) = flat acc fms
        | flat acc (fm :: fms) = flat (fm :: acc) fms
    in
      fn fm => flat [] [fm]
    end;

(* Disjunctions *)

fun listMkDisj fms =
    case List.rev fms of [] => False | fm :: fms => List.foldl Or fm fms;

local
  fun strip cs (Or (p,q)) = strip (p :: cs) q
    | strip cs fm = List.rev (fm :: cs);
in
  fun stripDisj False = []
    | stripDisj fm = strip [] fm;
end;

val flattenDisj =
    let
      fun flat acc [] = acc
        | flat acc (Or (p,q) :: fms) = flat acc (q :: p :: fms)
        | flat acc (False :: fms) = flat acc fms
        | flat acc (fm :: fms) = flat (fm :: acc) fms
    in
      fn fm => flat [] [fm]
    end;

(* Equivalences *)

fun listMkEquiv fms =
    case List.rev fms of [] => True | fm :: fms => List.foldl Iff fm fms;

local
  fun strip cs (Iff (p,q)) = strip (p :: cs) q
    | strip cs fm = List.rev (fm :: cs);
in
  fun stripEquiv True = []
    | stripEquiv fm = strip [] fm;
end;

val flattenEquiv =
    let
      fun flat acc [] = acc
        | flat acc (Iff (p,q) :: fms) = flat acc (q :: p :: fms)
        | flat acc (True :: fms) = flat acc fms
        | flat acc (fm :: fms) = flat (fm :: acc) fms
    in
      fn fm => flat [] [fm]
    end;

(* Universal quantifiers *)

fun destForall (Forall v_f) = v_f
  | destForall _ = raise Error "destForall";

val isForall = can destForall;

fun listMkForall ([],body) = body
  | listMkForall (v :: vs, body) = Forall (v, listMkForall (vs,body));

fun setMkForall (vs,body) = mlibNameSet.foldr Forall body vs;

local
  fun strip vs (Forall (v,b)) = strip (v :: vs) b
    | strip vs tm = (List.rev vs, tm);
in
  val stripForall = strip [];
end;

(* Existential quantifiers *)

fun destExists (Exists v_f) = v_f
  | destExists _ = raise Error "destExists";

val isExists = can destExists;

fun listMkExists ([],body) = body
  | listMkExists (v :: vs, body) = Exists (v, listMkExists (vs,body));

fun setMkExists (vs,body) = mlibNameSet.foldr Exists body vs;

local
  fun strip vs (Exists (v,b)) = strip (v :: vs) b
    | strip vs tm = (List.rev vs, tm);
in
  val stripExists = strip [];
end;

(* ------------------------------------------------------------------------- *)
(* The size of a formula in symbols.                                         *)
(* ------------------------------------------------------------------------- *)

local
  fun sz n [] = n
    | sz n (True :: fms) = sz (n + 1) fms
    | sz n (False :: fms) = sz (n + 1) fms
    | sz n (mlibAtom atm :: fms) = sz (n + mlibAtom.symbols atm) fms
    | sz n (Not p :: fms) = sz (n + 1) (p :: fms)
    | sz n (And (p,q) :: fms) = sz (n + 1) (p :: q :: fms)
    | sz n (Or (p,q) :: fms) = sz (n + 1) (p :: q :: fms)
    | sz n (Imp (p,q) :: fms) = sz (n + 1) (p :: q :: fms)
    | sz n (Iff (p,q) :: fms) = sz (n + 1) (p :: q :: fms)
    | sz n (Forall (_,p) :: fms) = sz (n + 1) (p :: fms)
    | sz n (Exists (_,p) :: fms) = sz (n + 1) (p :: fms);
in
  fun symbols fm = sz 0 [fm];
end;

(* ------------------------------------------------------------------------- *)
(* A total comparison function for formulas.                                 *)
(* ------------------------------------------------------------------------- *)

local
  fun cmp [] = EQUAL
    | cmp (f1_f2 :: fs) =
      if mlibPortable.pointerEqual f1_f2 then cmp fs
      else
        case f1_f2 of
          (True,True) => cmp fs
        | (True,_) => LESS
        | (_,True) => GREATER
        | (False,False) => cmp fs
        | (False,_) => LESS
        | (_,False) => GREATER
        | (mlibAtom atm1, mlibAtom atm2) =>
          (case mlibAtom.compare (atm1,atm2) of
             LESS => LESS
           | EQUAL => cmp fs
           | GREATER => GREATER)
        | (mlibAtom _, _) => LESS
        | (_, mlibAtom _) => GREATER
        | (Not p1, Not p2) => cmp ((p1,p2) :: fs)
        | (Not _, _) => LESS
        | (_, Not _) => GREATER
        | (And (p1,q1), And (p2,q2)) => cmp ((p1,p2) :: (q1,q2) :: fs)
        | (And _, _) => LESS
        | (_, And _) => GREATER
        | (Or (p1,q1), Or (p2,q2)) => cmp ((p1,p2) :: (q1,q2) :: fs)
        | (Or _, _) => LESS
        | (_, Or _) => GREATER
        | (Imp (p1,q1), Imp (p2,q2)) => cmp ((p1,p2) :: (q1,q2) :: fs)
        | (Imp _, _) => LESS
        | (_, Imp _) => GREATER
        | (Iff (p1,q1), Iff (p2,q2)) => cmp ((p1,p2) :: (q1,q2) :: fs)
        | (Iff _, _) => LESS
        | (_, Iff _) => GREATER
        | (Forall (v1,p1), Forall (v2,p2)) =>
          (case mlibName.compare (v1,v2) of
             LESS => LESS
           | EQUAL => cmp ((p1,p2) :: fs)
           | GREATER => GREATER)
        | (Forall _, Exists _) => LESS
        | (Exists _, Forall _) => GREATER
        | (Exists (v1,p1), Exists (v2,p2)) =>
          (case mlibName.compare (v1,v2) of
             LESS => LESS
           | EQUAL => cmp ((p1,p2) :: fs)
           | GREATER => GREATER);
in
  fun compare fm1_fm2 = cmp [fm1_fm2];
end;

fun equal fm1 fm2 = compare (fm1,fm2) = EQUAL;

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

fun freeIn v =
    let
      fun f [] = false
        | f (True :: fms) = f fms
        | f (False :: fms) = f fms
        | f (mlibAtom atm :: fms) = mlibAtom.freeIn v atm orelse f fms
        | f (Not p :: fms) = f (p :: fms)
        | f (And (p,q) :: fms) = f (p :: q :: fms)
        | f (Or (p,q) :: fms) = f (p :: q :: fms)
        | f (Imp (p,q) :: fms) = f (p :: q :: fms)
        | f (Iff (p,q) :: fms) = f (p :: q :: fms)
        | f (Forall (w,p) :: fms) =
          if mlibName.equal v w then f fms else f (p :: fms)
        | f (Exists (w,p) :: fms) =
          if mlibName.equal v w then f fms else f (p :: fms)
    in
      fn fm => f [fm]
    end;

local
  fun fv vs [] = vs
    | fv vs ((_,True) :: fms) = fv vs fms
    | fv vs ((_,False) :: fms) = fv vs fms
    | fv vs ((bv, mlibAtom atm) :: fms) =
      fv (mlibNameSet.union vs (mlibNameSet.difference (mlibAtom.freeVars atm) bv)) fms
    | fv vs ((bv, Not p) :: fms) = fv vs ((bv,p) :: fms)
    | fv vs ((bv, And (p,q)) :: fms) = fv vs ((bv,p) :: (bv,q) :: fms)
    | fv vs ((bv, Or (p,q)) :: fms) = fv vs ((bv,p) :: (bv,q) :: fms)
    | fv vs ((bv, Imp (p,q)) :: fms) = fv vs ((bv,p) :: (bv,q) :: fms)
    | fv vs ((bv, Iff (p,q)) :: fms) = fv vs ((bv,p) :: (bv,q) :: fms)
    | fv vs ((bv, Forall (v,p)) :: fms) = fv vs ((mlibNameSet.add bv v, p) :: fms)
    | fv vs ((bv, Exists (v,p)) :: fms) = fv vs ((mlibNameSet.add bv v, p) :: fms);

  fun add (fm,vs) = fv vs [(mlibNameSet.empty,fm)];
in
  fun freeVars fm = add (fm,mlibNameSet.empty);

  fun freeVarsList fms = List.foldl add mlibNameSet.empty fms;
end;

fun specialize fm = snd (stripForall fm);

fun generalize fm = listMkForall (mlibNameSet.toList (freeVars fm), fm);

(* ------------------------------------------------------------------------- *)
(* Substitutions.                                                            *)
(* ------------------------------------------------------------------------- *)

local
  fun substCheck sub fm = if mlibSubst.null sub then fm else substFm sub fm

  and substFm sub fm =
      case fm of
        True => fm
      | False => fm
      | mlibAtom (p,tms) =>
        let
          val tms' = mlibSharing.map (mlibSubst.subst sub) tms
        in
          if mlibPortable.pointerEqual (tms,tms') then fm else mlibAtom (p,tms')
        end
      | Not p =>
        let
          val p' = substFm sub p
        in
          if mlibPortable.pointerEqual (p,p') then fm else Not p'
        end
      | And (p,q) => substConn sub fm And p q
      | Or (p,q) => substConn sub fm Or p q
      | Imp (p,q) => substConn sub fm Imp p q
      | Iff (p,q) => substConn sub fm Iff p q
      | Forall (v,p) => substQuant sub fm Forall v p
      | Exists (v,p) => substQuant sub fm Exists v p

  and substConn sub fm conn p q =
      let
        val p' = substFm sub p
        and q' = substFm sub q
      in
        if mlibPortable.pointerEqual (p,p') andalso
           mlibPortable.pointerEqual (q,q')
        then fm
        else conn (p',q')
      end

  and substQuant sub fm quant v p =
      let
        val v' =
            let
              fun f (w,s) =
                  if mlibName.equal w v then s
                  else
                    case mlibSubst.peek sub w of
                      NONE => mlibNameSet.add s w
                    | SOME tm => mlibNameSet.union s (mlibTerm.freeVars tm)

              val vars = freeVars p
              val vars = mlibNameSet.foldl f mlibNameSet.empty vars
            in
              mlibTerm.variantPrime vars v
            end

        val sub =
            if mlibName.equal v v' then mlibSubst.remove sub (mlibNameSet.singleton v)
            else mlibSubst.insert sub (v, mlibTerm.Var v')

        val p' = substCheck sub p
      in
        if mlibName.equal v v' andalso mlibPortable.pointerEqual (p,p') then fm
        else quant (v',p')
      end;
in
  val subst = substCheck;
end;

(* ------------------------------------------------------------------------- *)
(* The equality relation.                                                    *)
(* ------------------------------------------------------------------------- *)

fun mkEq a_b = mlibAtom (mlibAtom.mkEq a_b);

fun destEq fm = mlibAtom.destEq (destAtom fm);

val isEq = can destEq;

fun mkNeq a_b = Not (mkEq a_b);

fun destNeq (Not fm) = destEq fm
  | destNeq _ = raise Error "mlibFormula.destNeq";

val isNeq = can destNeq;

fun mkRefl tm = mlibAtom (mlibAtom.mkRefl tm);

fun destRefl fm = mlibAtom.destRefl (destAtom fm);

val isRefl = can destRefl;

fun sym fm = mlibAtom (mlibAtom.sym (destAtom fm));

fun lhs fm = fst (destEq fm);

fun rhs fm = snd (destEq fm);

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty-printing.                                              *)
(* ------------------------------------------------------------------------- *)

type quotation = formula mlibParse.quotation;

val truthName = mlibName.fromString "T"
and falsityName = mlibName.fromString "F"
and conjunctionName = mlibName.fromString "/\\"
and disjunctionName = mlibName.fromString "\\/"
and implicationName = mlibName.fromString "==>"
and equivalenceName = mlibName.fromString "<=>"
and universalName = mlibName.fromString "!"
and existentialName = mlibName.fromString "?";

local
  fun demote True = mlibTerm.Fn (truthName,[])
    | demote False = mlibTerm.Fn (falsityName,[])
    | demote (mlibAtom (p,tms)) = mlibTerm.Fn (p,tms)
    | demote (Not p) =
      let
        val ref s = mlibTerm.negation
      in
        mlibTerm.Fn (mlibName.fromString s, [demote p])
      end
    | demote (And (p,q)) = mlibTerm.Fn (conjunctionName, [demote p, demote q])
    | demote (Or (p,q)) = mlibTerm.Fn (disjunctionName, [demote p, demote q])
    | demote (Imp (p,q)) = mlibTerm.Fn (implicationName, [demote p, demote q])
    | demote (Iff (p,q)) = mlibTerm.Fn (equivalenceName, [demote p, demote q])
    | demote (Forall (v,b)) = mlibTerm.Fn (universalName, [mlibTerm.Var v, demote b])
    | demote (Exists (v,b)) =
      mlibTerm.Fn (existentialName, [mlibTerm.Var v, demote b]);
in
  fun pp fm = mlibTerm.pp (demote fm);
end;

val toString = mlibPrint.toString pp;

local
  fun isQuant [mlibTerm.Var _, _] = true
    | isQuant _ = false;

  fun promote (mlibTerm.Var v) = mlibAtom (v,[])
    | promote (mlibTerm.Fn (f,tms)) =
      if mlibName.equal f truthName andalso List.null tms then
        True
      else if mlibName.equal f falsityName andalso List.null tms then
        False
      else if mlibName.toString f = !mlibTerm.negation andalso length tms = 1 then
        Not (promote (hd tms))
      else if mlibName.equal f conjunctionName andalso length tms = 2 then
        And (promote (hd tms), promote (List.nth (tms,1)))
      else if mlibName.equal f disjunctionName andalso length tms = 2 then
        Or (promote (hd tms), promote (List.nth (tms,1)))
      else if mlibName.equal f implicationName andalso length tms = 2 then
        Imp (promote (hd tms), promote (List.nth (tms,1)))
      else if mlibName.equal f equivalenceName andalso length tms = 2 then
        Iff (promote (hd tms), promote (List.nth (tms,1)))
      else if mlibName.equal f universalName andalso isQuant tms then
        Forall (mlibTerm.destVar (hd tms), promote (List.nth (tms,1)))
      else if mlibName.equal f existentialName andalso isQuant tms then
        Exists (mlibTerm.destVar (hd tms), promote (List.nth (tms,1)))
      else
        mlibAtom (f,tms);
in
  fun fromString s = promote (mlibTerm.fromString s);
end;

val parse = mlibParse.parseQuotation toString fromString;

(* ------------------------------------------------------------------------- *)
(* Splitting goals.                                                          *)
(* ------------------------------------------------------------------------- *)

local
  fun add_asms asms goal =
      if List.null asms then goal else Imp (listMkConj (List.rev asms), goal);

  fun add_var_asms asms v goal = add_asms asms (Forall (v,goal));

  fun split asms pol fm =
      case (pol,fm) of
        (* Positive splittables *)
        (true,True) => []
      | (true, Not f) => split asms false f
      | (true, And (f1,f2)) => split asms true f1 @ split (f1 :: asms) true f2
      | (true, Or (f1,f2)) => split (Not f1 :: asms) true f2
      | (true, Imp (f1,f2)) => split (f1 :: asms) true f2
      | (true, Iff (f1,f2)) =>
        split (f1 :: asms) true f2 @ split (f2 :: asms) true f1
      | (true, Forall (v,f)) => List.map (add_var_asms asms v) (split [] true f)
        (* Negative splittables *)
      | (false,False) => []
      | (false, Not f) => split asms true f
      | (false, And (f1,f2)) => split (f1 :: asms) false f2
      | (false, Or (f1,f2)) =>
        split asms false f1 @ split (Not f1 :: asms) false f2
      | (false, Imp (f1,f2)) => split asms true f1 @ split (f1 :: asms) false f2
      | (false, Iff (f1,f2)) =>
        split (f1 :: asms) false f2 @ split (f2 :: asms) false f1
      | (false, Exists (v,f)) => List.map (add_var_asms asms v) (split [] false f)
        (* Unsplittables *)
      | _ => [add_asms asms (if pol then fm else Not fm)];
in
  fun splitGoal fm = split [] true fm;
end;

(*MetisTrace3
val splitGoal = fn fm =>
    let
      val result = splitGoal fm
      val () = mlibPrint.trace pp "mlibFormula.splitGoal: fm" fm
      val () = mlibPrint.trace (mlibPrint.ppList pp) "mlibFormula.splitGoal: result" result
    in
      result
    end;
*)

end

structure mlibFormulaOrdered =
struct type t = mlibFormula.formula val compare = mlibFormula.compare end

structure mlibFormulaMap = mlibKeyMap (mlibFormulaOrdered);

structure mlibFormulaSet = mlibElementSet (mlibFormulaMap);
(* ========================================================================= *)
(* FIRST ORDER LOGIC LITERALS                                                *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibLiteral =
sig

(* ------------------------------------------------------------------------- *)
(* A type for storing first order logic literals.                            *)
(* ------------------------------------------------------------------------- *)

type polarity = bool

type literal = polarity * mlibAtom.atom

(* ------------------------------------------------------------------------- *)
(* Constructors and destructors.                                             *)
(* ------------------------------------------------------------------------- *)

val polarity : literal -> polarity

val atom : literal -> mlibAtom.atom

val name : literal -> mlibAtom.relationName

val arguments : literal -> mlibTerm.term list

val arity : literal -> int

val positive : literal -> bool

val negative : literal -> bool

val negate : literal -> literal

val relation : literal -> mlibAtom.relation

val functions : literal -> mlibNameAritySet.set

val functionNames : literal -> mlibNameSet.set

(* Binary relations *)

val mkBinop : mlibAtom.relationName -> polarity * mlibTerm.term * mlibTerm.term -> literal

val destBinop : mlibAtom.relationName -> literal -> polarity * mlibTerm.term * mlibTerm.term

val isBinop : mlibAtom.relationName -> literal -> bool

(* Formulas *)

val toFormula : literal -> mlibFormula.formula

val fromFormula : mlibFormula.formula -> literal

(* ------------------------------------------------------------------------- *)
(* The size of a literal in symbols.                                         *)
(* ------------------------------------------------------------------------- *)

val symbols : literal -> int

(* ------------------------------------------------------------------------- *)
(* A total comparison function for literals.                                 *)
(* ------------------------------------------------------------------------- *)

val compare : literal * literal -> order  (* negative < positive *)

val equal : literal -> literal -> bool

(* ------------------------------------------------------------------------- *)
(* Subterms.                                                                 *)
(* ------------------------------------------------------------------------- *)

val subterm : literal -> mlibTerm.path -> mlibTerm.term

val subterms : literal -> (mlibTerm.path * mlibTerm.term) list

val replace : literal -> mlibTerm.path * mlibTerm.term -> literal

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

val freeIn : mlibTerm.var -> literal -> bool

val freeVars : literal -> mlibNameSet.set

(* ------------------------------------------------------------------------- *)
(* Substitutions.                                                            *)
(* ------------------------------------------------------------------------- *)

val subst : mlibSubst.subst -> literal -> literal

(* ------------------------------------------------------------------------- *)
(* Matching.                                                                 *)
(* ------------------------------------------------------------------------- *)

val match :  (* raises Error *)
    mlibSubst.subst -> literal -> literal -> mlibSubst.subst

(* ------------------------------------------------------------------------- *)
(* Unification.                                                              *)
(* ------------------------------------------------------------------------- *)

val unify :  (* raises Error *)
    mlibSubst.subst -> literal -> literal -> mlibSubst.subst

(* ------------------------------------------------------------------------- *)
(* The equality relation.                                                    *)
(* ------------------------------------------------------------------------- *)

val mkEq : mlibTerm.term * mlibTerm.term -> literal

val destEq : literal -> mlibTerm.term * mlibTerm.term

val isEq : literal -> bool

val mkNeq : mlibTerm.term * mlibTerm.term -> literal

val destNeq : literal -> mlibTerm.term * mlibTerm.term

val isNeq : literal -> bool

val mkRefl : mlibTerm.term -> literal

val destRefl : literal -> mlibTerm.term

val isRefl : literal -> bool

val mkIrrefl : mlibTerm.term -> literal

val destIrrefl : literal -> mlibTerm.term

val isIrrefl : literal -> bool

(* The following work with both equalities and disequalities *)

val sym : literal -> literal  (* raises Error if given a refl or irrefl *)

val lhs : literal -> mlibTerm.term

val rhs : literal -> mlibTerm.term

(* ------------------------------------------------------------------------- *)
(* Special support for terms with type annotations.                          *)
(* ------------------------------------------------------------------------- *)

val typedSymbols : literal -> int

val nonVarTypedSubterms : literal -> (mlibTerm.path * mlibTerm.term) list

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty-printing.                                              *)
(* ------------------------------------------------------------------------- *)

val pp : literal mlibPrint.pp

val toString : literal -> string

val fromString : string -> literal

val parse : mlibTerm.term mlibParse.quotation -> literal

end
(* ========================================================================= *)
(* FIRST ORDER LOGIC LITERALS                                                *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibLiteral :> mlibLiteral =
struct

open mlibUseful;

(* ------------------------------------------------------------------------- *)
(* A type for storing first order logic literals.                            *)
(* ------------------------------------------------------------------------- *)

type polarity = bool;

type literal = polarity * mlibAtom.atom;

(* ------------------------------------------------------------------------- *)
(* Constructors and destructors.                                             *)
(* ------------------------------------------------------------------------- *)

fun polarity ((pol,_) : literal) = pol;

fun atom ((_,atm) : literal) = atm;

fun name lit = mlibAtom.name (atom lit);

fun arguments lit = mlibAtom.arguments (atom lit);

fun arity lit = mlibAtom.arity (atom lit);

fun positive lit = polarity lit;

fun negative lit = not (polarity lit);

fun negate (pol,atm) : literal = (not pol, atm)

fun relation lit = mlibAtom.relation (atom lit);

fun functions lit = mlibAtom.functions (atom lit);

fun functionNames lit = mlibAtom.functionNames (atom lit);

(* Binary relations *)

fun mkBinop rel (pol,a,b) : literal = (pol, mlibAtom.mkBinop rel (a,b));

fun destBinop rel ((pol,atm) : literal) =
    case mlibAtom.destBinop rel atm of (a,b) => (pol,a,b);

fun isBinop rel = can (destBinop rel);

(* Formulas *)

fun toFormula (true,atm) = mlibFormula.mlibAtom atm
  | toFormula (false,atm) = mlibFormula.Not (mlibFormula.mlibAtom atm);

fun fromFormula (mlibFormula.mlibAtom atm) = (true,atm)
  | fromFormula (mlibFormula.Not (mlibFormula.mlibAtom atm)) = (false,atm)
  | fromFormula _ = raise Error "mlibLiteral.fromFormula";

(* ------------------------------------------------------------------------- *)
(* The size of a literal in symbols.                                         *)
(* ------------------------------------------------------------------------- *)

fun symbols ((_,atm) : literal) = mlibAtom.symbols atm;

(* ------------------------------------------------------------------------- *)
(* A total comparison function for literals.                                 *)
(* ------------------------------------------------------------------------- *)

val compare = prodCompare boolCompare mlibAtom.compare;

fun equal (p1,atm1) (p2,atm2) = p1 = p2 andalso mlibAtom.equal atm1 atm2;

(* ------------------------------------------------------------------------- *)
(* Subterms.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun subterm lit path = mlibAtom.subterm (atom lit) path;

fun subterms lit = mlibAtom.subterms (atom lit);

fun replace (lit as (pol,atm)) path_tm =
    let
      val atm' = mlibAtom.replace atm path_tm
    in
      if mlibPortable.pointerEqual (atm,atm') then lit else (pol,atm')
    end;

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

fun freeIn v lit = mlibAtom.freeIn v (atom lit);

fun freeVars lit = mlibAtom.freeVars (atom lit);

(* ------------------------------------------------------------------------- *)
(* Substitutions.                                                            *)
(* ------------------------------------------------------------------------- *)

fun subst sub (lit as (pol,atm)) : literal =
    let
      val atm' = mlibAtom.subst sub atm
    in
      if mlibPortable.pointerEqual (atm',atm) then lit else (pol,atm')
    end;

(* ------------------------------------------------------------------------- *)
(* Matching.                                                                 *)
(* ------------------------------------------------------------------------- *)

fun match sub ((pol1,atm1) : literal) (pol2,atm2) =
    let
      val _ = pol1 = pol2 orelse raise Error "mlibLiteral.match"
    in
      mlibAtom.match sub atm1 atm2
    end;

(* ------------------------------------------------------------------------- *)
(* Unification.                                                              *)
(* ------------------------------------------------------------------------- *)

fun unify sub ((pol1,atm1) : literal) (pol2,atm2) =
    let
      val _ = pol1 = pol2 orelse raise Error "mlibLiteral.unify"
    in
      mlibAtom.unify sub atm1 atm2
    end;

(* ------------------------------------------------------------------------- *)
(* The equality relation.                                                    *)
(* ------------------------------------------------------------------------- *)

fun mkEq l_r : literal = (true, mlibAtom.mkEq l_r);

fun destEq ((true,atm) : literal) = mlibAtom.destEq atm
  | destEq (false,_) = raise Error "mlibLiteral.destEq";

val isEq = can destEq;

fun mkNeq l_r : literal = (false, mlibAtom.mkEq l_r);

fun destNeq ((false,atm) : literal) = mlibAtom.destEq atm
  | destNeq (true,_) = raise Error "mlibLiteral.destNeq";

val isNeq = can destNeq;

fun mkRefl tm = (true, mlibAtom.mkRefl tm);

fun destRefl (true,atm) = mlibAtom.destRefl atm
  | destRefl (false,_) = raise Error "mlibLiteral.destRefl";

val isRefl = can destRefl;

fun mkIrrefl tm = (false, mlibAtom.mkRefl tm);

fun destIrrefl (true,_) = raise Error "mlibLiteral.destIrrefl"
  | destIrrefl (false,atm) = mlibAtom.destRefl atm;

val isIrrefl = can destIrrefl;

fun sym (pol,atm) : literal = (pol, mlibAtom.sym atm);

fun lhs ((_,atm) : literal) = mlibAtom.lhs atm;

fun rhs ((_,atm) : literal) = mlibAtom.rhs atm;

(* ------------------------------------------------------------------------- *)
(* Special support for terms with type annotations.                          *)
(* ------------------------------------------------------------------------- *)

fun typedSymbols ((_,atm) : literal) = mlibAtom.typedSymbols atm;

fun nonVarTypedSubterms ((_,atm) : literal) = mlibAtom.nonVarTypedSubterms atm;

(* ------------------------------------------------------------------------- *)
(* Parsing and pretty-printing.                                              *)
(* ------------------------------------------------------------------------- *)

val pp = mlibPrint.ppMap toFormula mlibFormula.pp;

val toString = mlibPrint.toString pp;

fun fromString s = fromFormula (mlibFormula.fromString s);

val parse = mlibParse.parseQuotation mlibTerm.toString fromString;

end

structure mlibLiteralOrdered =
struct type t = mlibLiteral.literal val compare = mlibLiteral.compare end

structure mlibLiteralMap = mlibKeyMap (mlibLiteralOrdered);

structure mlibLiteralSet =
struct

  local
    structure S = mlibElementSet (mlibLiteralMap);
  in
    open S;
  end;

  fun negateMember lit set = member (mlibLiteral.negate lit) set;

  val negate =
      let
        fun f (lit,set) = add set (mlibLiteral.negate lit)
      in
        foldl f empty
      end;

  val relations =
      let
        fun f (lit,set) = mlibNameAritySet.add set (mlibLiteral.relation lit)
      in
        foldl f mlibNameAritySet.empty
      end;

  val functions =
      let
        fun f (lit,set) = mlibNameAritySet.union set (mlibLiteral.functions lit)
      in
        foldl f mlibNameAritySet.empty
      end;

  fun freeIn v = exists (mlibLiteral.freeIn v);

  val freeVars =
      let
        fun f (lit,set) = mlibNameSet.union set (mlibLiteral.freeVars lit)
      in
        foldl f mlibNameSet.empty
      end;

  val freeVarsList =
      let
        fun f (lits,set) = mlibNameSet.union set (freeVars lits)
      in
        List.foldl f mlibNameSet.empty
      end;

  val symbols =
      let
        fun f (lit,z) = mlibLiteral.symbols lit + z
      in
        foldl f 0
      end;

  val typedSymbols =
      let
        fun f (lit,z) = mlibLiteral.typedSymbols lit + z
      in
        foldl f 0
      end;

  fun subst sub lits =
      let
        fun substLit (lit,(eq,lits')) =
            let
              val lit' = mlibLiteral.subst sub lit
              val eq = eq andalso mlibPortable.pointerEqual (lit,lit')
            in
              (eq, add lits' lit')
            end

        val (eq,lits') = foldl substLit (true,empty) lits
      in
        if eq then lits else lits'
      end;

  fun conjoin set =
      mlibFormula.listMkConj (List.map mlibLiteral.toFormula (toList set));

  fun disjoin set =
      mlibFormula.listMkDisj (List.map mlibLiteral.toFormula (toList set));

  val pp =
      mlibPrint.ppMap
        toList
        (mlibPrint.ppBracket "{" "}" (mlibPrint.ppOpList "," mlibLiteral.pp));

end

structure mlibLiteralSetOrdered =
struct type t = mlibLiteralSet.set val compare = mlibLiteralSet.compare end

structure mlibLiteralSetMap = mlibKeyMap (mlibLiteralSetOrdered);

structure mlibLiteralSetSet = mlibElementSet (mlibLiteralSetMap);
(* ========================================================================= *)
(* A LOGICAL KERNEL FOR FIRST ORDER CLAUSAL THEOREMS                         *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

signature mlibThm =
sig

(* ------------------------------------------------------------------------- *)
(* An abstract type of first order logic theorems.                           *)
(* ------------------------------------------------------------------------- *)

type thm

(* ------------------------------------------------------------------------- *)
(* Theorem destructors.                                                      *)
(* ------------------------------------------------------------------------- *)

type clause = mlibLiteralSet.set

datatype inferenceType =
    Axiom
  | Assume
  | mlibSubst
  | Factor
  | Resolve
  | Refl
  | Equality

type inference = inferenceType * thm list

val clause : thm -> clause

val inference : thm -> inference

(* Tautologies *)

val isTautology : thm -> bool

(* Contradictions *)

val isContradiction : thm -> bool

(* Unit theorems *)

val destUnit : thm -> mlibLiteral.literal

val isUnit : thm -> bool

(* Unit equality theorems *)

val destUnitEq : thm -> mlibTerm.term * mlibTerm.term

val isUnitEq : thm -> bool

(* Literals *)

val member : mlibLiteral.literal -> thm -> bool

val negateMember : mlibLiteral.literal -> thm -> bool

(* ------------------------------------------------------------------------- *)
(* A total order.                                                            *)
(* ------------------------------------------------------------------------- *)

val compare : thm * thm -> order

val equal : thm -> thm -> bool

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

val freeIn : mlibTerm.var -> thm -> bool

val freeVars : thm -> mlibNameSet.set

(* ------------------------------------------------------------------------- *)
(* Pretty-printing.                                                          *)
(* ------------------------------------------------------------------------- *)

val ppInferenceType : inferenceType mlibPrint.pp

val inferenceTypeToString : inferenceType -> string

val pp : thm mlibPrint.pp

val toString : thm -> string

(* ------------------------------------------------------------------------- *)
(* Primitive rules of inference.                                             *)
(* ------------------------------------------------------------------------- *)

(* ------------------------------------------------------------------------- *)
(*                                                                           *)
(* ----- axiom C                                                             *)
(*   C                                                                       *)
(* ------------------------------------------------------------------------- *)

val axiom : clause -> thm

(* ------------------------------------------------------------------------- *)
(*                                                                           *)
(* ----------- assume L                                                      *)
(*   L \/ ~L                                                                 *)
(* ------------------------------------------------------------------------- *)

val assume : mlibLiteral.literal -> thm

(* ------------------------------------------------------------------------- *)
(*    C                                                                      *)
(* -------- subst s                                                          *)
(*   C[s]                                                                    *)
(* ------------------------------------------------------------------------- *)

val subst : mlibSubst.subst -> thm -> thm

(* ------------------------------------------------------------------------- *)
(*   L \/ C    ~L \/ D                                                       *)
(* --------------------- resolve L                                           *)
(*        C \/ D                                                             *)
(*                                                                           *)
(* The literal L must occur in the first theorem, and the literal ~L must    *)
(* occur in the second theorem.                                              *)
(* ------------------------------------------------------------------------- *)

val resolve : mlibLiteral.literal -> thm -> thm -> thm

(* ------------------------------------------------------------------------- *)
(*                                                                           *)
(* --------- refl t                                                          *)
(*   t = t                                                                   *)
(* ------------------------------------------------------------------------- *)

val refl : mlibTerm.term -> thm

(* ------------------------------------------------------------------------- *)
(*                                                                           *)
(* ------------------------ equality L p t                                   *)
(*   ~(s = t) \/ ~L \/ L'                                                    *)
(*                                                                           *)
(* where s is the subterm of L at path p, and L' is L with the subterm at    *)
(* path p being replaced by t.                                               *)
(* ------------------------------------------------------------------------- *)

val equality : mlibLiteral.literal -> mlibTerm.path -> mlibTerm.term -> thm

end
(* ========================================================================= *)
(* A LOGICAL KERNEL FOR FIRST ORDER CLAUSAL THEOREMS                         *)
(* Copyright (c) 2001 Joe Hurd, distributed under the MIT license            *)
(* ========================================================================= *)

structure mlibThm :> mlibThm =
struct

open mlibUseful;

(* ------------------------------------------------------------------------- *)
(* An abstract type of first order logic theorems.                           *)
(* ------------------------------------------------------------------------- *)

type clause = mlibLiteralSet.set;

datatype inferenceType =
    Axiom
  | Assume
  | mlibSubst
  | Factor
  | Resolve
  | Refl
  | Equality;

datatype thm = mlibThm of clause * (inferenceType * thm list);

type inference = inferenceType * thm list;

(* ------------------------------------------------------------------------- *)
(* Theorem destructors.                                                      *)
(* ------------------------------------------------------------------------- *)

fun clause (mlibThm (cl,_)) = cl;

fun inference (mlibThm (_,inf)) = inf;

(* Tautologies *)

local
  fun chk (_,NONE) = NONE
    | chk ((pol,atm), SOME set) =
      if (pol andalso mlibAtom.isRefl atm) orelse mlibAtomSet.member atm set then NONE
      else SOME (mlibAtomSet.add set atm);
in
  fun isTautology th =
      case mlibLiteralSet.foldl chk (SOME mlibAtomSet.empty) (clause th) of
        SOME _ => false
      | NONE => true;
end;

(* Contradictions *)

fun isContradiction th = mlibLiteralSet.null (clause th);

(* Unit theorems *)

fun destUnit (mlibThm (cl,_)) =
    if mlibLiteralSet.size cl = 1 then mlibLiteralSet.pick cl
    else raise Error "mlibThm.destUnit";

val isUnit = can destUnit;

(* Unit equality theorems *)

fun destUnitEq th = mlibLiteral.destEq (destUnit th);

val isUnitEq = can destUnitEq;

(* Literals *)

fun member lit (mlibThm (cl,_)) = mlibLiteralSet.member lit cl;

fun negateMember lit (mlibThm (cl,_)) = mlibLiteralSet.negateMember lit cl;

(* ------------------------------------------------------------------------- *)
(* A total order.                                                            *)
(* ------------------------------------------------------------------------- *)

fun compare (th1,th2) = mlibLiteralSet.compare (clause th1, clause th2);

fun equal th1 th2 = mlibLiteralSet.equal (clause th1) (clause th2);

(* ------------------------------------------------------------------------- *)
(* Free variables.                                                           *)
(* ------------------------------------------------------------------------- *)

fun freeIn v (mlibThm (cl,_)) = mlibLiteralSet.freeIn v cl;

fun freeVars (mlibThm (cl,_)) = mlibLiteralSet.freeVars cl;

(* ------------------------------------------------------------------------- *)
(* Pretty-printing.                                                          *)
(* ------------------------------------------------------------------------- *)

fun inferenceTypeToString Axiom = "Axiom"
  | inferenceTypeToString Assume = "Assume"
  | inferenceTypeToString mlibSubst = "mlibSubst"
  | inferenceTypeToString Factor = "Factor"
  | inferenceTypeToString Resolve = "Resolve"
  | inferenceTypeToString Refl = "Refl"
  | inferenceTypeToString Equality = "Equality";

fun ppInferenceType inf =
    mlibPrint.ppString (inferenceTypeToString inf);

local
  fun toFormula th =
      mlibFormula.listMkDisj
        (List.map mlibLiteral.toFormula (mlibLiteralSet.toList (clause th)));
in
  fun pp th =
      mlibPrint.inconsistentBlock 3
        [mlibPrint.ppString "|- ",
         mlibFormula.pp (toFormula th)];
end;

val toString = mlibPrint.toString pp;

(* ------------------------------------------------------------------------- *)
(* Primitive rules of inference.                                             *)
(* ------------------------------------------------------------------------- *)

(* ------------------------------------------------------------------------- *)
(*                                                                           *)
(* ----- axiom C                                                             *)
(*   C                                                                       *)
(* ------------------------------------------------------------------------- *)

fun axiom cl = mlibThm (cl,(Axiom,[]));

(* ------------------------------------------------------------------------- *)
(*                                                                           *)
(* ----------- assume L                                                      *)
(*   L \/ ~L                                                                 *)
(* ------------------------------------------------------------------------- *)

fun assume lit =
    mlibThm (mlibLiteralSet.fromList [lit, mlibLiteral.negate lit], (Assume,[]));

(* ------------------------------------------------------------------------- *)
(*    C                                                                      *)
(* -------- subst s                                                          *)
(*   C[s]                                                                    *)
(* ------------------------------------------------------------------------- *)

fun subst sub (th as mlibThm (cl,inf)) =
    let
      val cl' = mlibLiteralSet.subst sub cl
    in
      if mlibPortable.pointerEqual (cl,cl') then th
      else
        case inf of
          (mlibSubst,_) => mlibThm (cl',inf)
        | _ => mlibThm (cl',(mlibSubst,[th]))
    end;

(* ------------------------------------------------------------------------- *)
(*   L \/ C    ~L \/ D                                                       *)
(* --------------------- resolve L                                           *)
(*        C \/ D                                                             *)
(*                                                                           *)
(* The literal L must occur in the first theorem, and the literal ~L must    *)
(* occur in the second theorem.                                              *)
(* ------------------------------------------------------------------------- *)

fun resolve lit (th1 as mlibThm (cl1,_)) (th2 as mlibThm (cl2,_)) =
    let
      val cl1' = mlibLiteralSet.delete cl1 lit
      and cl2' = mlibLiteralSet.delete cl2 (mlibLiteral.negate lit)
    in
      mlibThm (mlibLiteralSet.union cl1' cl2', (Resolve,[th1,th2]))
    end;

(*MetisDebug
val resolve = fn lit => fn pos => fn neg =>
    resolve lit pos neg
    handle Error err =>
      raise Error ("mlibThm.resolve:\nlit = " ^ mlibLiteral.toString lit ^
                   "\npos = " ^ toString pos ^
                   "\nneg = " ^ toString neg ^ "\n" ^ err);
*)

(* ------------------------------------------------------------------------- *)
(*                                                                           *)
(* --------- refl t                                                          *)
(*   t = t                                                                   *)
(* ------------------------------------------------------------------------- *)

fun refl tm = mlibThm (mlibLiteralSet.singleton (true, mlibAtom.mkRefl tm), (Refl,[]));

(* ------------------------------------------------------------------------- *)
(*                                                                           *)
(* ------------------------ equality L p t                                   *)
(*   ~(s = t) \/ ~L \/ L'                                                    *)
(*                                                                           *)
(* where s is the subterm of L at path p, and L' is L with the subterm at    *)
(* path p being replaced by t.                                               *)
(* ------------------------------------------------------------------------- *)

fun equality lit path t =
    let
      val s = mlibLiteral.subterm lit path

      val lit' = mlibLiteral.replace lit (path,t)

      val eqLit = mlibLiteral.mkNeq (s,t)

      val cl = mlibLiteralSet.fromList [eqLit, mlibLiteral.negate lit, lit']
    in
      mlibThm (cl,(Equality,[]))
    end;

end
