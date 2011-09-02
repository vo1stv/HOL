structure Portable =
struct
  fun critical f () = f ();
  val pointerEqual = HOLPortable.pointer_eq
  fun randomBool _ = raise (Fail "Unimplemented: randomBool")
  fun randomInt  _ = raise (Fail "Unimplemented: randomInt")
  fun randomReal _ = raise (Fail "Unimplemented: randomReal")
  fun randomWord _ = raise (Fail "Unimplemented: randomWord")
  fun time       _ = raise (Fail "Unimplemented: randomWord")
end
