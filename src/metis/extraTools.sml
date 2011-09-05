structure extraTools :> extraTools =
struct
  val trace_level = ref 1

  local
    val generator = ref 0
  in
    fun new_int () = let val n = !generator val () = generator := n + 1 in n end;
  end;

end
