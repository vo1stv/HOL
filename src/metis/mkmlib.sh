gilith="$HOME/gilith"
b="$gilith/basic/src"
m="$gilith/metis/src"
cat\
  $b/Portable.sig\
  ./PortableHOL.ML\
  $b/Useful.s{ig,ml}\
  $b/Lazy.s{ig,ml}\
  $b/Stream.s{ig,ml}\
  $b/Ordered.s{ig,ml}\
  $b/KeyMap.s{ig,ml}\
  $b/ElementSet.s{ig,ml}\
  $b/Print.s{ig,ml}\
  $m/Name.s{ig,ml}\
  $m/NameArity.s{ig,ml}\
  $b/Parse.s{ig,ml}\
  $m/Term.s{ig,ml}\
  $b/Map.s{ig,ml}\
  $b/Sharing.s{ig,ml}\
  $m/Subst.s{ig,ml}\
  $m/Atom.s{ig,ml}\
  $m/Formula.s{ig,ml}\
  $m/Literal.s{ig,ml}\
  $m/Thm.s{ig,ml}\
> mlib.sml
for s in `egrep "^(signature|structure|functor)" mlib.sml | awk '{ print $2 }' | sort | uniq`
do
  sed -i "s/\\b$s\\b/mlib$s/g" mlib.sml
done
sed -i "s/HOLPortable/Portable/1" mlib.sml
