class BasicObject
  #
  #  call-seq:
  #     obj == other        -> true or false
  #     obj.equal?(other)   -> true or false
  #     obj.eql?(other)     -> true or false
  #
  #  Equality --- At the Object level, #== returns <code>true</code>
  #  only if +obj+ and +other+ are the same object.  Typically, this
  #  method is overridden in descendant classes to provide
  #  class-specific meaning.
  #
  #  Unlike #==, the #equal? method should never be overridden by
  #  subclasses as it is used to determine object identity (that is,
  #  <code>a.equal?(b)</code> if and only if <code>a</code> is the same
  #  object as <code>b</code>):
  #
  #    obj = "a"
  #    other = obj.dup
  #
  #    obj == other      #=> true
  #    obj.equal? other  #=> false
  #    obj.equal? obj    #=> true
  #
  #  The #eql? method returns <code>true</code> if +obj+ and +other+
  #  refer to the same hash key.  This is used by Hash to test members
  #  for equality.  For any pair of objects where #eql? returns +true+,
  #  the #hash value of both objects must be equal. So any subclass
  #  that overrides #eql? should also override #hash appropriately.
  #
  #  For objects of class Object, #eql?  is synonymous
  #  with #==.  Subclasses normally continue this tradition by aliasing
  #  #eql? to their overridden #== method, but there are exceptions.
  #  Numeric types, for example, perform type conversion across #==,
  #  but not across #eql?, so:
  #
  #     1 == 1.0     #=> true
  #     1.eql? 1.0   #=> false
  #--
  # \private
  #++
  #
  def ==(obj)
    Primitive.attr! 'inline'
    Primitive.cexpr! 'rb_obj_equal(self, obj)'
  end
end
