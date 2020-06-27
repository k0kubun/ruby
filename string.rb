class String
  #
  #  call-seq:
  #     str.to_s     -> str
  #
  #  Returns +self+.
  #
  #  If called on a subclass of String, converts the receiver to a String object.
  #
  def to_s
    Primitive.attr! 'inline'
    Primitive.cexpr! 'rb_str_to_s(self)'
  end
end
