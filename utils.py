class Monad[T]:
    def __init__(self, value: T, value_on_error: Any = None):
        self.value = value
        self.value_on_error = value_on_error

    def bind(self, function: Callable[[T], T]):
        try:
            new_monad = Monad(function(self.value))
        except Exception:
            self.value = self.value_on_error

        if self.value == self.value_on_error:
            return self

        def ibind_error(*args, **kwargs):
            raise AttributeError("Cannot use ibind after bind.")

        new_monad.ibind = ibind_error

        return new_monad

    def ibind(self, function: Callable[[T], T]):
        try:
            self.value = function(self.value)
        except Exception:
            self.value = self.value_on_error

        return self

    @staticmethod
    def unit(value: T) -> "Monad[T]":
        return Monad(value)

    def __repr__(self):
        return f"{type(self).__name__}({self.value!r})"
