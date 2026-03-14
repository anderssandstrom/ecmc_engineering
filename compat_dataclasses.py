"""Small dataclass compatibility layer for Python 3.6 environments."""

try:
    from dataclasses import dataclass, field
except ImportError:
    _MISSING = object()

    class _Field(object):
        def __init__(self, default=_MISSING, default_factory=_MISSING):
            self.default = default
            self.default_factory = default_factory

    def field(*, default=_MISSING, default_factory=_MISSING):
        if default is not _MISSING and default_factory is not _MISSING:
            raise ValueError("cannot specify both default and default_factory")
        return _Field(default=default, default_factory=default_factory)

    def dataclass(_cls=None, **_kwargs):
        def wrap(cls):
            annotations = getattr(cls, "__annotations__", {})
            field_names = []
            field_defaults = {}
            field_factories = {}

            for name in annotations:
                value = getattr(cls, name, _MISSING)
                field_names.append(name)
                if isinstance(value, _Field):
                    if value.default is not _MISSING:
                        field_defaults[name] = value.default
                    if value.default_factory is not _MISSING:
                        field_factories[name] = value.default_factory
                    delattr(cls, name)
                elif value is not _MISSING:
                    field_defaults[name] = value

            def __init__(self, *args, **kwargs):
                if len(args) > len(field_names):
                    raise TypeError(
                        "__init__() takes at most {} positional arguments ({} given)".format(
                            len(field_names), len(args)
                        )
                    )

                for index, name in enumerate(field_names):
                    if index < len(args):
                        value = args[index]
                    elif name in kwargs:
                        value = kwargs.pop(name)
                    elif name in field_factories:
                        value = field_factories[name]()
                    elif name in field_defaults:
                        value = field_defaults[name]
                    else:
                        raise TypeError("missing required argument: {!r}".format(name))
                    setattr(self, name, value)

                if kwargs:
                    unexpected = sorted(kwargs.keys())[0]
                    raise TypeError("got an unexpected keyword argument {!r}".format(unexpected))

            def __repr__(self):
                values = ", ".join(
                    "{}={!r}".format(name, getattr(self, name)) for name in field_names
                )
                return "{}({})".format(cls.__name__, values)

            def __eq__(self, other):
                if self.__class__ is not other.__class__:
                    return False
                return all(getattr(self, name) == getattr(other, name) for name in field_names)

            cls.__init__ = __init__
            cls.__repr__ = __repr__
            cls.__eq__ = __eq__
            return cls

        if _cls is None:
            return wrap
        return wrap(_cls)
